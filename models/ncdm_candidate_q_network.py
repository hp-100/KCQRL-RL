"""Candidate-conditioned dueling Q-network for NCDM-native adaptive testing."""
from __future__ import annotations
import warnings
import torch
import torch.nn as nn

NEG_INF_Q = -1.0e9

class CandidateConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim: int, d_model: int = 128, n_heads: int = 4, num_history_layers: int = 2, dropout: float = 0.1, candidate_set_encoder: str = "isab", num_set_layers: int = 1, num_inducing_points: int = 16, full_attention_max_candidates: int = 128, debug_mode: bool = False) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.candidate_set_encoder = str(candidate_set_encoder or "isab")
        self.full_attention_max_candidates = int(full_attention_max_candidates)
        self.debug_mode = bool(debug_mode)
        self.history_projector = nn.Linear(self.history_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_history_layers)
        self.candidate_projector = nn.Linear(self.candidate_feature_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_projector = nn.Linear(self.global_feature_dim, d_model)
        self.inducing_points = nn.Parameter(torch.randn(int(num_inducing_points), d_model) * 0.02)
        self.induce_attn = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(int(num_set_layers))])
        self.candidate_set_attn = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(int(num_set_layers))])
        self.full_set_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True) for _ in range(int(num_set_layers))])
        self.value_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.advantage_head = nn.Sequential(nn.Linear(4 * d_model + 3 * self.knowledge_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(self, history_features, history_mask, candidate_features, candidate_mask, global_features) -> None:
        b, _, hf = history_features.shape; bc, _, cf = candidate_features.shape
        if b != bc or history_mask.shape[:1] != (b,) or candidate_mask.shape[:1] != (b,) or global_features.shape != (b, self.global_feature_dim):
            raise ValueError("C3DQN batch dimensions are inconsistent")
        if hf != self.history_feature_dim or cf != self.candidate_feature_dim:
            raise ValueError(f"feature dims must be history={self.history_feature_dim}, candidate={self.candidate_feature_dim}")
        if not (history_mask.any(dim=1).all() and candidate_mask.any(dim=1).all()):
            raise ValueError("each sample must have at least one valid history and candidate")

    def _set_encode(self, candidate_context: torch.Tensor, candidate_mask: torch.Tensor, summary: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        valid = candidate_mask.bool()
        if self.candidate_set_encoder == "full_self_attention":
            if candidate_context.shape[1] > self.full_attention_max_candidates:
                raise RuntimeError(f"full_self_attention candidate count {candidate_context.shape[1]} exceeds full_attention_max_candidates={self.full_attention_max_candidates}; use ISAB or reduce prefilter_top_k")
            warnings.warn("full_self_attention is intended only for small-scale ablations", RuntimeWarning)
            out = candidate_context
            for layer in self.full_set_layers:
                out = layer(out, src_key_padding_mask=~valid)
            return out, out
        if self.candidate_set_encoder != "isab":
            raise ValueError(f"unknown candidate_set_encoder={self.candidate_set_encoder!r}")
        induced = summary
        if induced is None:
            induced = self.inducing_points.unsqueeze(0).expand(candidate_context.shape[0], -1, -1)
            for attn in self.induce_attn:
                induced, _ = attn(induced, candidate_context, candidate_context, key_padding_mask=~valid, need_weights=False)
        out = candidate_context
        for attn in self.candidate_set_attn:
            out, _ = attn(out, induced, induced, need_weights=False)
        return out, induced

    def _score_candidates(self, candidate_embeddings, candidate_context, set_context, candidate_features, candidate_mask, global_features, encoded_history, history_mask):
        global_embedding = self.global_projector(global_features)
        masked_hist = encoded_history * history_mask.unsqueeze(-1).float()
        pooled = masked_hist.sum(dim=1) / history_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        value = self.value_head(torch.cat([pooled, global_embedding], dim=-1))
        mastery = global_features[:, :self.knowledge_dim]
        candidate_q_mask = candidate_features[:, :, :self.knowledge_dim]
        candidate_masked_difficulty = candidate_features[:, :, self.knowledge_dim:2*self.knowledge_dim]
        mastered = mastery.unsqueeze(1) * candidate_q_mask
        weakness = (1.0 - mastery).unsqueeze(1) * candidate_q_mask
        difficulty_gap = mastered - candidate_masked_difficulty
        global_b = global_embedding.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        adv_in = torch.cat([candidate_embeddings, candidate_context, set_context, global_b, mastered, weakness, difficulty_gap], dim=-1)
        advantage = self.advantage_head(adv_in).squeeze(-1)
        valid = candidate_mask.bool()
        mean_adv = advantage.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        q_values = (value + advantage - mean_adv).masked_fill(~valid, NEG_INF_Q)
        return q_values, value, advantage, mean_adv, mastered, weakness, difficulty_gap

    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, return_attention: bool = False):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        if not all(torch.isfinite(x).all() for x in (history_features, candidate_features, global_features)):
            raise ValueError("non-finite C3DQN input")
        key_padding_mask = ~history_mask.bool()
        hist_emb = self.history_projector(history_features)
        encoded_history = self.history_encoder(hist_emb, src_key_padding_mask=key_padding_mask)
        candidate_embeddings = self.candidate_projector(candidate_features)
        candidate_context, attn = self.cross_attention(candidate_embeddings, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=return_attention)
        set_context, _ = self._set_encode(candidate_context, candidate_mask)
        q_values, value, advantage, mean_adv, mastered, weakness, difficulty_gap = self._score_candidates(candidate_embeddings, candidate_context, set_context, candidate_features, candidate_mask, global_features, encoded_history, history_mask)
        if not torch.isfinite(q_values).all():
            raise ValueError("non-finite q_values")
        self.last_debug = {}
        if self.debug_mode or return_attention:
            self.last_debug = {"value": value.detach(), "advantage": advantage.detach(), "masked_mean_advantage": mean_adv.detach()}
            if self.debug_mode or return_attention:
                self.last_debug.update({"mastered": mastered.detach(), "weakness": weakness.detach(), "difficulty_gap": difficulty_gap.detach(), "candidate_context": candidate_context.detach(), "set_context": set_context.detach()})
        return q_values, (attn if return_attention else None)

    @torch.no_grad()
    def forward_chunked(self, history_features, history_mask, candidate_features, candidate_mask, global_features, candidate_chunk_size: int = 256):
        if self.candidate_set_encoder != "isab" or candidate_features.shape[1] <= int(candidate_chunk_size):
            return self.forward(history_features, history_mask, candidate_features, candidate_mask, global_features)[0]
        key_padding_mask = ~history_mask.bool()
        encoded_history = self.history_encoder(self.history_projector(history_features), src_key_padding_mask=key_padding_mask)
        all_emb = self.candidate_projector(candidate_features)
        all_ctx, _ = self.cross_attention(all_emb, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=False)
        _, induced = self._set_encode(all_ctx, candidate_mask)
        outs = []
        for start in range(0, candidate_features.shape[1], int(candidate_chunk_size)):
            stop = start + int(candidate_chunk_size)
            chunk_ctx = all_ctx[:, start:stop]
            chunk_set, _ = self._set_encode(chunk_ctx, candidate_mask[:, start:stop], summary=induced)
            q, *_ = self._score_candidates(all_emb[:, start:stop], chunk_ctx, chunk_set, candidate_features[:, start:stop], candidate_mask[:, start:stop], global_features, encoded_history, history_mask)
            outs.append(q)
        return torch.cat(outs, dim=1)
