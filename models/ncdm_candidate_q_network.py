"""Candidate-conditioned dueling Q-network for NCDM-native adaptive testing."""
from __future__ import annotations
import torch
import torch.nn as nn

NEG_INF_Q = -1.0e9

class CandidateConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim: int, d_model: int = 128, n_heads: int = 4, num_history_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.history_projector = nn.Linear(self.history_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_history_layers)
        self.candidate_projector = nn.Linear(self.candidate_feature_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_projector = nn.Linear(self.global_feature_dim, d_model)
        self.value_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.advantage_head = nn.Sequential(nn.Linear(3 * d_model + 3 * self.knowledge_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(self, history_features, history_mask, candidate_features, candidate_mask, global_features) -> None:
        b, _, hf = history_features.shape; bc, _, cf = candidate_features.shape
        if b != bc or history_mask.shape[:1] != (b,) or candidate_mask.shape[:1] != (b,) or global_features.shape != (b, self.global_feature_dim):
            raise ValueError("C3DQN batch dimensions are inconsistent")
        if hf != self.history_feature_dim or cf != self.candidate_feature_dim:
            raise ValueError(f"feature dims must be history={self.history_feature_dim}, candidate={self.candidate_feature_dim}")
        if not (history_mask.any(dim=1).all() and candidate_mask.any(dim=1).all()):
            raise ValueError("each sample must have at least one valid history and candidate")

    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, return_attention: bool = False):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        if not all(torch.isfinite(x).all() for x in (history_features, candidate_features, global_features)):
            raise ValueError("non-finite C3DQN input")
        key_padding_mask = ~history_mask.bool()
        hist_emb = self.history_projector(history_features)
        encoded_history = self.history_encoder(hist_emb, src_key_padding_mask=key_padding_mask)
        candidate_embeddings = self.candidate_projector(candidate_features)
        candidate_context, attn = self.cross_attention(candidate_embeddings, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=return_attention)
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
        adv_in = torch.cat([candidate_embeddings, candidate_context, global_b, mastered, weakness, difficulty_gap], dim=-1)
        advantage = self.advantage_head(adv_in).squeeze(-1)
        valid = candidate_mask.bool()
        masked_adv = advantage.masked_fill(~valid, 0.0)
        mean_adv = masked_adv.sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        q_values = value + advantage - mean_adv
        q_values = q_values.masked_fill(~valid, NEG_INF_Q)
        if not torch.isfinite(q_values).all():
            raise ValueError("non-finite q_values")
        self.last_debug = {"value": value.detach(), "advantage": advantage.detach(), "masked_mean_advantage": mean_adv.detach(), "mastered": mastered.detach(), "weakness": weakness.detach(), "difficulty_gap": difficulty_gap.detach(), "candidate_context": candidate_context.detach()}
        return q_values, (attn if return_attention else None)


class SetConditionedNCDMQNetwork(CandidateConditionedNCDMQNetwork):
    """Set-C3DQN variant with a real chunked forward path.

    Candidate-history attention and advantage scoring can be chunked. The set
    encoder placeholder still observes the complete Top-K local candidate
    representation, so Top-K prefiltering remains the primary memory control.
    """
    def __init__(self, knowledge_dim: int, d_model: int = 128, n_heads: int = 4, num_history_layers: int = 2, dropout: float = 0.1, **kwargs) -> None:
        super().__init__(knowledge_dim, d_model=d_model, n_heads=n_heads, num_history_layers=num_history_layers, dropout=dropout)
        self.set_config = dict(kwargs)

    def _encode_history_once(self, history_features, history_mask):
        return self.history_encoder(self.history_projector(history_features), src_key_padding_mask=~history_mask.bool())

    def forward_chunked(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, chunk_size: int = 128):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        chunk_size = max(1, int(chunk_size))
        encoded_history = self._encode_history_once(history_features, history_mask)
        key_padding_mask = ~history_mask.bool()
        candidate_embeddings = self.candidate_projector(candidate_features)
        local_chunks=[]; context_chunks=[]
        for start in range(0, candidate_features.shape[1], chunk_size):
            end = min(candidate_features.shape[1], start + chunk_size)
            emb = candidate_embeddings[:, start:end]
            ctx, _ = self.cross_attention(emb, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=False)
            local_chunks.append(emb)
            context_chunks.append(ctx)
        set_aware_all = torch.cat(local_chunks, dim=1)
        context_all = torch.cat(context_chunks, dim=1)
        global_embedding = self.global_projector(global_features)
        masked_hist = encoded_history * history_mask.unsqueeze(-1).float()
        pooled = masked_hist.sum(dim=1) / history_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        value = self.value_head(torch.cat([pooled, global_embedding], dim=-1))
        mastery = global_features[:, :self.knowledge_dim]
        qmask = candidate_features[:, :, :self.knowledge_dim]
        mdiff = candidate_features[:, :, self.knowledge_dim:2*self.knowledge_dim]
        mastered = mastery.unsqueeze(1) * qmask
        weakness = (1.0 - mastery).unsqueeze(1) * qmask
        difficulty_gap = mastered - mdiff
        raw=[]
        for start in range(0, candidate_features.shape[1], chunk_size):
            end = min(candidate_features.shape[1], start + chunk_size)
            gb = global_embedding.unsqueeze(1).expand(-1, end-start, -1)
            adv_in = torch.cat([set_aware_all[:, start:end], context_all[:, start:end], gb, mastered[:, start:end], weakness[:, start:end], difficulty_gap[:, start:end]], dim=-1)
            raw.append(self.advantage_head(adv_in).squeeze(-1))
        advantage = torch.cat(raw, dim=1)
        valid = candidate_mask.bool()
        mean_adv = advantage.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        q_values = (value + advantage - mean_adv).masked_fill(~valid, NEG_INF_Q)
        return q_values, None
