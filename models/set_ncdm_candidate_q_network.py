"""Independent Set-conditioned C3DQN network for NCDM-native item selection."""
from __future__ import annotations
import torch
import torch.nn as nn
from models.ncdm_candidate_q_network import NEG_INF_Q
from models.set_attention import InducedSetAttentionBlock, MultiheadAttentionBlock

RELATIVE_FEATURE_NAMES = ["novelty_ratio", "covered_overlap_ratio", "mean_mastery_gap", "weakness_targeting", "concept_count_norm"]


class SetConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim: int, d_model: int = 128, n_heads: int = 4, num_history_layers: int = 2, dropout: float = 0.1, *, candidate_set_encoder: str = "isab", num_set_layers: int = 1, num_inducing_points: int = 16, set_attention_heads: int | None = None, use_relative_features: bool = True, set_pool_in_value_head: bool = True, full_attention_max_candidates: int = 512, debug_mode: bool = False) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.candidate_set_encoder = candidate_set_encoder
        self.num_set_layers = int(num_set_layers)
        self.num_inducing_points = int(num_inducing_points)
        self.set_attention_heads = int(set_attention_heads or n_heads)
        self.use_relative_features = bool(use_relative_features)
        self.relative_feature_names = list(RELATIVE_FEATURE_NAMES) if self.use_relative_features else []
        self.relative_feature_dim = 5 if self.use_relative_features else 0
        self.set_pool_in_value_head = bool(set_pool_in_value_head)
        self.full_attention_max_candidates = int(full_attention_max_candidates)
        self.debug_mode = bool(debug_mode)
        self.history_projector = nn.Linear(self.history_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_history_layers)
        self.candidate_projector = nn.Linear(self.candidate_feature_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_projector = nn.Linear(self.global_feature_dim, d_model)
        self.cognitive_projector = nn.Linear(3 * self.knowledge_dim, d_model)
        self.relative_projector = nn.Linear(self.relative_feature_dim, d_model) if self.use_relative_features else None
        if candidate_set_encoder == "none":
            self.set_layers = nn.ModuleList()
        elif candidate_set_encoder == "full_self_attention":
            self.set_layers = nn.ModuleList([MultiheadAttentionBlock(d_model, self.set_attention_heads, dropout) for _ in range(self.num_set_layers)])
        elif candidate_set_encoder == "isab":
            self.set_layers = nn.ModuleList([InducedSetAttentionBlock(d_model, self.set_attention_heads, self.num_inducing_points, dropout) for _ in range(self.num_set_layers)])
        else:
            raise ValueError("candidate_set_encoder must be none, full_self_attention, or isab")
        value_dim = 2 * d_model + (d_model if self.set_pool_in_value_head else 0)
        self.value_head = nn.Sequential(nn.Linear(value_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.advantage_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(self, history_features, history_mask, candidate_features, candidate_mask, global_features) -> None:
        b, _, hf = history_features.shape; bc, _, cf = candidate_features.shape
        if b != bc or global_features.shape != (b, self.global_feature_dim) or hf != self.history_feature_dim or cf != self.candidate_feature_dim:
            raise ValueError("Set-C3DQN batch dimensions are inconsistent")
        if not (history_mask.any(dim=1).all() and candidate_mask.any(dim=1).all()):
            raise ValueError("each sample must have at least one valid history and candidate")

    def _relative_features(self, candidate_features, global_features, coverage_count):
        k = self.knowledge_dim
        candidate_q_mask = candidate_features[:, :, :k]
        candidate_masked_difficulty = candidate_features[:, :, k:2*k]
        mastery = global_features[:, :k]
        if coverage_count is None:
            raise ValueError("Set-C3DQN requires explicit coverage_count")
        coverage_count = coverage_count.to(candidate_features.device).float()
        concept_count = candidate_q_mask.sum(-1, keepdim=True).clamp_min(1.0)
        novelty_ratio = (candidate_q_mask * (coverage_count.unsqueeze(1) == 0).float()).sum(-1, keepdim=True) / concept_count
        covered_overlap_ratio = (candidate_q_mask * (coverage_count.unsqueeze(1) > 0).float()).sum(-1, keepdim=True) / concept_count
        mean_mastery_gap = (torch.abs(mastery.unsqueeze(1) - candidate_masked_difficulty) * candidate_q_mask).sum(-1, keepdim=True) / concept_count
        weakness_targeting = ((1.0 - mastery).unsqueeze(1) * candidate_q_mask).sum(-1, keepdim=True) / concept_count
        concept_count_norm = candidate_q_mask.sum(-1, keepdim=True) / float(k)
        return torch.cat([novelty_ratio, covered_overlap_ratio, mean_mastery_gap, weakness_targeting, concept_count_norm], dim=-1)

    def _encode_history(self, history_features, history_mask):
        return self.history_encoder(self.history_projector(history_features), src_key_padding_mask=~history_mask.bool())

    def _local_candidates(self, encoded_history, history_mask, candidate_features, global_features, coverage_count):
        k = self.knowledge_dim
        cand_emb = self.candidate_projector(candidate_features)
        ctx, _ = self.cross_attention(cand_emb, encoded_history, encoded_history, key_padding_mask=~history_mask.bool(), need_weights=False)
        mastery = global_features[:, :k]
        qmask = candidate_features[:, :, :k]
        diff = candidate_features[:, :, k:2*k]
        cognitive = self.cognitive_projector(torch.cat([mastery.unsqueeze(1) * qmask, (1.0 - mastery).unsqueeze(1) * qmask, mastery.unsqueeze(1) * qmask - diff], dim=-1))
        local = cand_emb + ctx + self.global_projector(global_features).unsqueeze(1) + cognitive
        rel = self._relative_features(candidate_features, global_features, coverage_count) if self.use_relative_features else None
        if rel is not None:
            local = local + self.relative_projector(rel)
        return local, ctx, rel

    def _apply_set_encoder(self, local, candidate_mask):
        if self.candidate_set_encoder == "full_self_attention" and local.shape[1] > self.full_attention_max_candidates:
            raise ValueError("full_self_attention candidate count exceeds full_attention_max_candidates")
        out = local
        for layer in self.set_layers:
            if isinstance(layer, InducedSetAttentionBlock):
                out = layer(out, candidate_mask)
            else:
                out = layer(out, out, key_padding_mask=~candidate_mask.bool())
        return out

    def _finish(self, encoded_history, history_mask, candidate_mask, global_features, set_aware, ctx):
        masked_hist = encoded_history * history_mask.unsqueeze(-1).float()
        hist_pool = masked_hist.sum(1) / history_mask.sum(1, keepdim=True).clamp_min(1).float()
        value_parts = [hist_pool, self.global_projector(global_features)]
        if self.set_pool_in_value_head:
            m = candidate_mask.unsqueeze(-1).float()
            value_parts.append((set_aware * m).sum(1) / m.sum(1).clamp_min(1.0))
        value = self.value_head(torch.cat(value_parts, -1))
        adv = self.advantage_head(torch.cat([set_aware, ctx], -1)).squeeze(-1)
        valid = candidate_mask.bool()
        mean_adv = adv.masked_fill(~valid, 0.0).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1).float()
        q = (value + adv - mean_adv).masked_fill(~valid, NEG_INF_Q)
        return q, value, adv, mean_adv

    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, coverage_count=None, return_attention: bool = False):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        encoded = self._encode_history(history_features, history_mask)
        local, ctx, rel = self._local_candidates(encoded, history_mask, candidate_features, global_features, coverage_count)
        set_aware = self._apply_set_encoder(local, candidate_mask)
        q, value, adv, mean_adv = self._finish(encoded, history_mask, candidate_mask, global_features, set_aware, ctx)
        self.last_debug = {"value": value.detach(), "advantage": adv.detach(), "masked_mean_advantage": mean_adv.detach(), "relative_features": rel.detach() if rel is not None else torch.empty(0), "set_aware": set_aware.detach()}
        return q, None

    def forward_chunked(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, coverage_count, chunk_size: int = 128):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        encoded = self._encode_history(history_features, history_mask)
        locals_, ctxs = [], []
        for start in range(0, candidate_features.shape[1], int(chunk_size)):
            end = min(candidate_features.shape[1], start + int(chunk_size))
            loc, ctx, _ = self._local_candidates(encoded, history_mask, candidate_features[:, start:end], global_features, coverage_count)
            locals_.append(loc); ctxs.append(ctx)
        local = torch.cat(locals_, 1); ctx = torch.cat(ctxs, 1)
        set_aware = self._apply_set_encoder(local, candidate_mask)
        q, value, adv, mean_adv = self._finish(encoded, history_mask, candidate_mask, global_features, set_aware, ctx)
        return q, None
