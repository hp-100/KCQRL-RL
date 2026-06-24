"""Student-conditioned Set-C3DQN Q-network for NCDM-native adaptive testing."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.ncdm_candidate_q_network import NEG_INF_Q
from models.set_attention import InducedSetAttentionBlock, MultiheadAttentionBlock

RELATIVE_FEATURE_NAMES = [
    "novelty_ratio",
    "covered_overlap_ratio",
    "mean_mastery_gap",
    "weakness_targeting",
    "concept_count_norm",
]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class SetConditionedNCDMQNetwork(nn.Module):
    """Dueling candidate Q-network with student-conditioned set interaction."""

    def __init__(
        self,
        knowledge_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_history_layers: int = 2,
        dropout: float = 0.1,
        candidate_set_encoder: str = "isab",
        num_set_layers: int = 1,
        num_inducing_points: int = 16,
        set_attention_heads: int | None = None,
        use_relative_features: bool = True,
        set_pool_in_value_head: bool = True,
        full_attention_max_candidates: int = 128,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.d_model = int(d_model)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.candidate_set_encoder = str(candidate_set_encoder)
        self.num_set_layers = int(num_set_layers)
        self.num_inducing_points = int(num_inducing_points)
        self.set_attention_heads = int(set_attention_heads or n_heads)
        self.use_relative_features = bool(use_relative_features)
        self.relative_feature_dim = 5 if self.use_relative_features else 0
        self.set_pool_in_value_head = bool(set_pool_in_value_head)
        self.full_attention_max_candidates = int(full_attention_max_candidates)
        self.debug_mode = bool(debug_mode)

        if self.candidate_set_encoder not in {
            "none",
            "full_self_attention",
            "isab",
        }:
            raise ValueError(
                "candidate_set_encoder must be one of: none, full_self_attention, isab"
            )
        if self.num_set_layers < 0:
            raise ValueError("num_set_layers must be non-negative")
        if self.candidate_set_encoder != "none" and self.num_set_layers == 0:
            raise ValueError("set encoders require num_set_layers >= 1")

        self.history_projector = nn.Linear(self.history_feature_dim, self.d_model)
        history_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=int(num_history_layers),
        )
        self.candidate_projector = nn.Linear(
            self.candidate_feature_dim,
            self.d_model,
        )
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.global_projector = nn.Linear(self.global_feature_dim, self.d_model)
        self.cognitive_projector = nn.Linear(
            3 * self.knowledge_dim + self.relative_feature_dim,
            self.d_model,
        )
        self.local_norm = nn.LayerNorm(self.d_model)

        layers: list[nn.Module] = []
        for _ in range(self.num_set_layers):
            if self.candidate_set_encoder == "isab":
                layers.append(
                    InducedSetAttentionBlock(
                        self.d_model,
                        self.set_attention_heads,
                        self.num_inducing_points,
                        dropout=float(dropout),
                    )
                )
            elif self.candidate_set_encoder == "full_self_attention":
                layers.append(
                    MultiheadAttentionBlock(
                        self.d_model,
                        self.set_attention_heads,
                        dropout=float(dropout),
                    )
                )
        self.set_layers = nn.ModuleList(layers)

        value_input_dim = 3 * self.d_model if self.set_pool_in_value_head else 2 * self.d_model
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(3 * self.d_model + self.relative_feature_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        global_features: torch.Tensor,
        coverage_count: torch.Tensor,
    ) -> None:
        batch_size = history_features.shape[0]
        if history_features.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("history_features and candidate_features must be rank-3")
        if history_features.shape[-1] != self.history_feature_dim:
            raise ValueError("invalid history feature dimension")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("invalid candidate feature dimension")
        if history_mask.shape != history_features.shape[:2]:
            raise ValueError("history mask shape mismatch")
        if candidate_mask.shape != candidate_features.shape[:2]:
            raise ValueError("candidate mask shape mismatch")
        if global_features.shape != (batch_size, self.global_feature_dim):
            raise ValueError("global feature shape mismatch")
        if coverage_count.shape != (batch_size, self.knowledge_dim):
            raise ValueError("coverage_count shape mismatch")
        if not history_mask.bool().any(dim=1).all():
            raise ValueError("each sample requires at least one valid history item")
        if not candidate_mask.bool().any(dim=1).all():
            raise ValueError("each sample requires at least one valid candidate")
        tensors = (
            history_features,
            candidate_features,
            global_features,
            coverage_count,
        )
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Set-C3DQN received non-finite inputs")

    def _encode_history(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~history_mask.bool()
        encoded = self.history_encoder(
            self.history_projector(history_features),
            src_key_padding_mask=key_padding_mask,
        )
        pooled = masked_mean(encoded, history_mask.bool())
        return encoded, pooled

    def _relative_features(
        self,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        coverage_count: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_relative_features:
            return candidate_features.new_zeros(
                (*candidate_features.shape[:2], 0)
            )

        k = self.knowledge_dim
        q_mask = candidate_features[:, :, :k]
        masked_difficulty = candidate_features[:, :, k : 2 * k]
        mastery = global_features[:, :k]
        denominator = q_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        novelty_ratio = (
            q_mask * (coverage_count.unsqueeze(1) == 0).to(q_mask.dtype)
        ).sum(dim=-1, keepdim=True) / denominator
        covered_overlap_ratio = (
            q_mask * (coverage_count.unsqueeze(1) > 0).to(q_mask.dtype)
        ).sum(dim=-1, keepdim=True) / denominator
        mean_mastery_gap = (
            (mastery.unsqueeze(1) - masked_difficulty).abs() * q_mask
        ).sum(dim=-1, keepdim=True) / denominator
        weakness_targeting = (
            (1.0 - mastery).unsqueeze(1) * q_mask
        ).sum(dim=-1, keepdim=True) / denominator
        concept_count_norm = q_mask.sum(dim=-1, keepdim=True) / float(k)

        relative = torch.cat(
            [
                novelty_ratio,
                covered_overlap_ratio,
                mean_mastery_gap,
                weakness_targeting,
                concept_count_norm,
            ],
            dim=-1,
        )
        return torch.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)

    def _candidate_interactions(
        self,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
        relative_features: torch.Tensor,
    ) -> torch.Tensor:
        k = self.knowledge_dim
        q_mask = candidate_features[:, :, :k]
        masked_difficulty = candidate_features[:, :, k : 2 * k]
        mastery = global_features[:, :k]
        mastered = mastery.unsqueeze(1) * q_mask
        weakness = (1.0 - mastery).unsqueeze(1) * q_mask
        difficulty_gap = mastered - masked_difficulty
        return torch.cat(
            [mastered, weakness, difficulty_gap, relative_features],
            dim=-1,
        )

    def _local_candidate_chunk(
        self,
        candidate_features: torch.Tensor,
        encoded_history: torch.Tensor,
        history_key_padding_mask: torch.Tensor,
        global_embedding: torch.Tensor,
        global_features: torch.Tensor,
        coverage_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_embedding = self.candidate_projector(candidate_features)
        candidate_context, _ = self.cross_attention(
            candidate_embedding,
            encoded_history,
            encoded_history,
            key_padding_mask=history_key_padding_mask,
            need_weights=False,
        )
        relative = self._relative_features(
            candidate_features,
            global_features,
            coverage_count,
        )
        cognitive = self.cognitive_projector(
            self._candidate_interactions(
                candidate_features,
                global_features,
                relative,
            )
        )
        local = self.local_norm(
            candidate_embedding
            + candidate_context
            + global_embedding.unsqueeze(1)
            + cognitive
        )
        return local, candidate_context, relative

    def _apply_set_encoder(
        self,
        local_candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_counts = candidate_mask.bool().sum(dim=1)
        if (
            self.candidate_set_encoder == "full_self_attention"
            and int(valid_counts.max().item()) > self.full_attention_max_candidates
        ):
            raise ValueError(
                "full candidate self-attention exceeds configured candidate limit"
            )

        output = local_candidates
        for layer in self.set_layers:
            if self.candidate_set_encoder == "isab":
                output = layer(output, candidate_mask.bool())
            elif self.candidate_set_encoder == "full_self_attention":
                output = layer(
                    output,
                    output,
                    output,
                    key_padding_mask=~candidate_mask.bool(),
                    query_mask=candidate_mask.bool(),
                )
        return output * candidate_mask.unsqueeze(-1).to(output.dtype)

    def _raw_advantage(
        self,
        local_candidates: torch.Tensor,
        set_aware_candidates: torch.Tensor,
        candidate_context: torch.Tensor,
        relative_features: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            [
                local_candidates,
                set_aware_candidates,
                candidate_context,
                relative_features,
            ],
            dim=-1,
        )
        return self.advantage_head(inputs).squeeze(-1)

    def _value(
        self,
        pooled_history: torch.Tensor,
        global_embedding: torch.Tensor,
        set_aware_candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = [pooled_history, global_embedding]
        if self.set_pool_in_value_head:
            inputs.append(masked_mean(set_aware_candidates, candidate_mask.bool()))
        return self.value_head(torch.cat(inputs, dim=-1))

    @staticmethod
    def _dueling_q(
        value: torch.Tensor,
        advantage: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = candidate_mask.bool()
        mean_advantage = (
            advantage.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True)
            / valid.sum(dim=1, keepdim=True).clamp_min(1).to(advantage.dtype)
        )
        q_values = (value + advantage - mean_advantage).masked_fill(
            ~valid,
            NEG_INF_Q,
        )
        return q_values, mean_advantage

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        global_features: torch.Tensor,
        coverage_count: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, None]:
        del return_attention
        self._assert_shapes(
            history_features,
            history_mask,
            candidate_features,
            candidate_mask,
            global_features,
            coverage_count,
        )
        encoded_history, pooled_history = self._encode_history(
            history_features,
            history_mask,
        )
        global_embedding = self.global_projector(global_features)
        local, context, relative = self._local_candidate_chunk(
            candidate_features,
            encoded_history,
            ~history_mask.bool(),
            global_embedding,
            global_features,
            coverage_count,
        )
        set_aware = self._apply_set_encoder(local, candidate_mask)
        value = self._value(
            pooled_history,
            global_embedding,
            set_aware,
            candidate_mask,
        )
        advantage = self._raw_advantage(local, set_aware, context, relative)
        q_values, mean_advantage = self._dueling_q(
            value,
            advantage,
            candidate_mask,
        )
        if not torch.isfinite(q_values).all():
            raise ValueError("Set-C3DQN produced non-finite q_values")

        if self.debug_mode:
            self.last_debug = {
                "relative_features": relative.detach(),
                "local_candidates": local.detach(),
                "set_aware_candidates": set_aware.detach(),
                "advantage": advantage.detach(),
                "masked_mean_advantage": mean_advantage.detach(),
            }
        else:
            self.last_debug = {}
        return q_values, None

    def forward_chunked(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        global_features: torch.Tensor,
        coverage_count: torch.Tensor,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, None]:
        self._assert_shapes(
            history_features,
            history_mask,
            candidate_features,
            candidate_mask,
            global_features,
            coverage_count,
        )
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        encoded_history, pooled_history = self._encode_history(
            history_features,
            history_mask,
        )
        history_key_padding_mask = ~history_mask.bool()
        global_embedding = self.global_projector(global_features)

        local_chunks: list[torch.Tensor] = []
        context_chunks: list[torch.Tensor] = []
        relative_chunks: list[torch.Tensor] = []
        candidate_count = candidate_features.shape[1]
        for start in range(0, candidate_count, chunk_size):
            end = min(candidate_count, start + chunk_size)
            local, context, relative = self._local_candidate_chunk(
                candidate_features[:, start:end],
                encoded_history,
                history_key_padding_mask,
                global_embedding,
                global_features,
                coverage_count,
            )
            local_chunks.append(local)
            context_chunks.append(context)
            relative_chunks.append(relative)

        local_all = torch.cat(local_chunks, dim=1)
        context_all = torch.cat(context_chunks, dim=1)
        relative_all = torch.cat(relative_chunks, dim=1)
        set_aware_all = self._apply_set_encoder(local_all, candidate_mask)
        value = self._value(
            pooled_history,
            global_embedding,
            set_aware_all,
            candidate_mask,
        )

        advantage_chunks: list[torch.Tensor] = []
        for start in range(0, candidate_count, chunk_size):
            end = min(candidate_count, start + chunk_size)
            advantage_chunks.append(
                self._raw_advantage(
                    local_all[:, start:end],
                    set_aware_all[:, start:end],
                    context_all[:, start:end],
                    relative_all[:, start:end],
                )
            )
        advantage = torch.cat(advantage_chunks, dim=1)
        q_values, mean_advantage = self._dueling_q(
            value,
            advantage,
            candidate_mask,
        )
        if not torch.isfinite(q_values).all():
            raise ValueError("Set-C3DQN produced non-finite q_values")

        if self.debug_mode:
            self.last_debug = {
                "relative_features": relative_all.detach(),
                "local_candidates": local_all.detach(),
                "set_aware_candidates": set_aware_all.detach(),
                "advantage": advantage.detach(),
                "masked_mean_advantage": mean_advantage.detach(),
            }
        else:
            self.last_debug = {}
        return q_values, None
