"""Candidate-conditioned dueling Q-network for NCDM-native adaptive testing."""
from __future__ import annotations

import torch
import torch.nn as nn

NEG_INF_Q = -1.0e9


class CandidateConditionedNCDMQNetwork(nn.Module):
    """Base C3DQN network without candidate-set interaction."""

    def __init__(
        self,
        knowledge_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_history_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.history_projector = nn.Linear(self.history_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_history_layers,
        )
        self.candidate_projector = nn.Linear(self.candidate_feature_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_projector = nn.Linear(self.global_feature_dim, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(3 * d_model + 3 * self.knowledge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        global_features: torch.Tensor,
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
        if not history_mask.bool().any(dim=1).all():
            raise ValueError("each sample requires at least one valid history item")
        if not candidate_mask.bool().any(dim=1).all():
            raise ValueError("each sample requires at least one valid candidate")

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        global_features: torch.Tensor,
        return_attention: bool = False,
        coverage_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ``pad_c3dqn_batch`` exposes exact coverage for Set-C3DQN.  Accept and
        # deliberately ignore it here so legacy Base-C3DQN callers can pass a
        # shared batch dictionary without changing the Base architecture.
        del coverage_count
        self._assert_shapes(
            history_features,
            history_mask,
            candidate_features,
            candidate_mask,
            global_features,
        )
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (history_features, candidate_features, global_features)
        ):
            raise ValueError("non-finite C3DQN input")

        key_padding_mask = ~history_mask.bool()
        encoded_history = self.history_encoder(
            self.history_projector(history_features),
            src_key_padding_mask=key_padding_mask,
        )
        candidate_embeddings = self.candidate_projector(candidate_features)
        candidate_context, attention = self.cross_attention(
            candidate_embeddings,
            encoded_history,
            encoded_history,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )
        global_embedding = self.global_projector(global_features)
        masked_history = encoded_history * history_mask.unsqueeze(-1).to(
            encoded_history.dtype
        )
        pooled_history = masked_history.sum(dim=1) / history_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1).to(encoded_history.dtype)
        value = self.value_head(
            torch.cat([pooled_history, global_embedding], dim=-1)
        )

        mastery = global_features[:, : self.knowledge_dim]
        candidate_q_mask = candidate_features[:, :, : self.knowledge_dim]
        candidate_masked_difficulty = candidate_features[
            :, :, self.knowledge_dim : 2 * self.knowledge_dim
        ]
        mastered = mastery.unsqueeze(1) * candidate_q_mask
        weakness = (1.0 - mastery).unsqueeze(1) * candidate_q_mask
        difficulty_gap = mastered - candidate_masked_difficulty
        global_broadcast = global_embedding.unsqueeze(1).expand(
            -1,
            candidate_features.shape[1],
            -1,
        )
        advantage_inputs = torch.cat(
            [
                candidate_embeddings,
                candidate_context,
                global_broadcast,
                mastered,
                weakness,
                difficulty_gap,
            ],
            dim=-1,
        )
        advantage = self.advantage_head(advantage_inputs).squeeze(-1)
        valid = candidate_mask.bool()
        mean_advantage = (
            advantage.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True)
            / valid.sum(dim=1, keepdim=True).clamp_min(1).to(advantage.dtype)
        )
        q_values = (value + advantage - mean_advantage).masked_fill(
            ~valid,
            NEG_INF_Q,
        )
        if not torch.isfinite(q_values).all():
            raise ValueError("non-finite q_values")
        self.last_debug = {
            "value": value.detach(),
            "advantage": advantage.detach(),
            "masked_mean_advantage": mean_advantage.detach(),
            "mastered": mastered.detach(),
            "weakness": weakness.detach(),
            "difficulty_gap": difficulty_gap.detach(),
            "candidate_context": candidate_context.detach(),
        }
        return q_values, (attention if return_attention else None)
