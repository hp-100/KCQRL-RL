"""Permutation-equivariant Set Attention blocks for Set-C3DQN-NCDM."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiheadAttentionBlock(nn.Module):
    """Residual multi-head attention block with FFN and layer normalization."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * d_model, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(query + self.dropout1(attended))
        output = self.norm2(hidden + self.dropout2(self.ffn(hidden)))
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output


class InducedSetAttentionBlock(nn.Module):
    """Set Transformer ISAB with learned inducing points."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_inducing_points <= 0:
            raise ValueError("num_inducing_points must be positive")
        self.num_inducing_points = int(num_inducing_points)
        self.inducing_points = nn.Parameter(
            torch.randn(self.num_inducing_points, d_model) * 0.02
        )
        self.inducing_to_candidates = MultiheadAttentionBlock(
            d_model,
            n_heads,
            dropout=dropout,
        )
        self.candidates_to_induced = MultiheadAttentionBlock(
            d_model,
            n_heads,
            dropout=dropout,
        )
        self.last_induced_summary: torch.Tensor | None = None

    def forward(
        self,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 3 or candidate_mask.shape != candidates.shape[:2]:
            raise ValueError("ISAB candidates and mask have inconsistent shapes")
        if not candidate_mask.bool().any(dim=1).all():
            raise ValueError("ISAB requires at least one valid candidate per sample")

        batch_size = candidates.shape[0]
        inducing = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)
        induced = self.inducing_to_candidates(
            inducing,
            candidates,
            candidates,
            key_padding_mask=~candidate_mask.bool(),
        )
        output = self.candidates_to_induced(
            candidates,
            induced,
            induced,
            query_mask=candidate_mask.bool(),
        )
        self.last_induced_summary = induced.detach()
        return output
