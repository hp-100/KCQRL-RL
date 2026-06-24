"""Set-attention blocks for Set-C3DQN."""
from __future__ import annotations
import torch
import torch.nn as nn


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(query + self.dropout(out))
        y = self.ffn(x)
        return self.norm2(x + self.dropout(y))


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_inducing_points: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing_points, d_model) * 0.02)
        self.induce = MultiheadAttentionBlock(d_model, n_heads, dropout)
        self.project = MultiheadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, local_candidates: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        b = local_candidates.shape[0]
        inducing = self.inducing_points.unsqueeze(0).expand(b, -1, -1)
        h = self.induce(inducing, local_candidates, key_padding_mask=~candidate_mask.bool())
        return self.project(local_candidates, h)
