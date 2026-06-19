"""Formal multidimensional IRT utilities for benchmark_v2 item selection."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIRTModel(nn.Module):
    """MIRT item-parameter container with the original checkpoint structure."""

    def __init__(self, n_students: int, n_items: int, n_dims: int):
        super().__init__()
        self.theta_emb = nn.Embedding(int(n_students), int(n_dims))
        self.disc_emb = nn.Embedding(int(n_items), int(n_dims))
        self.diff_emb = nn.Embedding(int(n_items), 1)
        self.n_students = int(n_students)
        self.n_items = int(n_items)
        self.n_dims = int(n_dims)

    def predict_with_theta(self, theta: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return predict_with_theta(self, theta, item_ids)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("MIRT checkpoint must be a state_dict or contain a state_dict key")
    return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}


def load_mirt_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> MIRTModel:
    """Load a MIRT checkpoint and infer student/item/dimension counts from tensors."""
    state = _state_dict_from_checkpoint(torch.load(Path(path), map_location=device))
    required = ("theta_emb.weight", "disc_emb.weight", "diff_emb.weight")
    missing = [k for k in required if k not in state]
    if missing:
        raise KeyError(f"MIRT checkpoint missing required tensors: {missing}")
    theta_w, disc_w, diff_w = state["theta_emb.weight"], state["disc_emb.weight"], state["diff_emb.weight"]
    if theta_w.ndim != 2 or disc_w.ndim != 2 or diff_w.ndim != 2 or diff_w.shape[1] != 1:
        raise ValueError("Invalid MIRT checkpoint tensor shapes")
    if theta_w.shape[1] != disc_w.shape[1]:
        raise ValueError("MIRT theta and discrimination dimensions differ")
    if disc_w.shape[0] != diff_w.shape[0]:
        raise ValueError("MIRT discrimination and difficulty item counts differ")
    model = MIRTModel(theta_w.shape[0], disc_w.shape[0], disc_w.shape[1]).to(device)
    model.load_state_dict({k: state[k] for k in required}, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def predict_with_theta(model: MIRTModel, theta: torch.Tensor, item_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
    if not torch.is_tensor(item_ids):
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=theta.device)
    else:
        item_ids = item_ids.to(device=theta.device, dtype=torch.long)
    disc = model.disc_emb(item_ids)
    diff = model.diff_emb(item_ids)
    theta = theta.to(disc.device)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0).expand(disc.shape[0], -1)
    logit = (disc * theta).sum(dim=1, keepdim=True) - diff
    return torch.sigmoid(logit).squeeze(-1)


def fit_student_theta(
    model: MIRTModel,
    history_item_ids: Sequence[int],
    history_responses: Sequence[float],
    *,
    steps: int = 30,
    lr: float = 0.05,
    theta_l2: float = 0.01,
    grad_clip: float = 5.0,
    early_stop_tol: float = 1e-5,
    device: torch.device | str | None = None,
    return_losses: bool = False,
):
    """Fit an independent test-student theta from history while freezing item params."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    theta = torch.empty(model.n_dims, device=device, requires_grad=True)
    theta.data.normal_(0.0, 0.01, generator=g)
    item_ids = torch.tensor(list(history_item_ids), dtype=torch.long, device=device)
    responses = torch.tensor(list(history_responses), dtype=torch.float32, device=device)
    losses: list[float] = []
    if item_ids.numel() == 0:
        return (theta.detach(), losses) if return_losses else theta.detach()
    opt = torch.optim.Adam([theta], lr=float(lr))
    prev = None
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        preds = predict_with_theta(model, theta, item_ids).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(preds, responses) + float(theta_l2) * theta.pow(2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], float(grad_clip))
        opt.step()
        val = float(loss.detach().cpu())
        losses.append(val)
        if prev is not None and abs(prev - val) < float(early_stop_tol):
            break
        prev = val
    out = theta.detach()
    return (out, losses) if return_losses else out
