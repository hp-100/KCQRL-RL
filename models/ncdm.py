"""Neural Cognitive Diagnosis Model utilities extracted from legacy notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim


class NoneNegClipper:
    """Clamp prediction network weights to the non-negative range used by NCDM."""

    def __call__(self, module: nn.Module) -> None:
        if hasattr(module, "weight"):
            module.weight.data.add_(torch.relu(torch.neg(module.weight.data)))


class OfficialNCDM(nn.Module):
    """Official NCDM architecture from the legacy training scripts."""

    def __init__(self, student_n: int, exer_n: int, knowledge_n: int):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.prednet_len1 = 512
        self.prednet_len2 = 256
        self.student_emb = nn.Embedding(student_n, knowledge_n)
        self.k_difficulty = nn.Embedding(exer_n, knowledge_n)
        self.e_discrimination = nn.Embedding(exer_n, 1)
        self.prednet_full1 = nn.Linear(knowledge_n, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id: torch.Tensor, exer_id: torch.Tensor, kn_emb: torch.Tensor) -> torch.Tensor:
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10.0
        x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        x = self.drop_1(torch.sigmoid(self.prednet_full1(x)))
        x = self.drop_2(torch.sigmoid(self.prednet_full2(x)))
        return torch.sigmoid(self.prednet_full3(x)).squeeze(-1)

    def predict_with_alpha(self, alpha: torch.Tensor, exer_id: torch.Tensor, q_matrix: torch.Tensor) -> torch.Tensor:
        diff = torch.sigmoid(self.k_difficulty(exer_id))
        disc = torch.sigmoid(self.e_discrimination(exer_id)) * 10.0
        x = disc * (torch.sigmoid(alpha) - diff) * q_matrix[exer_id]
        x = self.drop_1(torch.sigmoid(self.prednet_full1(x)))
        x = self.drop_2(torch.sigmoid(self.prednet_full2(x)))
        return torch.sigmoid(self.prednet_full3(x)).squeeze(-1)

    def apply_clipper(self) -> None:
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


def load_q_matrix(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    q_matrix = torch.load(Path(path), map_location=device)
    if not torch.is_tensor(q_matrix):
        q_matrix = torch.as_tensor(q_matrix, dtype=torch.float32, device=device)
    return q_matrix.float().to(device)


def safe_load_ncdm_checkpoint(model: OfficialNCDM, path: str | Path, device: torch.device | str = "cpu") -> tuple[list[str], list[str]]:
    """Load a checkpoint while dropping mismatched student embeddings."""
    checkpoint = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = dict(checkpoint)
    own = model.state_dict()
    if "student_emb.weight" in checkpoint and checkpoint["student_emb.weight"].shape != own["student_emb.weight"].shape:
        del checkpoint["student_emb.weight"]
    incompatible = model.load_state_dict(checkpoint, strict=False)
    model.apply_clipper()
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def fit_student_alpha(model: OfficialNCDM, q_matrix: torch.Tensor, history_i: Sequence[int], history_r: Sequence[float], *, steps: int = 8, lr: float = 0.05, device=None) -> torch.Tensor:
    device = device or q_matrix.device
    if len(history_i) == 0:
        return torch.zeros((1, model.knowledge_dim), device=device)
    items = torch.tensor(history_i, dtype=torch.long, device=device)
    responses = torch.tensor(history_r, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean_diff = torch.sigmoid(model.k_difficulty(items)).mean(dim=0).clamp(1e-4, 1 - 1e-4)
        init = torch.log(mean_diff / (1.0 - mean_diff))
    alpha = init.unsqueeze(0).detach().clone().requires_grad_(True)
    opt = optim.Adam([alpha], lr=lr)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    for step_idx in range(steps):
        opt.zero_grad()
        loss = nn.BCELoss()(model.predict_with_alpha(alpha, items, q_matrix), responses)
        loss.backward()
        opt.step()
        if step_idx >= 5 and loss.item() < 1e-3:
            break
    model.eval()
    return alpha.detach()


def predict_remaining(model: OfficialNCDM, q_matrix: torch.Tensor, history_i: Sequence[int], history_r: Sequence[float], target_i: Sequence[int], *, device=None) -> torch.Tensor:
    device = device or q_matrix.device
    if not target_i:
        return torch.empty(0, device=device)
    alpha = fit_student_alpha(model, q_matrix, history_i, history_r, device=device)
    with torch.no_grad():
        return model.predict_with_alpha(alpha, torch.tensor(target_i, dtype=torch.long, device=device), q_matrix)
