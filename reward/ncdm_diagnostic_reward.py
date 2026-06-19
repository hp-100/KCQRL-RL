"""Diagnostic-quality reward for C3DQN-NCDM."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class NCDMDiagnosticRewardConfig:
    prediction_weight: float = 1.0
    diagnosis_weight: float = 0.2
    coverage_weight: float = 0.05
    prediction_scale: float = 10.0
    reward_clip: float = 5.0

@dataclass(frozen=True)
class NCDMDiagnosticReward:
    total: float
    prediction_gain: float
    diagnosis_gain: float
    coverage_gain: float


def mastery_entropy(mastery: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Binary entropy diagnosis-confidence proxy; not a true mastery-error metric."""
    m = mastery.float().clamp(eps, 1.0 - eps)
    ent = -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m)) / math.log(2.0)
    return ent.mean()


def compute_ncdm_diagnostic_reward(query_nll_before: float, query_nll_after: float, mastery_before: torch.Tensor, mastery_after: torch.Tensor, selected_q_mask: torch.Tensor, coverage_count_before: torch.Tensor, config: NCDMDiagnosticRewardConfig | None = None) -> NCDMDiagnosticReward:
    cfg = config or NCDMDiagnosticRewardConfig()
    prediction_gain = float(torch.clamp(torch.tensor(cfg.prediction_scale * (query_nll_before - query_nll_after)), -1.0, 1.0).item())
    diagnosis_gain = float((mastery_entropy(mastery_before) - mastery_entropy(mastery_after)).item())
    q_mask = selected_q_mask.float()
    new_concepts = q_mask * (coverage_count_before.float() == 0).float()
    coverage_gain = float((new_concepts.sum() / q_mask.sum().clamp_min(1.0)).item())
    total = cfg.prediction_weight * prediction_gain + cfg.diagnosis_weight * diagnosis_gain + cfg.coverage_weight * coverage_gain
    total = float(torch.clamp(torch.tensor(total), -cfg.reward_clip, cfg.reward_clip).item())
    if not math.isfinite(total):
        raise ValueError("non-finite NCDM diagnostic reward")
    return NCDMDiagnosticReward(total, prediction_gain, diagnosis_gain, coverage_gain)
