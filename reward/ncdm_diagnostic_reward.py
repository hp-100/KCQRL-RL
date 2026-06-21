"""Diagnostic-quality rewards for Base/Set C3DQN-NCDM.

The original reward is retained as ``legacy`` for checkpoint compatibility. Two
simpler ablations are also available:

``prediction``
    Optimize only the held-out query NLL improvement.

``prediction_coverage``
    Optimize query NLL improvement plus a small marginal concept-coverage bonus.

The Double-DQN target and network architecture are intentionally unchanged so the
ablation isolates the effect of reward design.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


SUPPORTED_REWARD_MODES = {
    "legacy",
    "prediction",
    "prediction_coverage",
}


@dataclass(frozen=True)
class NCDMDiagnosticRewardConfig:
    mode: str = "legacy"
    prediction_weight: float = 1.0
    diagnosis_weight: float = 0.2
    coverage_weight: float = 0.05
    prediction_scale: float = 10.0
    reward_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                "reward mode must be one of: "
                + ", ".join(sorted(SUPPORTED_REWARD_MODES))
            )
        if self.prediction_scale <= 0:
            raise ValueError("prediction_scale must be positive")
        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")


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


def _prediction_gain(
    query_nll_before: float,
    query_nll_after: float,
    config: NCDMDiagnosticRewardConfig,
) -> float:
    scaled_gain = config.prediction_scale * (
        float(query_nll_before) - float(query_nll_after)
    )
    if config.mode == "legacy":
        transformed = torch.clamp(torch.tensor(scaled_gain), -1.0, 1.0)
    else:
        transformed = torch.tanh(torch.tensor(scaled_gain))
    return float(transformed.item())


def compute_ncdm_diagnostic_reward(
    query_nll_before: float,
    query_nll_after: float,
    mastery_before: torch.Tensor,
    mastery_after: torch.Tensor,
    selected_q_mask: torch.Tensor,
    coverage_count_before: torch.Tensor,
    config: NCDMDiagnosticRewardConfig | None = None,
) -> NCDMDiagnosticReward:
    cfg = config or NCDMDiagnosticRewardConfig()
    prediction_gain = _prediction_gain(query_nll_before, query_nll_after, cfg)
    diagnosis_gain = float(
        (mastery_entropy(mastery_before) - mastery_entropy(mastery_after)).item()
    )

    q_mask = selected_q_mask.float()
    new_concepts = q_mask * (coverage_count_before.float() == 0).float()
    coverage_gain = float(
        (new_concepts.sum() / q_mask.sum().clamp_min(1.0)).item()
    )

    if cfg.mode == "legacy":
        total = (
            cfg.prediction_weight * prediction_gain
            + cfg.diagnosis_weight * diagnosis_gain
            + cfg.coverage_weight * coverage_gain
        )
    elif cfg.mode == "prediction":
        total = cfg.prediction_weight * prediction_gain
    elif cfg.mode == "prediction_coverage":
        total = (
            cfg.prediction_weight * prediction_gain
            + cfg.coverage_weight * coverage_gain
        )
    else:  # pragma: no cover - guarded by config validation
        raise AssertionError(f"unsupported reward mode: {cfg.mode}")

    total = float(
        torch.clamp(torch.tensor(total), -cfg.reward_clip, cfg.reward_clip).item()
    )
    if not math.isfinite(total):
        raise ValueError("non-finite NCDM diagnostic reward")
    return NCDMDiagnosticReward(
        total,
        prediction_gain,
        diagnosis_gain,
        coverage_gain,
    )
