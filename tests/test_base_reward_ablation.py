from __future__ import annotations

import math

import pytest
import torch

from reward.ncdm_diagnostic_reward import (
    NCDMDiagnosticRewardConfig,
    compute_ncdm_diagnostic_reward,
)
from scripts.make_base_reward_ablation_configs import (
    build_reward_ablation_configs,
)


def _reward(config: NCDMDiagnosticRewardConfig):
    return compute_ncdm_diagnostic_reward(
        query_nll_before=0.60,
        query_nll_after=0.50,
        mastery_before=torch.tensor([0.50, 0.50]),
        mastery_after=torch.tensor([0.80, 0.50]),
        selected_q_mask=torch.tensor([1.0, 1.0]),
        coverage_count_before=torch.tensor([0.0, 1.0]),
        config=config,
    )


def test_legacy_reward_remains_backward_compatible() -> None:
    reward = _reward(NCDMDiagnosticRewardConfig())
    expected_prediction = 1.0
    expected = expected_prediction + 0.2 * reward.diagnosis_gain + 0.05 * 0.5
    assert reward.prediction_gain == pytest.approx(expected_prediction)
    assert reward.coverage_gain == pytest.approx(0.5)
    assert reward.total == pytest.approx(expected)


def test_prediction_reward_uses_smooth_tanh_and_ignores_other_components() -> None:
    config = NCDMDiagnosticRewardConfig(
        mode="prediction",
        prediction_scale=5.0,
        diagnosis_weight=999.0,
        coverage_weight=999.0,
    )
    reward = _reward(config)
    assert reward.prediction_gain == pytest.approx(math.tanh(0.5))
    assert reward.total == pytest.approx(reward.prediction_gain)


def test_prediction_coverage_adds_only_small_marginal_coverage_bonus() -> None:
    config = NCDMDiagnosticRewardConfig(
        mode="prediction_coverage",
        prediction_scale=5.0,
        coverage_weight=0.02,
        diagnosis_weight=999.0,
    )
    reward = _reward(config)
    assert reward.total == pytest.approx(math.tanh(0.5) + 0.02 * 0.5)


def test_invalid_reward_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward mode"):
        NCDMDiagnosticRewardConfig(mode="not-a-mode")


def test_reward_ablation_configs_change_only_reward_and_output() -> None:
    configs = build_reward_ablation_configs(
        q_matrix="q.pt",
        ncdm_checkpoint="ncdm.pt",
        train_valid_sequences="train.csv",
        output_root="outputs/reward",
        max_students=200,
        epochs=3,
        selection_horizon=10,
        seed=42,
        device="cuda",
        use_amp=True,
        top_k=64,
    )
    prediction = configs["prediction"]
    coverage = configs["prediction_coverage"]

    assert prediction["model"] == coverage["model"]
    assert prediction["candidate_pool"] == coverage["candidate_pool"]
    assert {
        key: value
        for key, value in prediction["training"].items()
        if key != "output_dir"
    } == {
        key: value
        for key, value in coverage["training"].items()
        if key != "output_dir"
    }
    assert prediction["model"]["architecture"] == "base_c3dqn"
    assert prediction["reward"]["mode"] == "prediction"
    assert coverage["reward"]["mode"] == "prediction_coverage"
    assert coverage["reward"]["coverage_weight"] == pytest.approx(0.02)
