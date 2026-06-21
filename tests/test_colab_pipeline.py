from __future__ import annotations

import csv
import json

import pytest

from scripts.colab_pipeline import (
    _benchmark_required_outputs,
    _discover_test_sequence_candidates,
    _pilot_checkpoint_paths,
    _required_files_exist,
    _validate_step_zero,
)


def test_discover_test_sequences_prefers_exact_names_and_excludes_train_valid(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    exact = data_root / "test_question_window_sequences.csv"
    exact.write_text("student_id,item_id,response\n")
    other = data_root / "fold_test_sequences.csv"
    other.write_text("student_id,item_id,response\n")
    excluded = data_root / "train_valid_sequences.csv"
    excluded.write_text("student_id,item_id,response\n")

    candidates = _discover_test_sequence_candidates(data_root)

    assert candidates == [exact, other]
    assert excluded not in candidates


def test_pilot_checkpoint_paths_are_seed_specific(tmp_path):
    base, set_path = _pilot_checkpoint_paths(tmp_path, 43)
    assert base == tmp_path / "seed_43/base_c3dqn_topk64/best_checkpoint.pt"
    assert set_path == tmp_path / "seed_43/set_c3dqn_topk64/best_checkpoint.pt"


def test_required_benchmark_outputs_detect_completion(tmp_path):
    output_dir = tmp_path / "benchmark"
    output_dir.mkdir()
    required = _benchmark_required_outputs(output_dir)
    assert not _required_files_exist(required)
    for path in required:
        path.write_text("x")
    assert _required_files_exist(required)


def _write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_validate_step_zero_accepts_paired_metrics_and_non_privileged_policies(tmp_path):
    policies = ["Random-NCDM", "C3DQN-NCDM", "Set-C3DQN-NCDM"]
    rows = []
    for policy in policies:
        rows.append(
            {
                "policy": policy,
                "seed": 101,
                "step": 0,
                "evaluated_students": 20,
                "accuracy_micro": 0.8,
                "auc_micro": 0.7,
                "nll_micro": 0.4,
                "brier_micro": 0.12,
                "accuracy_macro": 0.8,
                "auc_macro": 0.7,
                "nll_macro": 0.4,
                "brier_macro": 0.12,
            }
        )
    _write_csv(tmp_path / "per_seed.csv", rows)
    (tmp_path / "policy_metadata.json").write_text(
        json.dumps(
            {
                policy: {"uses_privileged_information": False}
                for policy in policies
            }
        )
    )

    _validate_step_zero(tmp_path)


def test_validate_step_zero_rejects_metric_mismatch(tmp_path):
    rows = []
    for policy, auc in (
        ("Random-NCDM", 0.7),
        ("C3DQN-NCDM", 0.7),
        ("Set-C3DQN-NCDM", 0.71),
    ):
        rows.append(
            {
                "policy": policy,
                "seed": 101,
                "step": 0,
                "evaluated_students": 20,
                "accuracy_micro": 0.8,
                "auc_micro": auc,
                "nll_micro": 0.4,
                "brier_micro": 0.12,
                "accuracy_macro": 0.8,
                "auc_macro": 0.7,
                "nll_macro": 0.4,
                "brier_macro": 0.12,
            }
        )
    _write_csv(tmp_path / "per_seed.csv", rows)
    (tmp_path / "policy_metadata.json").write_text(
        json.dumps(
            {
                policy: {"uses_privileged_information": False}
                for policy in ("Random-NCDM", "C3DQN-NCDM", "Set-C3DQN-NCDM")
            }
        )
    )

    with pytest.raises(ValueError, match="step-0 mismatch"):
        _validate_step_zero(tmp_path)
