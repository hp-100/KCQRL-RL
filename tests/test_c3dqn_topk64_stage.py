from __future__ import annotations

import pytest

from scripts.make_c3dqn_topk64_configs import build_paired_configs
from scripts.profile_c3dqn_ncdm_topk64 import _assert_shared_protocol


def test_paired_topk64_configs_share_training_and_candidate_pool():
    base, set_cfg = build_paired_configs(
        q_matrix="q.pt",
        ncdm_checkpoint="ncdm.pt",
        train_valid_sequences="sequences.csv",
        output_root="outputs/paired",
        max_students=20,
        epochs=1,
        selection_horizon=5,
        seed=42,
        device="cuda",
        use_amp=True,
        top_k=64,
    )

    assert base["model"]["architecture"] == "base_c3dqn"
    assert set_cfg["model"]["architecture"] == "set_c3dqn"
    assert set_cfg["model"]["candidate_set_encoder"] == "isab"
    assert base["candidate_pool"] == set_cfg["candidate_pool"]
    assert base["candidate_pool"]["prefilter_top_k"] == 64
    assert base["alpha_fit"] == set_cfg["alpha_fit"]
    assert base["paths"] == set_cfg["paths"]

    base_training = dict(base["training"])
    set_training = dict(set_cfg["training"])
    base_training.pop("output_dir")
    set_training.pop("output_dir")
    assert base_training == set_training


def test_profile_protocol_accepts_matching_topk64_metadata():
    common = {
        "knowledge_dim": 36,
        "selection_horizon": 5,
        "warm_start_items": 1,
        "q_matrix_item_count": 7652,
        "ncdm_item_count": 7652,
        "alpha_fit": {"initial_steps": 8, "incremental_steps": 3},
        "reward_config": {},
        "candidate_pool_config": {
            "prefilter_enabled": True,
            "prefilter_top_k": 64,
        },
    }
    assert _assert_shared_protocol(common, dict(common), top_k=64) == 5


def test_profile_protocol_rejects_unfair_candidate_pool():
    base = {
        "knowledge_dim": 36,
        "selection_horizon": 5,
        "warm_start_items": 1,
        "q_matrix_item_count": 7652,
        "ncdm_item_count": 7652,
        "alpha_fit": {},
        "reward_config": {},
        "candidate_pool_config": {
            "prefilter_enabled": True,
            "prefilter_top_k": 64,
        },
    }
    set_metadata = {
        **base,
        "candidate_pool_config": {
            "prefilter_enabled": True,
            "prefilter_top_k": 128,
        },
    }
    with pytest.raises(ValueError, match="Set checkpoint Top-K mismatch"):
        _assert_shared_protocol(base, set_metadata, top_k=64)
