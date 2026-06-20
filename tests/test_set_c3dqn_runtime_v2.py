from __future__ import annotations

import torch
import pytest
from torch import nn

from agents.ncdm_c3dqn_trainer import C3DQNTransition
from agents.ncdm_c3dqn_trainer_v2 import (
    build_set_checkpoint_metadata,
    compute_double_dqn_loss,
    load_c3dqn_checkpoint,
    load_set_c3dqn_checkpoint,
    transitions_to_batches,
)
from evaluation.policies.c3dqn_ncdm_policy import SetC3DQNNCDMPolicy
from models.ncdm_candidate_features import NCDMItemFeatureCache
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork
from scripts.ncdm_c3dqn_app import build_q_network_from_config


class TinyNCDM(nn.Module):
    def __init__(self, item_count: int, knowledge_dim: int) -> None:
        super().__init__()
        self.knowledge_dim = knowledge_dim
        self.k_difficulty = nn.Embedding(item_count, knowledge_dim)
        self.e_discrimination = nn.Embedding(item_count, 1)

    def predict_with_alpha(
        self,
        alpha: torch.Tensor,
        item_ids: torch.Tensor,
        q_matrix: torch.Tensor,
    ) -> torch.Tensor:
        difficulty = torch.sigmoid(self.k_difficulty(item_ids))
        discrimination = torch.sigmoid(self.e_discrimination(item_ids)) * 3.0
        mastery = torch.sigmoid(alpha)
        return torch.sigmoid(
            (
                discrimination
                * (mastery - difficulty)
                * q_matrix[item_ids]
            ).sum(dim=-1)
        )


def make_assets(item_count: int = 12, knowledge_dim: int = 4):
    torch.manual_seed(7)
    q_matrix = torch.randint(0, 2, (item_count, knowledge_dim)).float()
    q_matrix[:, 0] = 1.0
    ncdm = TinyNCDM(item_count, knowledge_dim)
    cache = NCDMItemFeatureCache(ncdm, q_matrix, "cpu")
    return ncdm, q_matrix, cache


def make_model(knowledge_dim: int = 4, **kwargs):
    return SetConditionedNCDMQNetwork(
        knowledge_dim,
        d_model=16,
        n_heads=4,
        num_history_layers=1,
        dropout=0.0,
        candidate_set_encoder=kwargs.pop("candidate_set_encoder", "isab"),
        num_set_layers=kwargs.pop("num_set_layers", 1),
        num_inducing_points=kwargs.pop("num_inducing_points", 4),
        set_attention_heads=4,
        use_relative_features=True,
        set_pool_in_value_head=True,
        full_attention_max_candidates=kwargs.pop(
            "full_attention_max_candidates",
            8,
        ),
        **kwargs,
    )


def make_forward_inputs(batch_size: int = 2, candidates: int = 7, k: int = 4):
    history = torch.randn(batch_size, 3, 2 * k + 3)
    history_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    q_mask = torch.randint(0, 2, (batch_size, candidates, k)).float()
    q_mask[:, :, 0] = 1.0
    difficulty = torch.rand(batch_size, candidates, k) * q_mask
    discrimination = torch.rand(batch_size, candidates, 1)
    candidate_features = torch.cat(
        [q_mask, difficulty, discrimination],
        dim=-1,
    )
    candidate_mask = torch.ones(batch_size, candidates, dtype=torch.bool)
    mastery = torch.rand(batch_size, k)
    coverage = torch.rand(batch_size, k)
    step = torch.zeros(batch_size, 1)
    global_features = torch.cat([mastery, coverage, step], dim=-1)
    coverage_count = torch.randint(0, 3, (batch_size, k)).float()
    return (
        history,
        history_mask,
        candidate_features,
        candidate_mask,
        global_features,
        coverage_count,
    )


def test_build_q_network_dispatches_complete_set_model():
    network = build_q_network_from_config(
        {
            "architecture": "set_c3dqn",
            "d_model": 16,
            "n_heads": 4,
            "candidate_set_encoder": "isab",
            "num_inducing_points": 6,
        },
        4,
    )
    assert isinstance(network, SetConditionedNCDMQNetwork)
    assert network.set_layers
    assert network.num_inducing_points == 6


def test_full_and_chunked_forward_match():
    model = make_model().eval()
    inputs = make_forward_inputs()
    with torch.no_grad():
        full, _ = model(*inputs)
        chunked, _ = model.forward_chunked(*inputs, chunk_size=3)
    torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-6)
    assert torch.equal(full.argmax(dim=1), chunked.argmax(dim=1))


def test_full_attention_limit_is_enforced_before_attention():
    model = make_model(
        candidate_set_encoder="full_self_attention",
        full_attention_max_candidates=4,
    ).eval()
    inputs = make_forward_inputs(candidates=5)
    with pytest.raises(ValueError, match="exceeds configured candidate limit"):
        model(*inputs)


def test_prefilter_disabled_preserves_original_order():
    ncdm, q_matrix, cache = make_assets()
    prefilter = NCDMCandidatePrefilter(
        q_matrix=q_matrix,
        feature_cache=cache,
        ncdm=ncdm,
        config={"prefilter_enabled": False, "prefilter_top_k": 2},
    )
    original = [7, 2, 9, 1]
    filtered, summary = prefilter.select(
        original,
        torch.zeros(1, cache.knowledge_dim),
        torch.full((cache.knowledge_dim,), 0.5),
        torch.zeros(cache.knowledge_dim),
    )
    assert filtered == original
    assert summary["filtered_candidate_count"] == len(original)


def test_set_checkpoint_roundtrip_and_base_rejection(tmp_path):
    ncdm, q_matrix, _cache = make_assets()
    model = make_model().eval()
    model_config = {
        "architecture": "set_c3dqn",
        "d_model": 16,
        "n_heads": 4,
        "num_history_layers": 1,
        "dropout": 0.0,
        "candidate_set_encoder": "isab",
        "num_set_layers": 1,
        "num_inducing_points": 4,
        "set_attention_heads": 4,
        "use_relative_features": True,
        "set_pool_in_value_head": True,
        "full_attention_max_candidates": 8,
        "debug_mode": False,
    }
    metadata = build_set_checkpoint_metadata(
        knowledge_dim=4,
        selection_horizon=5,
        warm_start_items=1,
        alpha_fit={"initial_steps": 2, "incremental_steps": 1},
        reward_config={},
        model_config=model_config,
        candidate_pool_config={"prefilter_enabled": True, "prefilter_top_k": 8},
        ncdm_item_count=12,
        q_matrix_item_count=12,
        training_seed=1,
        validation_metrics={},
        epoch=1,
        strict_item_count_check=True,
        requested_amp=False,
        effective_amp=False,
    )
    checkpoint_path = tmp_path / "set.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        checkpoint_path,
    )
    loaded, loaded_metadata = load_set_c3dqn_checkpoint(
        checkpoint_path,
        ncdm=ncdm,
        q_matrix=q_matrix,
    )
    assert isinstance(loaded, SetConditionedNCDMQNetwork)
    assert loaded_metadata["actor_architecture"].startswith("set_conditioned")
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_c3dqn_checkpoint(
            checkpoint_path,
            ncdm=ncdm,
            q_matrix=q_matrix,
        )


def test_mixed_terminal_double_dqn_supports_set_model():
    _ncdm, _q_matrix, cache = make_assets()
    online = make_model()
    target = make_model()
    target.load_state_dict(online.state_dict())
    k = cache.knowledge_dim
    current_count = [0.0] * k
    next_count = [1.0] + [0.0] * (k - 1)
    transitions = [
        C3DQNTransition(
            [0],
            [1.0],
            [1, 2, 3],
            [0.5] * k,
            [0.0] * k,
            0,
            1,
            0.2,
            {},
            [0, 1],
            [1.0, 0.0],
            [2, 3],
            [0.55] * k,
            [0.2] * k,
            1,
            False,
            current_count,
            next_count,
        ),
        C3DQNTransition(
            [0],
            [1.0],
            [4, 5],
            [0.5] * k,
            [0.0] * k,
            0,
            4,
            0.1,
            {},
            [0, 4],
            [1.0, 1.0],
            [],
            [0.6] * k,
            [0.2] * k,
            1,
            True,
            current_count,
            next_count,
        ),
    ]
    batch, next_batch, rewards, dones, indices = transitions_to_batches(
        transitions,
        cache,
        5,
        require_exact_coverage=True,
    )
    loss, stats = compute_double_dqn_loss(
        online,
        target,
        batch,
        next_batch,
        rewards,
        dones,
        0.99,
        indices,
        chunk_size=2,
    )
    assert torch.isfinite(loss)
    assert stats["next_q_mean"] == pytest.approx(stats["next_q_mean"])


def test_set_policy_loads_set_network_and_selects_real_item(tmp_path):
    ncdm, q_matrix, cache = make_assets()
    model = make_model().eval()
    model_config = {
        "architecture": "set_c3dqn",
        "d_model": 16,
        "n_heads": 4,
        "num_history_layers": 1,
        "dropout": 0.0,
        "candidate_set_encoder": "isab",
        "num_set_layers": 1,
        "num_inducing_points": 4,
        "set_attention_heads": 4,
        "use_relative_features": True,
        "set_pool_in_value_head": True,
        "full_attention_max_candidates": 8,
        "debug_mode": False,
    }
    metadata = build_set_checkpoint_metadata(
        knowledge_dim=4,
        selection_horizon=5,
        warm_start_items=1,
        alpha_fit={"initial_steps": 1, "incremental_steps": 1, "lr": 0.01},
        reward_config={},
        model_config=model_config,
        candidate_pool_config={"prefilter_enabled": False},
        ncdm_item_count=12,
        q_matrix_item_count=12,
        training_seed=1,
        validation_metrics={},
        epoch=1,
        strict_item_count_check=True,
        requested_amp=False,
        effective_amp=False,
    )
    checkpoint_path = tmp_path / "set_policy.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        checkpoint_path,
    )
    policy = SetC3DQNNCDMPolicy(
        checkpoint_path,
        ncdm,
        q_matrix,
        cache=cache,
        candidate_chunk_size=2,
    )
    policy.reset("s1", 7, {})
    selected = policy.select([1, 2, 3], [0], [1.0], {"policy_step": 0})
    assert selected in {1, 2, 3}
    assert policy.last_prefiltered_candidate_ids == [1, 2, 3]
