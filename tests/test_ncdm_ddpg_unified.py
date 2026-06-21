from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from evaluation.policies.ddpg_policy import (
    NCDMDDPGPolicy,
    load_lstm_actor_checkpoint,
)
from models.actor import LSTMActor
from scripts.run_ncdm_ddpg_unified_benchmark import (
    discover_c3dqn_checkpoints,
    discover_ddpg_checkpoint,
)


class TinyNCDM(nn.Module):
    def __init__(self, item_count: int, knowledge_dim: int):
        super().__init__()
        self.k_difficulty = nn.Embedding(item_count, knowledge_dim)
        self.e_discrimination = nn.Embedding(item_count, 1)
        nn.init.zeros_(self.k_difficulty.weight)
        nn.init.zeros_(self.e_discrimination.weight)


class FixedIdealActor(nn.Module):
    action_dim = 5
    hidden_dim = 3

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "ideal",
            torch.tensor([[0.0, 1.0, 0.5, 0.5, 0.5]]),
        )

    def init_hidden(self, batch_size: int, device):
        return (
            torch.zeros((batch_size, self.hidden_dim), device=device),
            torch.zeros((batch_size, self.hidden_dim), device=device),
        )

    def forward(self, semantic, q_mask, difficulty, discrimination, response, hx, cx):
        del semantic, q_mask, difficulty, discrimination, response
        return self.ideal.to(hx.device), hx + 1.0, cx + 1.0


def test_ncdm_ddpg_selects_nearest_real_candidate(tmp_path: Path) -> None:
    q_matrix = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    item_bank = torch.eye(3, 4)
    policy = NCDMDDPGPolicy(
        tmp_path / "unused.pt",
        actor=FixedIdealActor(),
        q_matrix=q_matrix,
        item_bank=item_bank,
        ncdm=TinyNCDM(3, 2),
        device="cpu",
    )
    policy.reset("student", 101, {})
    selected = policy.select(
        [1, 2],
        [0],
        [1.0],
        {"policy_step": 0, "selection_horizon": 10},
    )
    assert selected == 2
    assert policy.name == "NCDM-DDPG"
    assert policy.metadata.evaluator_model == "NCDM"


def test_ncdm_ddpg_rejects_privileged_context(tmp_path: Path) -> None:
    policy = NCDMDDPGPolicy(
        tmp_path / "unused.pt",
        actor=FixedIdealActor(),
        q_matrix=torch.ones((3, 2)),
        item_bank=torch.ones((3, 4)),
        ncdm=TinyNCDM(3, 2),
        device="cpu",
    )
    policy.reset("student", 101, {})
    with pytest.raises(ValueError, match="privileged"):
        policy.select(
            [1, 2],
            [0],
            [1.0],
            {"query_item_ids": [2]},
        )


def test_lstm_actor_loader_accepts_raw_and_wrapped_state_dicts(tmp_path: Path) -> None:
    actor = LSTMActor(semantic_dim=4, q_dim=2)
    raw_path = tmp_path / "raw.pt"
    wrapped_path = tmp_path / "wrapped.pt"
    torch.save(actor.state_dict(), raw_path)
    torch.save({"actor_state_dict": actor.state_dict()}, wrapped_path)

    raw_loaded = load_lstm_actor_checkpoint(
        raw_path,
        semantic_dim=4,
        q_dim=2,
        device="cpu",
    )
    wrapped_loaded = load_lstm_actor_checkpoint(
        wrapped_path,
        semantic_dim=4,
        q_dim=2,
        device="cpu",
    )
    assert raw_loaded.action_dim == 5
    assert wrapped_loaded.action_dim == 5


def test_checkpoint_discovery_prefers_highest_ddpg_epoch(tmp_path: Path) -> None:
    metadata = tmp_path / "data/XES3G5M/metadata"
    metadata.mkdir(parents=True)
    low = metadata / "ddpg_enhanced_36d_actor_epoch2.pt"
    high = metadata / "ddpg_enhanced_36d_actor_epoch7.pt"
    low.touch()
    high.touch()
    assert discover_ddpg_checkpoint(tmp_path, None) == high.resolve()


def test_c3dqn_discovery_returns_all_prediction_seeds(tmp_path: Path) -> None:
    reward_root = tmp_path / "outputs/base_c3dqn_reward_ablation/prediction"
    expected = []
    for seed in (42, 43, 44):
        checkpoint = reward_root / f"seed_{seed}/best_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        expected.append(checkpoint.resolve())
    assert discover_c3dqn_checkpoints(tmp_path, None) == expected
