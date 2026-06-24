from pathlib import Path

import torch
from torch import nn

from evaluation.policies.ddpg_policy import NCDMDDPGDiversePolicy


class TinyNCDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_difficulty = nn.Embedding(3, 2)
        self.e_discrimination = nn.Embedding(3, 1)
        nn.init.zeros_(self.k_difficulty.weight)
        nn.init.zeros_(self.e_discrimination.weight)


class FixedActor(nn.Module):
    action_dim = 5
    hidden_dim = 3

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "ideal",
            torch.tensor([[0.0, 1.0, 0.5, 0.5, 0.5]]),
        )

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros((batch_size, self.hidden_dim), device=device),
            torch.zeros((batch_size, self.hidden_dim), device=device),
        )

    def forward(self, semantic, q_mask, difficulty, discrimination, response, hx, cx):
        del semantic, q_mask, difficulty, discrimination, response
        return self.ideal.to(hx.device), hx, cx


def test_margin_preserves_clearly_better_actor_match(tmp_path: Path):
    policy = NCDMDDPGDiversePolicy(
        tmp_path / "unused.pt",
        actor=FixedActor(),
        q_matrix=torch.tensor(
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        item_bank=torch.ones((3, 4)),
        ncdm=TinyNCDM(),
        device="cpu",
        top_k=2,
        exposure_weight=100.0,
        novelty_weight=0.0,
        coverage_weight=0.0,
        distance_margin_ratio=0.01,
        distance_mode="euclidean",
    )
    policy.global_exposure[1] = 1000
    policy.reset("student", 101, {})

    selected = policy.select(
        [1, 2],
        [0],
        [1.0],
        {"policy_step": 0, "selection_horizon": 10},
    )

    assert selected == 1
