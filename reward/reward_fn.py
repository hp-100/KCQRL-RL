"""Reward functions for active diagnosis."""
from __future__ import annotations
import torch


def entropy_from_predictions(preds: torch.Tensor) -> torch.Tensor:
    p = preds.clamp(1e-5, 1.0 - 1e-5)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()


def nll_gain(model, before_state, after_state, item, response):
    before = model.predict(before_state, item)
    after = model.predict(after_state, item)
    return -(response * after.log() + (1 - response) * (1 - after).log()) + (response * before.log() + (1 - response) * (1 - before).log())


def coverage_bonus(new_items, total_items):
    return len(set(new_items)) / max(1, total_items)


def information_gain_reward(prev_entropy: float, curr_entropy: float, scale: float = 10.0, clip: float = 10.0) -> float:
    return max(-clip, min(clip, (prev_entropy - curr_entropy) * scale))


def reward_fn(nll_gain, coverage, lambda_cov=0.03):
    return nll_gain + lambda_cov * coverage
