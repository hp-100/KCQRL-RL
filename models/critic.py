"""DDPG critic network."""
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128, action_dim: int = 73, q_clip: float = 20.0):
        super().__init__()
        self.q_clip = float(q_clip)
        self.critic_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, hx, action_vector):
        q = self.critic_net(torch.cat([hx, action_vector], dim=-1))
        return torch.clamp(q, -self.q_clip, self.q_clip)

Critic = LSTMCritic
