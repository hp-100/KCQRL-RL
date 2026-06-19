"""MIRT-native DDPG MLP actor."""
from __future__ import annotations
import torch
import torch.nn as nn

STATE_DEFINITION_VERSION = "mirt_state_v1_theta_lastitem_response_step"
ACTOR_ARCHITECTURE = "mlp_explicit_state"

class MIRTActor(nn.Module):
    """Feed-forward actor over the explicit 75-dimensional MIRT state."""

    def __init__(self, state_dim: int = 75, hidden_dim: int = 128, action_dim: int = 37):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state.float())
