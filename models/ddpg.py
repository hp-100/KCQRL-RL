"""DDPG agent assembly helpers."""
from __future__ import annotations
import torch
from models.actor import LSTMActor
from models.critic import LSTMCritic


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


class DDPGAgent:
    def __init__(self, q_dim=36, semantic_dim=128, hidden_dim=128, actor_lr=1e-4, critic_lr=1e-4, device="cpu"):
        self.device = torch.device(device)
        self.actor = LSTMActor(semantic_dim, q_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = LSTMCritic(hidden_dim, self.actor.action_dim).to(self.device)
        self.target_actor = LSTMActor(semantic_dim, q_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_critic = LSTMCritic(hidden_dim, self.actor.action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
