"""DDPG actor network for white-box item recommendation."""
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMActor(nn.Module):
    def __init__(self, semantic_dim: int = 128, q_dim: int = 36, resp_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.q_dim = q_dim
        self.hidden_dim = hidden_dim
        self.action_dim = q_dim * 2 + 1
        self.response_emb = nn.Linear(1, resp_dim)
        input_dim = semantic_dim + q_dim + q_dim + 1 + resp_dim
        self.norm = nn.LayerNorm(input_dim)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, self.action_dim), nn.Sigmoid())

    def forward(self, semantic_vec, q_mask_vec, diff_vec, disc_vec, response_val, hx, cx):
        resp_f = torch.relu(self.response_emb(response_val.unsqueeze(-1)))
        x = torch.cat([semantic_vec, q_mask_vec, diff_vec, disc_vec, resp_f], dim=-1)
        hx, cx = self.lstm_cell(self.norm(x), (hx, cx))
        return self.policy_head(hx), hx, cx

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

Actor = LSTMActor
