"""MIRT-native DDPG critic."""
from __future__ import annotations
import torch
import torch.nn as nn

class MIRTCritic(nn.Module):
    def __init__(self, state_dim:int=75, action_dim:int=37, hidden_dim:int=128, q_clip:float=20.0):
        super().__init__(); self.q_clip=float(q_clip)
        self.net=nn.Sequential(nn.LayerNorm(state_dim+action_dim), nn.Linear(state_dim+action_dim,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self,state,action):
        if state.dim()==1: state=state.unsqueeze(0)
        if action.dim()==1: action=action.unsqueeze(0)
        return torch.clamp(self.net(torch.cat([state.float(),action.float()],-1)), -self.q_clip, self.q_clip)
