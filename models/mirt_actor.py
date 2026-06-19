"""MIRT-native DDPG actor."""
from __future__ import annotations
import torch
import torch.nn as nn

STATE_DEFINITION_VERSION = "mirt_state_v1_theta_lastitem_response_step"

class MIRTActor(nn.Module):
    def __init__(self, state_dim: int = 75, hidden_dim: int = 128, action_dim: int = 37):
        super().__init__()
        self.state_dim=int(state_dim); self.hidden_dim=int(hidden_dim); self.action_dim=int(action_dim)
        self.norm=nn.LayerNorm(self.state_dim)
        self.cell=nn.LSTMCell(self.state_dim, self.hidden_dim)
        self.head=nn.Sequential(nn.Linear(self.hidden_dim,128), nn.ReLU(), nn.Linear(128,self.action_dim))
    def forward(self, state, hx=None, cx=None):
        if state.dim()==1: state=state.unsqueeze(0)
        x=self.norm(state.float())
        b=x.shape[0]
        if hx is None: hx=torch.zeros(b,self.hidden_dim,device=x.device,dtype=x.dtype)
        if cx is None: cx=torch.zeros(b,self.hidden_dim,device=x.device,dtype=x.dtype)
        hx,cx=self.cell(x,(hx,cx))
        return self.head(hx), hx, cx
