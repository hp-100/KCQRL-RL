"""Recurrent MIRT critic aligned with actor sequence histories."""
from __future__ import annotations
import torch
import torch.nn as nn

class MIRTRecurrentCritic(nn.Module):
    def __init__(self, state_dim:int=75, action_dim:int=37, hidden_dim:int=128, q_clip:float=20.0):
        super().__init__(); self.state_dim=int(state_dim); self.action_dim=int(action_dim); self.hidden_dim=int(hidden_dim); self.q_clip=float(q_clip)
        self.norm=nn.LayerNorm(self.state_dim)
        self.lstm=nn.LSTM(input_size=self.state_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.q=nn.Sequential(nn.Linear(self.hidden_dim+self.action_dim,128), nn.ReLU(), nn.Linear(128,1))
    def init_hidden(self,batch_size:int,device=None):
        device=device or next(self.parameters()).device
        return (torch.zeros(1,batch_size,self.hidden_dim,device=device), torch.zeros(1,batch_size,self.hidden_dim,device=device))
    def forward_sequence(self, states, actions, hidden=None):
        if states.dim()!=3 or actions.dim()!=3: raise ValueError('states/actions must be [B,T,D]')
        if hidden is None: hidden=self.init_hidden(states.shape[0], states.device)
        enc,next_hidden=self.lstm(self.norm(states.float()), hidden)
        q=self.q(torch.cat([enc, actions.float()], dim=-1))
        return torch.clamp(q, -self.q_clip, self.q_clip), next_hidden
