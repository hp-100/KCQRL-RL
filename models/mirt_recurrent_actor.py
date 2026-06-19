"""Recurrent deterministic MIRT actor trained with sequence BPTT."""
from __future__ import annotations
import torch
import torch.nn as nn
from models.mirt_actor import STATE_DEFINITION_VERSION

ACTOR_ARCHITECTURE = "lstm_sequence_bptt"

class MIRTRecurrentActor(nn.Module):
    def __init__(self, state_dim:int=75, hidden_dim:int=128, action_dim:int=37):
        super().__init__()
        self.state_dim=int(state_dim); self.hidden_dim=int(hidden_dim); self.action_dim=int(action_dim)
        self.norm=nn.LayerNorm(self.state_dim)
        self.lstm=nn.LSTM(input_size=self.state_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.head=nn.Sequential(nn.Linear(self.hidden_dim,128), nn.ReLU(), nn.Linear(128,self.action_dim))
    def init_hidden(self, batch_size:int, device=None):
        device=device or next(self.parameters()).device
        return (torch.zeros(1,batch_size,self.hidden_dim,device=device), torch.zeros(1,batch_size,self.hidden_dim,device=device))
    def forward_sequence(self, states, hidden=None):
        if states.dim()!=3: raise ValueError(f"states must be [B,T,{self.state_dim}], got {tuple(states.shape)}")
        if hidden is None: hidden=self.init_hidden(states.shape[0], states.device)
        out,next_hidden=self.lstm(self.norm(states.float()), hidden)
        return self.head(out), next_hidden
    def forward_step(self, state, hidden):
        if state.dim()==1: state=state.view(1,1,-1)
        elif state.dim()==2: state=state.unsqueeze(1)
        actions,next_hidden=self.forward_sequence(state, hidden)
        return actions[:,0,:], next_hidden
