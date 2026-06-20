"""Permutation-equivariant set attention blocks used by Set-C3DQN-NCDM."""
from __future__ import annotations
import torch
import torch.nn as nn

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.0, ffn_multiplier:int=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,ffn_multiplier*d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ffn_multiplier*d_model,d_model),nn.Dropout(dropout))
        self.norm2=nn.LayerNorm(d_model)
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, key_padding_mask:torch.Tensor|None=None):
        a,_=self.attn(q,k,v,key_padding_mask=key_padding_mask,need_weights=False)
        h=self.norm1(q+a)
        return self.norm2(h+self.ffn(h))

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, num_inducing_points:int, dropout:float=0.0):
        super().__init__()
        self.inducing_points=nn.Parameter(torch.randn(num_inducing_points,d_model)*0.02)
        self.mab1=MultiheadAttentionBlock(d_model,n_heads,dropout)
        self.mab2=MultiheadAttentionBlock(d_model,n_heads,dropout)
        self.last_induced_summary: torch.Tensor|None=None
    def forward(self, x:torch.Tensor, candidate_mask:torch.Tensor|None=None):
        b=x.shape[0]
        i=self.inducing_points.unsqueeze(0).expand(b,-1,-1)
        kpm=(~candidate_mask.bool()) if candidate_mask is not None else None
        h=self.mab1(i,x,x,key_padding_mask=kpm)
        y=self.mab2(x,h,h)
        if candidate_mask is not None:
            y=y*candidate_mask.unsqueeze(-1).to(y.dtype)
        self.last_induced_summary=h.detach()
        return y
