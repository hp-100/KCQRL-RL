"""Set-attention blocks used by Set-C3DQN."""
from __future__ import annotations
import torch
import torch.nn as nn

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.0):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.ln1=nn.LayerNorm(d_model); self.ln2=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,4*d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(4*d_model,d_model))
        self.drop=nn.Dropout(dropout)
    def forward(self, x:torch.Tensor, y:torch.Tensor|None=None, *, key_padding_mask:torch.Tensor|None=None)->torch.Tensor:
        if y is None: y=x
        a,_=self.attn(x,y,y,key_padding_mask=key_padding_mask,need_weights=False)
        h=self.ln1(x+self.drop(a))
        return self.ln2(h+self.drop(self.ffn(h)))

class InducedSetAttentionBlock(nn.Module):
    def __init__(self,d_model:int,n_heads:int,num_inducing_points:int,dropout:float=0.0):
        super().__init__(); self.inducing_points=nn.Parameter(torch.randn(1,int(num_inducing_points),d_model)*0.02)
        self.mab1=MultiheadAttentionBlock(d_model,n_heads,dropout); self.mab2=MultiheadAttentionBlock(d_model,n_heads,dropout)
    def forward(self,x:torch.Tensor, *, key_padding_mask:torch.Tensor|None=None)->torch.Tensor:
        b=x.shape[0]; i=self.inducing_points.expand(b,-1,-1)
        h=self.mab1(i,x,key_padding_mask=key_padding_mask)
        return self.mab2(x,h,key_padding_mask=None)
