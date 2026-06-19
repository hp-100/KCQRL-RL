"""Pure MIRT state and action-feature utilities for DDPG-MIRT."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from models.mirt import fit_student_theta

@dataclass
class ActionNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    def normalize(self,x): return (x-self.mean.to(x.device))/(self.std.to(x.device).clamp_min(1e-8))
    def denormalize(self,x): return x*self.std.to(x.device).clamp_min(1e-8)+self.mean.to(x.device)
    def state_dict(self): return {"mean":self.mean.detach().cpu(),"std":self.std.detach().cpu()}
    @classmethod
    def from_state_dict(cls,d): return cls(torch.as_tensor(d["mean"]).float(), torch.as_tensor(d["std"]).float())

def item_action_features(mirt, item_ids: Sequence[int], device=None):
    device=device or next(mirt.parameters()).device
    ids=torch.tensor(list(item_ids),dtype=torch.long,device=device)
    with torch.no_grad(): return torch.cat([mirt.disc_emb(ids), mirt.diff_emb(ids)], dim=1)

def compute_action_normalizer(mirt, valid_item_ids: Sequence[int], device=None):
    feats=item_action_features(mirt, valid_item_ids, device)
    return ActionNormalizer(feats.mean(0).detach().cpu(), feats.std(0,unbiased=False).clamp_min(1e-6).detach().cpu())

def nearest_item(action, candidate_item_ids, mirt, normalizer:ActionNormalizer, device=None):
    if not candidate_item_ids: raise ValueError("empty candidate_item_ids")
    device=device or (action.device if torch.is_tensor(action) else next(mirt.parameters()).device)
    a=torch.as_tensor(action,dtype=torch.float32,device=device).view(1,-1)
    feats=normalizer.normalize(item_action_features(mirt,candidate_item_ids,device))
    idx=int(torch.argmin(torch.cdist(a,feats)).item())
    return int(list(candidate_item_ids)[idx])

def build_mirt_state(mirt, history_item_ids, history_responses, step:int, max_steps:int, theta_cfg=None, device=None):
    theta_cfg=theta_cfg or {}; device=device or next(mirt.parameters()).device
    theta=fit_student_theta(mirt, history_item_ids, history_responses, device=device, **theta_cfg)
    if history_item_ids:
        last=int(history_item_ids[-1]); disc=mirt.disc_emb(torch.tensor([last],device=device)).squeeze(0).detach(); diff=mirt.diff_emb(torch.tensor([last],device=device)).view(1).detach(); resp=torch.tensor([float(history_responses[-1])],device=device)
    else:
        disc=torch.zeros(mirt.n_dims,device=device); diff=torch.zeros(1,device=device); resp=torch.zeros(1,device=device)
    ns=torch.tensor([float(step)/max(1,int(max_steps))],device=device)
    return torch.cat([theta.detach(),disc,diff,resp,ns]).float()
