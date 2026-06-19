"""MIRT-native query reward helpers."""
from __future__ import annotations
import torch
from models.mirt import predict_with_theta

def bernoulli_nll(probs, labels):
    p=torch.as_tensor(probs).float().clamp(1e-7,1-1e-7); y=torch.as_tensor(labels).float().to(p.device)
    return float((-(y*torch.log(p)+(1-y)*torch.log(1-p))).mean().detach().cpu())

def query_nll(mirt, theta, query_item_ids, query_responses):
    with torch.no_grad(): p=predict_with_theta(mirt, theta, query_item_ids)
    return bernoulli_nll(p, query_responses)

def nll_drop_reward(previous_query_nll, current_query_nll, reward_scale=10.0, reward_clip=5.0):
    r=(float(previous_query_nll)-float(current_query_nll))*float(reward_scale)
    return max(-float(reward_clip), min(float(reward_clip), r))
