"""Cached NCDM-native item features and padded batch builders for C3DQN-NCDM."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from models.ncdm import OfficialNCDM

@dataclass(frozen=True)
class NCDMFeatureDims:
    knowledge_dim: int
    history_feature_dim: int
    candidate_feature_dim: int
    global_feature_dim: int

    @classmethod
    def from_knowledge_dim(cls, knowledge_dim: int) -> "NCDMFeatureDims":
        return cls(knowledge_dim, 2 * knowledge_dim + 3, 2 * knowledge_dim + 1, 2 * knowledge_dim + 1)

class NCDMItemFeatureCache:
    """Precomputes q_mask, masked difficulty and normalized discrimination once per item."""
    def __init__(self, ncdm: OfficialNCDM, q_matrix: torch.Tensor, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.q_matrix = q_matrix.float().to(self.device)
        self.knowledge_dim = int(self.q_matrix.shape[1])
        self.dims = NCDMFeatureDims.from_knowledge_dim(self.knowledge_dim)
        item_count = min(int(self.q_matrix.shape[0]), int(ncdm.k_difficulty.num_embeddings), int(ncdm.e_discrimination.num_embeddings))
        item_ids = torch.arange(item_count, dtype=torch.long, device=self.device)
        with torch.no_grad():
            q_mask = self.q_matrix[:item_count].clamp(0, 1)
            difficulty = torch.sigmoid(ncdm.k_difficulty(item_ids))
            masked_difficulty = q_mask * difficulty
            disc_norm = torch.sigmoid(ncdm.e_discrimination(item_ids))
            candidate_features = torch.cat([q_mask, masked_difficulty, disc_norm], dim=1).float()
        expected = self.dims.candidate_feature_dim
        if candidate_features.shape != (item_count, expected):
            raise ValueError(f"candidate feature cache shape mismatch: {tuple(candidate_features.shape)} != {(item_count, expected)}")
        if not torch.isfinite(candidate_features).all():
            raise ValueError("candidate feature cache contains non-finite values")
        self.item_count = item_count
        self.q_masks = q_mask.detach()
        self.masked_difficulties = masked_difficulty.detach()
        self.disc_norms = disc_norm.detach()
        self.candidate_features = candidate_features.detach()

    def candidate(self, item_ids: Sequence[int]) -> torch.Tensor:
        ids = torch.as_tensor(list(item_ids), dtype=torch.long, device=self.device)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.item_count):
            raise IndexError(f"item id outside cached range [0,{self.item_count}): {ids.tolist()}")
        return self.candidate_features[ids]

    def history(self, item_ids: Sequence[int], responses: Sequence[float], selection_horizon: int) -> torch.Tensor:
        if len(item_ids) != len(responses):
            raise ValueError("history item/response lengths differ")
        base = self.candidate(item_ids)
        if len(item_ids) == 0:
            return torch.empty((0, self.dims.history_feature_dim), device=self.device)
        resp = torch.as_tensor(responses, dtype=torch.float32, device=self.device).view(-1, 1)
        if not torch.all((resp == 0.0) | (resp == 1.0)):
            raise ValueError("responses must be raw scalar 0/1 values")
        pos = (torch.arange(len(item_ids), dtype=torch.float32, device=self.device).view(-1, 1) + 1.0) / float(selection_horizon)
        return torch.cat([base, resp, pos], dim=1)

def build_global_feature(mastery: torch.Tensor, coverage: torch.Tensor, policy_step: int, selection_horizon: int) -> torch.Tensor:
    mastery = mastery.float().flatten()
    coverage = coverage.float().flatten().clamp(0, 1)
    if mastery.shape != coverage.shape:
        raise ValueError(f"mastery/coverage shape mismatch: {mastery.shape} != {coverage.shape}")
    step = torch.tensor([float(policy_step) / float(selection_horizon)], dtype=mastery.dtype, device=mastery.device)
    out = torch.cat([mastery, coverage, step], dim=0)
    if not torch.isfinite(out).all():
        raise ValueError("global feature contains non-finite values")
    return out

def pad_c3dqn_batch(samples: Sequence[dict], cache: NCDMItemFeatureCache, selection_horizon: int) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot build an empty C3DQN batch")
    max_t = max(1, max(len(s["history_item_ids"]) for s in samples))
    max_c = max(1, max(len(s["candidate_item_ids"]) for s in samples))
    b = len(samples); k = cache.knowledge_dim
    hist = torch.zeros((b, max_t, 2*k+3), device=cache.device)
    hist_mask = torch.zeros((b, max_t), dtype=torch.bool, device=cache.device)
    cand = torch.zeros((b, max_c, 2*k+1), device=cache.device)
    cand_mask = torch.zeros((b, max_c), dtype=torch.bool, device=cache.device)
    glob = torch.zeros((b, 2*k+1), device=cache.device)
    action = torch.zeros((b,), dtype=torch.long, device=cache.device)
    for row, s in enumerate(samples):
        h = cache.history(s["history_item_ids"], s["history_responses"], selection_horizon)
        hist[row, :h.shape[0]] = h; hist_mask[row, :h.shape[0]] = True
        cids = [int(x) for x in s["candidate_item_ids"]]
        if int(s["selected_item_id"]) not in cids:
            raise ValueError(f"selected_item_id {s['selected_item_id']} is not in candidate_item_ids")
        cf = cache.candidate(cids); cand[row, :cf.shape[0]] = cf; cand_mask[row, :cf.shape[0]] = True
        glob[row] = build_global_feature(torch.as_tensor(s["mastery"], device=cache.device), torch.as_tensor(s["coverage"], device=cache.device), int(s["policy_step"]), selection_horizon)
        action[row] = cids.index(int(s["selected_item_id"]))
    return {"history_features": hist, "history_mask": hist_mask, "candidate_features": cand, "candidate_mask": cand_mask, "global_features": glob, "action_index": action}
