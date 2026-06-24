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
    def __init__(self, ncdm: OfficialNCDM, q_matrix: torch.Tensor, device: torch.device | str = "cpu", *, allow_item_count_intersection: bool = False) -> None:
        self.device = torch.device(device)
        self.q_matrix = q_matrix.float().to(self.device)
        self.knowledge_dim = int(self.q_matrix.shape[1])
        self.dims = NCDMFeatureDims.from_knowledge_dim(self.knowledge_dim)
        q_count = int(self.q_matrix.shape[0])
        ncdm_items = int(ncdm.k_difficulty.num_embeddings)
        disc_items = int(ncdm.e_discrimination.num_embeddings)
        if not allow_item_count_intersection and not (q_count == ncdm_items == disc_items):
            raise ValueError(
                "strict item count check failed: "
                f"q_matrix_item_count={q_count}, ncdm_difficulty_items={ncdm_items}, "
                f"ncdm_discrimination_items={disc_items}"
            )
        item_count = min(q_count, ncdm_items, disc_items)
        self.strict_item_count_check = not allow_item_count_intersection
        self.q_matrix_item_count = q_count
        self.ncdm_item_count = ncdm_items
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

@dataclass(frozen=True)
class NCDMDiagnosticState:
    alpha: torch.Tensor
    mastery: torch.Tensor
    coverage_count: torch.Tensor
    coverage: torch.Tensor
    query_nll: float | None = None


def prefilter_candidates_vectorized(
    candidate_item_ids: Sequence[int],
    *,
    cache: NCDMItemFeatureCache,
    ncdm: OfficialNCDM | None,
    alpha: torch.Tensor,
    mastery: torch.Tensor,
    coverage_count: torch.Tensor,
    top_k: int = 256,
    uncertainty_weight: float = 0.45,
    weakness_weight: float = 0.30,
    novelty_weight: float = 0.15,
    discrimination_weight: float = 0.10,
    diversity_enabled: bool = True,
    diversity_fraction: float = 0.20,
) -> list[int]:
    """Deterministic O(CK) diagnostic prefilter before candidate-history attention."""
    ids = torch.as_tensor(list(candidate_item_ids), dtype=torch.long, device=cache.device)
    if ids.numel() == 0:
        return []
    k = min(int(top_k), int(ids.numel()))
    q_mask = cache.q_masks[ids]
    denom = q_mask.sum(-1).clamp_min(1.0)
    if ncdm is not None:
        with torch.no_grad():
            p_correct = ncdm.predict_with_alpha(alpha, ids, cache.q_matrix).float().clamp(0, 1)
        uncertainty = 4.0 * p_correct * (1.0 - p_correct)
    else:
        uncertainty = torch.zeros(ids.shape[0], device=cache.device)
    weakness = ((1.0 - mastery.float()).view(1, -1) * q_mask).sum(-1) / denom
    novelty = (q_mask * (coverage_count.float().view(1, -1) == 0).float()).sum(-1) / denom
    discrimination = cache.disc_norms[ids].squeeze(-1)
    score = float(uncertainty_weight) * uncertainty + float(weakness_weight) * weakness + float(novelty_weight) * novelty + float(discrimination_weight) * discrimination
    if not diversity_enabled or k <= 1:
        return ids[torch.argsort(score, descending=True, stable=True)[:k]].detach().cpu().tolist()
    primary_k = max(1, int(round(k * (1.0 - float(diversity_fraction)))))
    primary_idx = torch.argsort(score, descending=True, stable=True)[:primary_k]
    selected = primary_idx.detach().cpu().tolist()
    selected_set = set(selected)
    covered = q_mask[primary_idx].sum(0) if selected else torch.zeros(cache.knowledge_dim, device=cache.device)
    for concept in torch.argsort(covered, stable=True).detach().cpu().tolist():
        if len(selected) >= k:
            break
        has_concept = torch.nonzero(q_mask[:, int(concept)] > 0, as_tuple=False).flatten()
        if has_concept.numel() == 0:
            continue
        ordered = has_concept[torch.argsort(score[has_concept], descending=True, stable=True)]
        for idx in ordered.detach().cpu().tolist():
            if idx not in selected_set:
                selected.append(idx); selected_set.add(idx); break
    if len(selected) < k:
        for idx in torch.argsort(score, descending=True, stable=True).detach().cpu().tolist():
            if idx not in selected_set:
                selected.append(idx); selected_set.add(idx)
                if len(selected) >= k: break
    return ids[torch.tensor(selected, dtype=torch.long, device=cache.device)].detach().cpu().tolist()
