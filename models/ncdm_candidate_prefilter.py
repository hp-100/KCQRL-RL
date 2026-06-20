"""Shared NCDM candidate prefilter for training and evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import time
import torch

class NCDMCandidatePrefilter:
    def __init__(self, *, q_matrix: torch.Tensor, feature_cache, ncdm, config: dict[str, Any] | None = None) -> None:
        self.q_matrix = q_matrix.to(feature_cache.device).float()
        self.feature_cache = feature_cache
        self.ncdm = ncdm
        self.config = dict(config or {})

    def select(self, candidate_item_ids: Sequence[int], alpha: torch.Tensor, mastery: torch.Tensor, coverage_count: torch.Tensor):
        t0 = time.perf_counter()
        ids = torch.as_tensor(list(candidate_item_ids), dtype=torch.long, device=self.feature_cache.device)
        if ids.numel() == 0:
            raise ValueError("candidate prefilter requires at least one candidate")
        q_mask = self.feature_cache.q_masks[ids].float()
        concept_count = q_mask.sum(-1).clamp_min(1.0)
        with torch.no_grad():
            p_correct = self.ncdm.predict_with_alpha(alpha, ids, self.q_matrix).float().flatten()
        uncertainty = 4.0 * p_correct * (1.0 - p_correct)
        mastery = mastery.to(self.feature_cache.device).float().flatten()
        coverage_count = coverage_count.to(self.feature_cache.device).float().flatten()
        weakness = ((1.0 - mastery).unsqueeze(0) * q_mask).sum(-1) / concept_count
        novelty = (q_mask * (coverage_count.unsqueeze(0) == 0).float()).sum(-1) / concept_count
        masked_difficulty = self.feature_cache.masked_difficulties[ids].float()
        mean_abs_gap = (torch.abs(mastery.unsqueeze(0) - masked_difficulty) * q_mask).sum(-1) / concept_count
        difficulty_match = 1.0 - mean_abs_gap.clamp(0, 1)
        discrimination = self.feature_cache.disc_norms[ids].float().mean(-1).clamp(0, 1)
        w = self.config
        score = (float(w.get("w_uncertainty", 1.0)) * uncertainty + float(w.get("w_weakness", 1.0)) * weakness + float(w.get("w_novelty", 1.0)) * novelty + float(w.get("w_difficulty", 1.0)) * difficulty_match + float(w.get("w_discrimination", 1.0)) * discrimination)
        enabled = bool(w.get("prefilter_enabled", True))
        top_k = int(w.get("prefilter_top_k", ids.numel()))
        if (not enabled) or top_k <= 0 or top_k >= ids.numel():
            order = torch.argsort(score, descending=True, stable=True)
        else:
            order = torch.argsort(score, descending=True, stable=True)[:top_k]
        filtered = ids[order].detach().cpu().tolist()
        summary = {"raw_candidate_count": int(ids.numel()), "filtered_candidate_count": len(filtered), "candidate_prefilter_seconds": time.perf_counter() - t0, "uncertainty_mean": float(uncertainty.mean()), "weakness_mean": float(weakness.mean()), "novelty_mean": float(novelty.mean()), "difficulty_match_mean": float(difficulty_match.mean()), "discrimination_mean": float(discrimination.mean())}
        return filtered, summary
