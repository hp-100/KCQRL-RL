"""Shared NCDM candidate prefilter for training and evaluation policies."""
from __future__ import annotations

import time
from typing import Any, Sequence

import torch


class NCDMCandidatePrefilter:
    """Vectorized NCDM diagnostic prefilter with deterministic diversity slots."""

    def __init__(
        self,
        *,
        q_matrix: torch.Tensor,
        feature_cache,
        ncdm,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.q_matrix = q_matrix.to(feature_cache.device).float()
        self.feature_cache = feature_cache
        self.ncdm = ncdm
        self.config = dict(config or {})

    def _weights(self) -> dict[str, float]:
        nested = dict(self.config.get("weights") or {})
        defaults = {
            "uncertainty": 0.35,
            "weakness": 0.25,
            "novelty": 0.15,
            "difficulty": 0.15,
            "discrimination": 0.10,
        }
        return {
            name: float(
                nested.get(
                    name,
                    self.config.get(f"w_{name}", default),
                )
            )
            for name, default in defaults.items()
        }

    def select(
        self,
        candidate_item_ids: Sequence[int],
        alpha: torch.Tensor,
        mastery: torch.Tensor,
        coverage_count: torch.Tensor,
    ) -> tuple[list[int], dict[str, float | int | bool]]:
        start_time = time.perf_counter()
        original_ids = [int(item_id) for item_id in candidate_item_ids]
        if not original_ids:
            raise ValueError("candidate prefilter requires at least one candidate")

        enabled = bool(self.config.get("prefilter_enabled", True))
        if not enabled:
            return original_ids, {
                "prefilter_enabled": False,
                "raw_candidate_count": len(original_ids),
                "filtered_candidate_count": len(original_ids),
                "diversity_selected_count": 0,
                "candidate_prefilter_seconds": time.perf_counter() - start_time,
            }

        device = self.feature_cache.device
        ids = torch.as_tensor(original_ids, dtype=torch.long, device=device)
        q_mask = self.feature_cache.q_masks[ids].float()
        concept_count = q_mask.sum(dim=-1).clamp_min(1.0)
        mastery = mastery.to(device).float().flatten()
        coverage_count = coverage_count.to(device).float().flatten()
        if mastery.shape != (self.feature_cache.knowledge_dim,):
            raise ValueError("mastery has invalid shape")
        if coverage_count.shape != (self.feature_cache.knowledge_dim,):
            raise ValueError("coverage_count has invalid shape")

        with torch.no_grad():
            p_correct = self.ncdm.predict_with_alpha(
                alpha,
                ids,
                self.q_matrix,
            ).float().flatten()
        uncertainty = 4.0 * p_correct * (1.0 - p_correct)
        weakness = (
            (1.0 - mastery).unsqueeze(0) * q_mask
        ).sum(dim=-1) / concept_count
        novelty = (
            q_mask * (coverage_count.unsqueeze(0) == 0).to(q_mask.dtype)
        ).sum(dim=-1) / concept_count
        masked_difficulty = self.feature_cache.masked_difficulties[ids].float()
        mean_abs_gap = (
            (mastery.unsqueeze(0) - masked_difficulty).abs() * q_mask
        ).sum(dim=-1) / concept_count
        difficulty_match = 1.0 - mean_abs_gap.clamp(0, 1)
        discrimination = self.feature_cache.disc_norms[ids].float().mean(
            dim=-1
        ).clamp(0, 1)

        weights = self._weights()
        score = (
            weights["uncertainty"] * uncertainty
            + weights["weakness"] * weakness
            + weights["novelty"] * novelty
            + weights["difficulty"] * difficulty_match
            + weights["discrimination"] * discrimination
        )

        candidate_count = int(ids.numel())
        configured_top_k = int(
            self.config.get("prefilter_top_k", candidate_count)
        )
        if configured_top_k <= 0:
            raise ValueError("prefilter_top_k must be positive")
        top_k = min(configured_top_k, candidate_count)
        diversity_slots = min(
            max(0, int(self.config.get("diversity_quota", 0))),
            top_k,
        )
        primary_slots = top_k - diversity_slots

        score_order = torch.argsort(score, descending=True, stable=True).tolist()
        selected_positions = score_order[:primary_slots]
        selected_set = set(selected_positions)
        current_coverage = coverage_count.clone()
        if selected_positions:
            current_coverage = current_coverage + q_mask[
                torch.as_tensor(selected_positions, device=device)
            ].sum(dim=0)

        diversity_positions: list[int] = []
        for _ in range(diversity_slots):
            remaining = [
                position
                for position in range(candidate_count)
                if position not in selected_set
            ]
            if not remaining:
                break
            uncovered = (current_coverage == 0).to(q_mask.dtype)
            best_position = max(
                remaining,
                key=lambda position: (
                    float(
                        (
                            q_mask[position] * uncovered
                        ).sum().item()
                        / float(concept_count[position].item())
                    ),
                    float(score[position].item()),
                    -position,
                ),
            )
            selected_positions.append(best_position)
            diversity_positions.append(best_position)
            selected_set.add(best_position)
            current_coverage = current_coverage + q_mask[best_position]

        if len(selected_positions) < top_k:
            for position in score_order:
                if position not in selected_set:
                    selected_positions.append(position)
                    selected_set.add(position)
                if len(selected_positions) == top_k:
                    break

        filtered = [original_ids[position] for position in selected_positions]
        summary: dict[str, float | int | bool] = {
            "prefilter_enabled": True,
            "raw_candidate_count": candidate_count,
            "filtered_candidate_count": len(filtered),
            "diversity_selected_count": len(diversity_positions),
            "candidate_prefilter_seconds": time.perf_counter() - start_time,
            "uncertainty_mean": float(uncertainty.mean().item()),
            "weakness_mean": float(weakness.mean().item()),
            "novelty_mean": float(novelty.mean().item()),
            "difficulty_match_mean": float(difficulty_match.mean().item()),
            "discrimination_mean": float(discrimination.mean().item()),
            "score_mean": float(score.mean().item()),
            "score_max": float(score.max().item()),
        }
        return filtered, summary
