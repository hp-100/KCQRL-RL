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
        return cls(
            knowledge_dim,
            2 * knowledge_dim + 3,
            2 * knowledge_dim + 1,
            2 * knowledge_dim + 1,
        )


class NCDMItemFeatureCache:
    """Precompute Q masks, masked difficulty and discrimination once per item."""

    def __init__(
        self,
        ncdm: OfficialNCDM,
        q_matrix: torch.Tensor,
        device: torch.device | str = "cpu",
        *,
        allow_item_count_intersection: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.q_matrix = q_matrix.float().to(self.device)
        self.knowledge_dim = int(self.q_matrix.shape[1])
        self.dims = NCDMFeatureDims.from_knowledge_dim(self.knowledge_dim)
        q_count = int(self.q_matrix.shape[0])
        ncdm_items = int(ncdm.k_difficulty.num_embeddings)
        disc_items = int(ncdm.e_discrimination.num_embeddings)
        if not allow_item_count_intersection and not (
            q_count == ncdm_items == disc_items
        ):
            raise ValueError(
                "strict item count check failed: "
                f"q_matrix_item_count={q_count}, "
                f"ncdm_difficulty_items={ncdm_items}, "
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
            candidate_features = torch.cat(
                [q_mask, masked_difficulty, disc_norm],
                dim=1,
            ).float()
        expected = self.dims.candidate_feature_dim
        if candidate_features.shape != (item_count, expected):
            raise ValueError(
                "candidate feature cache shape mismatch: "
                f"{tuple(candidate_features.shape)} != {(item_count, expected)}"
            )
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
            raise IndexError(
                f"item id outside cached range [0,{self.item_count}): {ids.tolist()}"
            )
        return self.candidate_features[ids]

    def history(
        self,
        item_ids: Sequence[int],
        responses: Sequence[float],
        selection_horizon: int,
    ) -> torch.Tensor:
        if len(item_ids) != len(responses):
            raise ValueError("history item/response lengths differ")
        base = self.candidate(item_ids)
        if len(item_ids) == 0:
            return torch.empty(
                (0, self.dims.history_feature_dim),
                device=self.device,
            )
        responses_tensor = torch.as_tensor(
            responses,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        if not torch.all(
            (responses_tensor == 0.0) | (responses_tensor == 1.0)
        ):
            raise ValueError("responses must be raw scalar 0/1 values")
        positions = (
            torch.arange(
                len(item_ids),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, 1)
            + 1.0
        ) / float(selection_horizon)
        return torch.cat([base, responses_tensor, positions], dim=1)


def build_global_feature(
    mastery: torch.Tensor,
    coverage: torch.Tensor,
    policy_step: int,
    selection_horizon: int,
) -> torch.Tensor:
    mastery = mastery.float().flatten()
    coverage = coverage.float().flatten().clamp(0, 1)
    if mastery.shape != coverage.shape:
        raise ValueError(
            f"mastery/coverage shape mismatch: {mastery.shape} != {coverage.shape}"
        )
    step = torch.tensor(
        [float(policy_step) / float(selection_horizon)],
        dtype=mastery.dtype,
        device=mastery.device,
    )
    output = torch.cat([mastery, coverage, step], dim=0)
    if not torch.isfinite(output).all():
        raise ValueError("global feature contains non-finite values")
    return output


def pad_c3dqn_batch(
    samples: Sequence[dict],
    cache: NCDMItemFeatureCache,
    selection_horizon: int,
    *,
    require_exact_coverage: bool = False,
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot build an empty C3DQN batch")
    max_history = max(1, max(len(sample["history_item_ids"]) for sample in samples))
    max_candidates = max(
        1,
        max(len(sample["candidate_item_ids"]) for sample in samples),
    )
    batch_size = len(samples)
    knowledge_dim = cache.knowledge_dim

    history = torch.zeros(
        (batch_size, max_history, 2 * knowledge_dim + 3),
        device=cache.device,
    )
    history_mask = torch.zeros(
        (batch_size, max_history),
        dtype=torch.bool,
        device=cache.device,
    )
    candidates = torch.zeros(
        (batch_size, max_candidates, 2 * knowledge_dim + 1),
        device=cache.device,
    )
    candidate_mask = torch.zeros(
        (batch_size, max_candidates),
        dtype=torch.bool,
        device=cache.device,
    )
    global_features = torch.zeros(
        (batch_size, 2 * knowledge_dim + 1),
        device=cache.device,
    )
    exact_coverage = torch.zeros(
        (batch_size, knowledge_dim),
        device=cache.device,
    )
    action_index = torch.zeros(
        (batch_size,),
        dtype=torch.long,
        device=cache.device,
    )

    for row_index, sample in enumerate(samples):
        history_features = cache.history(
            sample["history_item_ids"],
            sample["history_responses"],
            selection_horizon,
        )
        history[row_index, : history_features.shape[0]] = history_features
        history_mask[row_index, : history_features.shape[0]] = True

        candidate_ids = [int(item_id) for item_id in sample["candidate_item_ids"]]
        selected_item_id = int(sample["selected_item_id"])
        if selected_item_id not in candidate_ids:
            raise ValueError(
                f"selected_item_id {selected_item_id} is not in candidate_item_ids"
            )
        candidate_features = cache.candidate(candidate_ids)
        candidates[row_index, : candidate_features.shape[0]] = candidate_features
        candidate_mask[row_index, : candidate_features.shape[0]] = True

        coverage_value = sample.get("coverage_count")
        if coverage_value is None:
            if require_exact_coverage:
                raise ValueError("Set-C3DQN sample requires exact coverage_count")
            normalized_coverage = torch.as_tensor(
                sample["coverage"],
                device=cache.device,
            ).float()
            coverage_count = normalized_coverage * float(selection_horizon)
        else:
            coverage_count = torch.as_tensor(
                coverage_value,
                device=cache.device,
            ).float()
            normalized_coverage = (
                coverage_count / float(selection_horizon)
            ).clamp(0, 1)
        if coverage_count.shape != (knowledge_dim,):
            raise ValueError("coverage_count has invalid shape")
        exact_coverage[row_index] = coverage_count
        global_features[row_index] = build_global_feature(
            torch.as_tensor(sample["mastery"], device=cache.device),
            normalized_coverage,
            int(sample["policy_step"]),
            selection_horizon,
        )
        action_index[row_index] = candidate_ids.index(selected_item_id)

    return {
        "history_features": history,
        "history_mask": history_mask,
        "candidate_features": candidates,
        "candidate_mask": candidate_mask,
        "global_features": global_features,
        "coverage_count": exact_coverage,
        "action_index": action_index,
    }
