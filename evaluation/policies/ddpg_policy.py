"""DDPG policies for NCDM-based adaptive item selection."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.actor import LSTMActor
from .base import BaseCATPolicy, PolicyMetadata


FORBIDDEN_CONTEXT_KEYS = {
    "query_item_ids",
    "query_responses",
    "query_labels",
    "future_responses",
    "future_item_ids",
    "candidate_response_lookup",
    "query_loss",
}


def _extract_actor_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract an actor state dict from raw or wrapped checkpoints."""
    if isinstance(checkpoint, Mapping):
        for key in (
            "actor_state_dict",
            "model_state_dict",
            "state_dict",
            "actor",
        ):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "DDPG checkpoint must be a raw actor state_dict or contain one of "
        "actor_state_dict/model_state_dict/state_dict/actor"
    )


def load_lstm_actor_checkpoint(
    checkpoint_path: str | Path,
    *,
    semantic_dim: int,
    q_dim: int,
    device: str | torch.device = "cpu",
) -> LSTMActor:
    """Load the 36D white-box LSTM actor used by NCDM-DDPG."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"DDPG actor checkpoint not found: {path}")
    target_device = torch.device(device)
    actor = LSTMActor(semantic_dim=int(semantic_dim), q_dim=int(q_dim)).to(
        target_device
    )
    payload = torch.load(path, map_location=target_device)
    actor.load_state_dict(_extract_actor_state(payload), strict=True)
    actor.eval()
    return actor


class DDPGPolicy(BaseCATPolicy):
    """Continuous ideal-item actor with nearest-real-item projection."""

    name = "DDPG"
    metadata = PolicyMetadata(
        name=name,
        implementation="checkpoint",
        selection_model="lstm_ddpg_continuous_ideal_item_nearest_neighbor",
        evaluator_model="NCDM",
        uses_query_labels=False,
        uses_privileged_information=False,
    )

    def __init__(
        self,
        checkpoint: str | Path,
        actor=None,
        q_matrix=None,
        item_bank=None,
        ncdm=None,
        device: str | torch.device = "cpu",
        allow_debug_fallback: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.q_matrix = (
            torch.as_tensor(q_matrix, dtype=torch.float32, device=self.device)
            if q_matrix is not None
            else None
        )
        self.item_bank = (
            torch.nn.functional.normalize(
                torch.as_tensor(item_bank, dtype=torch.float32, device=self.device),
                p=2,
                dim=1,
            )
            if item_bank is not None
            else None
        )
        self.ncdm = ncdm.to(self.device).eval() if hasattr(ncdm, "to") else ncdm
        self.allow_debug_fallback = bool(allow_debug_fallback)

        if actor is None and self.checkpoint.exists():
            if self.q_matrix is None or self.item_bank is None or self.ncdm is None:
                raise ValueError(
                    "loading a DDPG actor requires q_matrix, item_bank and ncdm"
                )
            actor = load_lstm_actor_checkpoint(
                self.checkpoint,
                semantic_dim=int(self.item_bank.shape[1]),
                q_dim=int(self.q_matrix.shape[1]),
                device=self.device,
            )
        self.actor = actor.to(self.device).eval() if hasattr(actor, "to") else actor

        if self.actor is None:
            if self.allow_debug_fallback:
                self.metadata = PolicyMetadata(
                    name=self.name,
                    implementation="explicit_debug_fallback",
                    selection_model="first_candidate",
                    evaluator_model="NCDM",
                    notes=f"Checkpoint missing: {self.checkpoint}",
                )
            else:
                raise FileNotFoundError(
                    f"DDPG actor checkpoint not found: {self.checkpoint}"
                )
        else:
            self._validate_assets()

        self.hx: torch.Tensor | None = None
        self.cx: torch.Tensor | None = None
        self._processed_items: list[int] = []
        self._processed_responses: list[float] = []

    def _validate_assets(self) -> None:
        if self.q_matrix is None or self.item_bank is None or self.ncdm is None:
            raise ValueError("DDPG policy requires q_matrix, item_bank and ncdm")
        if self.q_matrix.ndim != 2 or self.item_bank.ndim != 2:
            raise ValueError("q_matrix and item_bank must be rank-2 tensors")
        item_count = min(
            int(self.q_matrix.shape[0]),
            int(self.item_bank.shape[0]),
            int(self.ncdm.k_difficulty.num_embeddings),
            int(self.ncdm.e_discrimination.num_embeddings),
        )
        if item_count <= 0:
            raise ValueError("DDPG policy has no common valid item IDs")
        expected_action_dim = int(self.q_matrix.shape[1]) * 2 + 1
        actor_action_dim = int(
            getattr(self.actor, "action_dim", expected_action_dim)
        )
        if actor_action_dim != expected_action_dim:
            raise ValueError(
                f"actor action_dim={actor_action_dim}, expected {expected_action_dim}"
            )
        self.valid_item_count = item_count

    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)
        self.hx = self.cx = None
        self._processed_items = []
        self._processed_responses = []

    def _init_hidden(self) -> None:
        if self.actor is None:
            return
        if hasattr(self.actor, "init_hidden"):
            self.hx, self.cx = self.actor.init_hidden(1, self.device)
        else:
            hidden_dim = int(getattr(self.actor, "hidden_dim", 128))
            self.hx = torch.zeros((1, hidden_dim), device=self.device)
            self.cx = torch.zeros((1, hidden_dim), device=self.device)

    def _process_event(self, item_id: int, response: float) -> torch.Tensor:
        if self.actor is None:
            raise RuntimeError("cannot process history without an actor")
        if self.hx is None or self.cx is None:
            self._init_hidden()
        assert self.hx is not None and self.cx is not None

        item = int(item_id)
        if item < 0 or item >= self.valid_item_count:
            raise ValueError(f"history item ID outside common asset bounds: {item}")
        item_tensor = torch.tensor([item], dtype=torch.long, device=self.device)
        response_tensor = torch.tensor(
            [float(response)], dtype=torch.float32, device=self.device
        )
        semantic = self.item_bank[item].unsqueeze(0)
        q_mask = self.q_matrix[item].unsqueeze(0)
        difficulty = torch.sigmoid(self.ncdm.k_difficulty(item_tensor))
        discrimination = torch.sigmoid(
            self.ncdm.e_discrimination(item_tensor)
        )
        ideal, self.hx, self.cx = self.actor(
            semantic,
            q_mask,
            difficulty,
            discrimination,
            response_tensor,
            self.hx,
            self.cx,
        )
        return ideal

    def _sync_history(
        self,
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
    ) -> torch.Tensor:
        items = [int(item) for item in history_item_ids]
        responses = [float(response) for response in history_responses]
        if len(items) != len(responses):
            raise ValueError("history item/response lengths do not match")
        if not items:
            raise ValueError(f"{self.name} requires warm-start history")

        is_continuation = (
            items[: len(self._processed_items)] == self._processed_items
            and responses[: len(self._processed_responses)]
            == self._processed_responses
        )
        if not is_continuation:
            self.hx = self.cx = None
            self._processed_items = []
            self._processed_responses = []

        ideal: torch.Tensor | None = None
        start = len(self._processed_items)
        for item, response in zip(items[start:], responses[start:]):
            ideal = self._process_event(item, response)
            self._processed_items.append(item)
            self._processed_responses.append(response)

        if ideal is None:
            self.hx = self.cx = None
            self._processed_items = []
            self._processed_responses = []
            for item, response in zip(items, responses):
                ideal = self._process_event(item, response)
                self._processed_items.append(item)
                self._processed_responses.append(response)
        assert ideal is not None
        return ideal

    def _validate_selection_inputs(
        self,
        candidate_item_ids: Sequence[int],
        context: dict[str, Any],
    ) -> list[int]:
        candidates = [int(item) for item in candidate_item_ids]
        if not candidates:
            raise ValueError(f"{self.name} requires at least one candidate")
        leaked = FORBIDDEN_CONTEXT_KEYS & set((context or {}).keys())
        if leaked:
            raise ValueError(
                f"{self.name} received privileged context keys: {sorted(leaked)}"
            )
        invalid = [
            item for item in candidates if item < 0 or item >= self.valid_item_count
        ]
        if invalid:
            raise ValueError(
                f"candidate IDs outside common asset bounds: {invalid[:5]}"
            )
        return candidates

    def _candidate_feature_blocks(
        self,
        candidate_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_features = self.q_matrix[candidate_tensor]
        difficulty = torch.sigmoid(self.ncdm.k_difficulty(candidate_tensor))
        discrimination = torch.sigmoid(
            self.ncdm.e_discrimination(candidate_tensor)
        )
        return q_features, difficulty, discrimination

    def select(
        self,
        candidate_item_ids: Sequence[int],
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
        context: dict[str, Any],
    ) -> int:
        candidates = self._validate_selection_inputs(candidate_item_ids, context)
        if self.actor is None:
            return candidates[0]

        with torch.no_grad():
            ideal = self._sync_history(history_item_ids, history_responses)
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=self.device
            )
            q_features, difficulty, discrimination = self._candidate_feature_blocks(
                candidate_tensor
            )
            candidate_features = torch.cat(
                [q_features, difficulty, discrimination],
                dim=-1,
            )
            distances = torch.cdist(ideal, candidate_features).squeeze(0)
            selected_index = int(torch.argmin(distances).item())
        return candidates[selected_index]


class NCDMDDPGPolicy(DDPGPolicy):
    """Paper-facing name for the white-box NCDM-DDPG selector."""

    name = "NCDM-DDPG"
    metadata = PolicyMetadata(
        name=name,
        implementation="checkpoint",
        selection_model="lstm_ddpg_continuous_73d_ideal_item_nearest_neighbor",
        evaluator_model="NCDM",
        uses_query_labels=False,
        uses_privileged_information=False,
        notes="Actor action = Q-mask + NCDM difficulty + NCDM discrimination",
    )


class NCDMDDPGDiversePolicy(NCDMDDPGPolicy):
    """Conservative exposure-aware reranker using a frozen DDPG actor.

    The default path preserves the actor's original 73D Euclidean geometry.  It
    only reranks real items that are both in the nearest ``top_k`` set and
    within a small relative distance margin of the best geometric match.  This
    prevents the exposure term from replacing a clearly superior actor match.
    Optional novelty and coverage terms remain available for later ablations,
    but are disabled by default.
    """

    name = "NCDM-DDPG-Diverse"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        top_k: int = 16,
        exposure_weight: float = 0.005,
        novelty_weight: float = 0.0,
        coverage_weight: float = 0.0,
        distance_margin_ratio: float = 0.02,
        distance_mode: str = "euclidean",
        q_distance_weight: float = 1.0,
        difficulty_distance_weight: float = 1.0,
        discrimination_distance_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint, **kwargs)
        self.top_k = int(top_k)
        self.exposure_weight = float(exposure_weight)
        self.novelty_weight = float(novelty_weight)
        self.coverage_weight = float(coverage_weight)
        self.distance_margin_ratio = float(distance_margin_ratio)
        self.distance_mode = str(distance_mode)
        self.q_distance_weight = float(q_distance_weight)
        self.difficulty_distance_weight = float(difficulty_distance_weight)
        self.discrimination_distance_weight = float(
            discrimination_distance_weight
        )
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.distance_margin_ratio < 0:
            raise ValueError("distance_margin_ratio must be non-negative")
        if self.distance_mode not in {"euclidean", "block_mse"}:
            raise ValueError(
                "distance_mode must be either 'euclidean' or 'block_mse'"
            )
        for label, value in {
            "exposure_weight": self.exposure_weight,
            "novelty_weight": self.novelty_weight,
            "coverage_weight": self.coverage_weight,
            "q_distance_weight": self.q_distance_weight,
            "difficulty_distance_weight": self.difficulty_distance_weight,
            "discrimination_distance_weight": self.discrimination_distance_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{label} must be non-negative")

        self.global_exposure: Counter[int] = Counter()
        self.metadata = PolicyMetadata(
            name=self.name,
            implementation="checkpoint_plus_conservative_reranker",
            selection_model="lstm_ddpg_topk_margin_exposure_reranker",
            evaluator_model="NCDM",
            uses_query_labels=False,
            uses_privileged_information=False,
            notes=(
                f"top_k={self.top_k}; exposure={self.exposure_weight}; "
                f"novelty={self.novelty_weight}; coverage={self.coverage_weight}; "
                f"distance_mode={self.distance_mode}; "
                f"margin_ratio={self.distance_margin_ratio}"
            ),
        )

    def _candidate_distance(
        self,
        ideal: torch.Tensor,
        q_features: torch.Tensor,
        difficulty: torch.Tensor,
        discrimination: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance_mode == "euclidean":
            candidate_features = torch.cat(
                [q_features, difficulty, discrimination],
                dim=-1,
            )
            return torch.cdist(ideal, candidate_features).squeeze(0)

        q_dim = int(self.q_matrix.shape[1])
        ideal_q = ideal[:, :q_dim]
        ideal_difficulty = ideal[:, q_dim : 2 * q_dim]
        ideal_discrimination = ideal[:, 2 * q_dim : 2 * q_dim + 1]
        q_distance = torch.mean((q_features - ideal_q) ** 2, dim=1)
        difficulty_distance = torch.mean(
            (difficulty - ideal_difficulty) ** 2,
            dim=1,
        )
        discrimination_distance = torch.mean(
            (discrimination - ideal_discrimination) ** 2,
            dim=1,
        )
        return (
            self.q_distance_weight * q_distance
            + self.difficulty_distance_weight * difficulty_distance
            + self.discrimination_distance_weight * discrimination_distance
        )

    def _novelty_scores(
        self,
        candidate_tensor: torch.Tensor,
        history_item_ids: Sequence[int],
    ) -> torch.Tensor:
        history = [
            int(item)
            for item in history_item_ids
            if 0 <= int(item) < self.valid_item_count
        ]
        if not history:
            return torch.zeros(
                candidate_tensor.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
        history_tensor = torch.tensor(
            history,
            dtype=torch.long,
            device=self.device,
        )
        candidate_q = torch.nn.functional.normalize(
            self.q_matrix[candidate_tensor],
            p=2,
            dim=1,
        )
        history_q = torch.nn.functional.normalize(
            self.q_matrix[history_tensor],
            p=2,
            dim=1,
        )
        q_similarity = torch.clamp(
            candidate_q @ history_q.transpose(0, 1),
            min=0.0,
            max=1.0,
        ).max(dim=1).values
        semantic_similarity = torch.clamp(
            self.item_bank[candidate_tensor]
            @ self.item_bank[history_tensor].transpose(0, 1),
            min=-1.0,
            max=1.0,
        ).max(dim=1).values
        semantic_similarity = (semantic_similarity + 1.0) * 0.5
        return 1.0 - 0.5 * (q_similarity + semantic_similarity)

    def _coverage_gain_scores(
        self,
        candidate_tensor: torch.Tensor,
        history_item_ids: Sequence[int],
    ) -> torch.Tensor:
        q_dim = int(self.q_matrix.shape[1])
        history = [
            int(item)
            for item in history_item_ids
            if 0 <= int(item) < self.valid_item_count
        ]
        if history:
            history_tensor = torch.tensor(
                history,
                dtype=torch.long,
                device=self.device,
            )
            covered = self.q_matrix[history_tensor].sum(dim=0) > 0
        else:
            covered = torch.zeros(q_dim, dtype=torch.bool, device=self.device)
        candidate_concepts = self.q_matrix[candidate_tensor] > 0
        new_concepts = candidate_concepts & (~covered.unsqueeze(0))
        return new_concepts.float().sum(dim=1) / max(1, q_dim)

    def select(
        self,
        candidate_item_ids: Sequence[int],
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
        context: dict[str, Any],
    ) -> int:
        candidates = self._validate_selection_inputs(candidate_item_ids, context)
        if self.actor is None:
            selected = candidates[0]
            self.global_exposure[selected] += 1
            return selected

        with torch.no_grad():
            ideal = self._sync_history(history_item_ids, history_responses)
            candidate_tensor = torch.tensor(
                candidates,
                dtype=torch.long,
                device=self.device,
            )
            q_features, difficulty, discrimination = self._candidate_feature_blocks(
                candidate_tensor
            )
            distance = self._candidate_distance(
                ideal,
                q_features,
                difficulty,
                discrimination,
            )
            top_k = min(self.top_k, len(candidates))
            top_indices = torch.argsort(distance, stable=True)[:top_k]
            top_candidates = candidate_tensor[top_indices]
            top_distance = distance[top_indices]

            best_distance = top_distance[0]
            margin = torch.clamp(
                torch.abs(best_distance) * self.distance_margin_ratio,
                min=1e-8,
            )
            eligible_mask = top_distance <= best_distance + margin
            eligible_candidates = top_candidates[eligible_mask]
            eligible_distance = top_distance[eligible_mask]

            exposure_penalty = torch.tensor(
                [
                    math.log1p(self.global_exposure[int(item)])
                    for item in eligible_candidates.detach().cpu().tolist()
                ],
                dtype=torch.float32,
                device=self.device,
            )
            score = -eligible_distance - self.exposure_weight * exposure_penalty

            if self.novelty_weight > 0:
                score = score + self.novelty_weight * self._novelty_scores(
                    eligible_candidates,
                    history_item_ids,
                )
            if self.coverage_weight > 0:
                score = score + self.coverage_weight * self._coverage_gain_scores(
                    eligible_candidates,
                    history_item_ids,
                )

            best_local_index = int(torch.argmax(score).item())
            selected = int(eligible_candidates[best_local_index].item())

        self.global_exposure[selected] += 1
        return selected
