"""DDPG policies for NCDM-based adaptive item selection."""
from __future__ import annotations

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
            # The evaluator should call select once per new response. Replaying the
            # final event keeps behavior defined for repeated diagnostic calls.
            self.hx = self.cx = None
            self._processed_items = []
            self._processed_responses = []
            for item, response in zip(items, responses):
                ideal = self._process_event(item, response)
                self._processed_items.append(item)
                self._processed_responses.append(response)
        assert ideal is not None
        return ideal

    def select(
        self,
        candidate_item_ids: Sequence[int],
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
        context: dict[str, Any],
    ) -> int:
        candidates = [int(item) for item in candidate_item_ids]
        if not candidates:
            raise ValueError(f"{self.name} requires at least one candidate")
        leaked = FORBIDDEN_CONTEXT_KEYS & set((context or {}).keys())
        if leaked:
            raise ValueError(
                f"{self.name} received privileged context keys: {sorted(leaked)}"
            )
        if self.actor is None:
            return candidates[0]
        invalid = [
            item for item in candidates if item < 0 or item >= self.valid_item_count
        ]
        if invalid:
            raise ValueError(
                f"candidate IDs outside common asset bounds: {invalid[:5]}"
            )

        with torch.no_grad():
            ideal = self._sync_history(history_item_ids, history_responses)
            candidate_tensor = torch.tensor(
                candidates, dtype=torch.long, device=self.device
            )
            candidate_features = torch.cat(
                [
                    self.q_matrix[candidate_tensor],
                    torch.sigmoid(
                        self.ncdm.k_difficulty(candidate_tensor)
                    ),
                    torch.sigmoid(
                        self.ncdm.e_discrimination(candidate_tensor)
                    ),
                ],
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
