"""NCDM-native Base and Set C3DQN evaluation policies."""
from __future__ import annotations

from typing import Any, Sequence
import random

import torch

from agents.ncdm_c3dqn_trainer_v2 import (
    forward_q_network,
    load_c3dqn_checkpoint,
    load_set_c3dqn_checkpoint,
)
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork

FORBIDDEN_CONTEXT_KEYS = {
    "query_item_ids",
    "query_responses",
    "query_labels",
    "future_responses",
    "future_item_ids",
    "candidate_response_lookup",
    "query_loss",
}


class RandomNCDMPolicy(BaseCATPolicy):
    name = "Random-NCDM"
    metadata = PolicyMetadata(
        name=name,
        implementation="ncdm_native",
        selection_model="random",
        evaluator_model="NCDM",
    )

    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)
        self.rng = random.Random(f"{student_id}:{seed}")

    def select(
        self,
        candidate_item_ids: Sequence[int],
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
        context: dict[str, Any],
    ) -> int:
        del history_item_ids, history_responses, context
        return int(self.rng.choice(list(candidate_item_ids)))


class C3DQNNCDMPolicy(BaseCATPolicy):
    name = "C3DQN-NCDM"
    metadata = PolicyMetadata(
        name=name,
        implementation="ncdm_native",
        selection_model="candidate_conditioned_attention_dueling_double_dqn",
        evaluator_model="NCDM",
        uses_query_labels=False,
        uses_privileged_information=False,
    )
    checkpoint_loader = staticmethod(load_c3dqn_checkpoint)
    expected_network_type = CandidateConditionedNCDMQNetwork

    def __init__(
        self,
        checkpoint_path,
        ncdm,
        q_matrix: torch.Tensor,
        device: str | torch.device = "cpu",
        expected_protocol_config: dict[str, Any] | None = None,
        network: CandidateConditionedNCDMQNetwork | SetConditionedNCDMQNetwork | None = None,
        cache: NCDMItemFeatureCache | None = None,
        selection_horizon: int | None = None,
        alpha_fit: dict | None = None,
        candidate_pool_config: dict | None = None,
        candidate_chunk_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.ncdm = ncdm.to(self.device).eval() if hasattr(ncdm, "to") else ncdm
        self.q_matrix = q_matrix.to(self.device).float()
        protocol = dict(expected_protocol_config or {})

        if network is None:
            self.network, metadata = self.checkpoint_loader(
                checkpoint_path,
                ncdm=self.ncdm,
                q_matrix=self.q_matrix,
                device=self.device,
                expected_protocol_config=protocol,
            )
            if not isinstance(self.network, self.expected_network_type):
                raise TypeError("checkpoint loader returned the wrong network type")
            self.selection_horizon = int(metadata["selection_horizon"])
            self.alpha_fit = dict(metadata.get("alpha_fit") or {})
            self.candidate_pool_config = dict(
                candidate_pool_config
                or protocol.get("candidate_pool_config")
                or metadata.get("candidate_pool_config")
                or {}
            )
            self.candidate_chunk_size = candidate_chunk_size
            if self.candidate_chunk_size is None:
                self.candidate_chunk_size = metadata.get("candidate_chunk_size")
        else:
            if not isinstance(network, self.expected_network_type):
                raise TypeError("injected network has the wrong policy architecture")
            self.network = network.to(self.device).eval()
            self.selection_horizon = int(
                selection_horizon or protocol.get("selection_horizon", 5)
            )
            self.alpha_fit = dict(
                alpha_fit
                or {
                    "initial_steps": 8,
                    "incremental_steps": 3,
                    "lr": 0.05,
                    "early_stop_tol": 1e-5,
                }
            )
            self.candidate_pool_config = dict(
                candidate_pool_config
                or protocol.get("candidate_pool_config")
                or {}
            )
            self.candidate_chunk_size = candidate_chunk_size

        self.cache = cache or NCDMItemFeatureCache(
            self.ncdm,
            self.q_matrix,
            self.device,
        )
        self.prefilter = NCDMCandidatePrefilter(
            q_matrix=self.q_matrix,
            feature_cache=self.cache,
            ncdm=self.ncdm,
            config=self.candidate_pool_config,
        )
        self._alpha: torch.Tensor | None = None
        self._history_items: list[int] = []
        self._history_responses: list[float] = []
        self.last_prefiltered_candidate_ids: list[int] = []
        self.last_prefilter_summary: dict[str, Any] = {}

    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)
        self._alpha = None
        self._history_items = []
        self._history_responses = []
        self.last_prefiltered_candidate_ids = []
        self.last_prefilter_summary = {}

    def _validate_context(self, context: dict[str, Any] | None) -> None:
        leaked = FORBIDDEN_CONTEXT_KEYS & set((context or {}).keys())
        if leaked:
            raise ValueError(
                f"{self.name} policy received privileged context keys: "
                f"{sorted(leaked)}"
            )

    def _fit_alpha_cached(
        self,
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
    ) -> torch.Tensor:
        items = [int(item) for item in history_item_ids]
        responses = [float(response) for response in history_responses]
        if items == self._history_items and responses == self._history_responses:
            if self._alpha is None:
                raise RuntimeError("alpha cache is inconsistent")
            return self._alpha

        is_continuation = (
            self._alpha is not None
            and items[:-1] == self._history_items
            and responses[:-1] == self._history_responses
        )
        initial_alpha = self._alpha if is_continuation else None
        steps = int(
            self.alpha_fit.get(
                "incremental_steps" if is_continuation else "initial_steps",
                self.alpha_fit.get("steps", 3 if is_continuation else 8),
            )
        )
        with torch.enable_grad():
            alpha = fit_student_alpha(
                self.ncdm,
                self.q_matrix,
                items,
                responses,
                initial_alpha=initial_alpha,
                steps=steps,
                lr=float(self.alpha_fit.get("lr", 0.05)),
                early_stop_tol=float(self.alpha_fit.get("early_stop_tol", 1e-5)),
                grad_clip=self.alpha_fit.get("grad_clip"),
                device=self.device,
            )
        self._alpha = alpha.detach()
        self._history_items = items
        self._history_responses = responses
        return alpha

    def select(
        self,
        candidate_item_ids: Sequence[int],
        history_item_ids: Sequence[int],
        history_responses: Sequence[float],
        context: dict[str, Any] | None = None,
    ) -> int:
        self._validate_context(context)
        if not candidate_item_ids:
            raise ValueError(f"{self.name} requires at least one candidate item")
        if not history_item_ids:
            raise ValueError(f"{self.name} requires warm-start history")

        alpha = self._fit_alpha_cached(history_item_ids, history_responses)
        with torch.no_grad():
            mastery = torch.sigmoid(alpha).squeeze(0)
            history_tensor = torch.as_tensor(
                history_item_ids,
                dtype=torch.long,
                device=self.device,
            )
            coverage_count = self.cache.q_masks[history_tensor].sum(dim=0)
            filtered, summary = self.prefilter.select(
                candidate_item_ids,
                alpha,
                mastery,
                coverage_count,
            )
            self.last_prefiltered_candidate_ids = list(filtered)
            self.last_prefilter_summary = dict(summary)
            policy_step = int(
                (context or {}).get("policy_step", len(history_item_ids))
            )
            row = {
                "history_item_ids": list(history_item_ids),
                "history_responses": list(history_responses),
                "candidate_item_ids": list(filtered),
                "mastery": mastery.detach().cpu().tolist(),
                "coverage_count": coverage_count.detach().cpu().tolist(),
                "coverage": (
                    coverage_count / float(self.selection_horizon)
                ).clamp(0, 1).detach().cpu().tolist(),
                "policy_step": policy_step,
                "selected_item_id": int(filtered[0]),
            }
            batch = pad_c3dqn_batch(
                [row],
                self.cache,
                self.selection_horizon,
                require_exact_coverage=isinstance(
                    self.network,
                    SetConditionedNCDMQNetwork,
                ),
            )
            q_values, _ = forward_q_network(
                self.network,
                batch,
                chunk_size=self.candidate_chunk_size,
            )
            selected_index = int(q_values.argmax(dim=1).item())
            return int(filtered[selected_index])


class SetC3DQNNCDMPolicy(C3DQNNCDMPolicy):
    name = "Set-C3DQN-NCDM"
    metadata = PolicyMetadata(
        name=name,
        implementation="ncdm_native",
        selection_model="set_conditioned_candidate_attention_dueling_double_dqn",
        evaluator_model="NCDM",
        uses_query_labels=False,
        uses_privileged_information=False,
    )
    checkpoint_loader = staticmethod(load_set_c3dqn_checkpoint)
    expected_network_type = SetConditionedNCDMQNetwork
