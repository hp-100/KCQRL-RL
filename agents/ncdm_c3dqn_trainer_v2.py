"""Corrected unified runtime for Base and Set C3DQN-NCDM.

The legacy trainer remains available for backward imports. This module preserves
its public API while replacing model forwarding, mixed-terminal Double DQN,
alpha warm starts, AMP, checkpoint dispatch, and candidate-pool handling.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import csv
import math
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from agents.ncdm_c3dqn_trainer import (
    C3DQNTransition,
    NCDMC3DQNTrainer as LegacyNCDMC3DQNTrainer,
    build_checkpoint_metadata as build_base_checkpoint_metadata,
    load_c3dqn_checkpoint as legacy_load_c3dqn_checkpoint,
    masked_argmax,
    validate_c3dqn_checkpoint_metadata,
)
from evaluation.metrics import auc_score, brier_score, gini, nll_score
from evaluation.offline_eval import CATOfflineEvaluator, StudentSequence
from evaluation.protocol import make_student_split
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import (
    RELATIVE_FEATURE_NAMES,
    SetConditionedNCDMQNetwork,
)
from reward.ncdm_diagnostic_reward import (
    compute_ncdm_diagnostic_reward,
    mastery_entropy,
)

BASE_ARCHITECTURE = "candidate_conditioned_attention_dueling_double_dqn"
SET_ARCHITECTURE = "set_conditioned_candidate_attention_dueling_double_dqn"


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def forward_q_network(
    network: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(network, SetConditionedNCDMQNetwork):
        coverage_count = batch.get("coverage_count")
        if coverage_count is None:
            raise ValueError("Set-C3DQN requires exact coverage_count")
        kwargs = {
            "history_features": batch["history_features"],
            "history_mask": batch["history_mask"],
            "candidate_features": batch["candidate_features"],
            "candidate_mask": batch["candidate_mask"],
            "global_features": batch["global_features"],
            "coverage_count": coverage_count,
        }
        if chunk_size is not None:
            return network.forward_chunked(
                **kwargs,
                chunk_size=int(chunk_size),
            )
        return network(**kwargs)
    return network(
        batch["history_features"],
        batch["history_mask"],
        batch["candidate_features"],
        batch["candidate_mask"],
        batch["global_features"],
    )


def _sample_from_transition(
    transition: C3DQNTransition,
    *,
    next_state: bool,
) -> dict[str, Any]:
    candidates = (
        transition.next_candidate_item_ids
        if next_state
        else transition.candidate_item_ids
    )
    if not candidates:
        raise ValueError("C3DQN batch sample requires at least one candidate")
    return {
        "history_item_ids": (
            transition.next_history_item_ids
            if next_state
            else transition.history_item_ids
        ),
        "history_responses": (
            transition.next_history_responses
            if next_state
            else transition.history_responses
        ),
        "candidate_item_ids": candidates,
        "mastery": transition.next_mastery if next_state else transition.mastery,
        "coverage": (
            transition.next_coverage if next_state else transition.coverage
        ),
        "coverage_count": (
            transition.next_coverage_count
            if next_state
            else transition.coverage_count
        ),
        "policy_step": (
            transition.next_policy_step
            if next_state
            else transition.policy_step
        ),
        "selected_item_id": (
            candidates[0] if next_state else transition.selected_item_id
        ),
    }


def transitions_to_batches(
    transitions: Sequence[C3DQNTransition],
    cache: NCDMItemFeatureCache,
    selection_horizon: int,
    *,
    require_exact_coverage: bool,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    current_rows = [
        _sample_from_transition(transition, next_state=False)
        for transition in transitions
    ]
    batch = pad_c3dqn_batch(
        current_rows,
        cache,
        selection_horizon,
        require_exact_coverage=require_exact_coverage,
    )

    next_rows: list[dict[str, Any]] = []
    non_terminal: list[int] = []
    for index, transition in enumerate(transitions):
        if not transition.done and transition.next_candidate_item_ids:
            non_terminal.append(index)
            next_rows.append(_sample_from_transition(transition, next_state=True))
    next_batch = (
        pad_c3dqn_batch(
            next_rows,
            cache,
            selection_horizon,
            require_exact_coverage=require_exact_coverage,
        )
        if next_rows
        else None
    )
    rewards = torch.tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=cache.device,
    )
    dones = torch.tensor(
        [transition.done for transition in transitions],
        dtype=torch.bool,
        device=cache.device,
    )
    indices = torch.tensor(
        non_terminal,
        dtype=torch.long,
        device=cache.device,
    )
    return batch, next_batch, rewards, dones, indices


def compute_double_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    batch: dict[str, torch.Tensor],
    next_batch: dict[str, torch.Tensor] | None,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    non_terminal_indices: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_values, _ = forward_q_network(
        online_net,
        batch,
        chunk_size=chunk_size,
    )
    chosen_q = q_values.gather(
        1,
        batch["action_index"].view(-1, 1),
    ).squeeze(1)

    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        next_action_mean = 0.0
        if next_batch is not None:
            if next_batch["history_features"].shape[0] != int(
                non_terminal_indices.numel()
            ):
                raise ValueError(
                    "next batch rows do not match non-terminal indices"
                )
            next_online_q, _ = forward_q_network(
                online_net,
                next_batch,
                chunk_size=chunk_size,
            )
            next_actions = masked_argmax(
                next_online_q,
                next_batch["candidate_mask"],
            )
            next_target_q, _ = forward_q_network(
                target_net,
                next_batch,
                chunk_size=chunk_size,
            )
            evaluated = next_target_q.gather(
                1,
                next_actions.view(-1, 1),
            ).squeeze(1)
            next_q[non_terminal_indices] = evaluated
            next_action_mean = float(next_actions.float().mean().item())
        target = rewards + float(gamma) * next_q

    loss = F.smooth_l1_loss(chosen_q, target)
    return loss, {
        "mean_q": float(chosen_q.detach().mean().item()),
        "target_q_mean": float(target.detach().mean().item()),
        "next_q_mean": float(next_q.detach().mean().item()),
        "next_action_mean": next_action_mean,
    }


def load_c3dqn_checkpoint(
    checkpoint_path: str | Path,
    *,
    ncdm,
    q_matrix: torch.Tensor,
    device: str | torch.device = "cpu",
    expected_protocol_config: dict[str, Any] | None = None,
):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    metadata = dict(checkpoint.get("metadata") or {})
    if metadata.get("actor_architecture", BASE_ARCHITECTURE) != BASE_ARCHITECTURE:
        raise ValueError("C3DQN checkpoint architecture mismatch")
    return legacy_load_c3dqn_checkpoint(
        checkpoint_path,
        ncdm=ncdm,
        q_matrix=q_matrix,
        device=device,
        expected_protocol_config=expected_protocol_config,
    )


def build_set_checkpoint_metadata(
    *,
    knowledge_dim: int,
    selection_horizon: int,
    warm_start_items: int,
    alpha_fit: dict,
    reward_config: dict,
    model_config: dict,
    candidate_pool_config: dict,
    ncdm_item_count: int,
    q_matrix_item_count: int,
    training_seed: int,
    validation_metrics: dict,
    epoch: int,
    strict_item_count_check: bool,
    requested_amp: bool,
    effective_amp: bool,
) -> dict[str, Any]:
    metadata = build_base_checkpoint_metadata(
        knowledge_dim=knowledge_dim,
        selection_horizon=selection_horizon,
        warm_start_items=warm_start_items,
        alpha_fit=alpha_fit,
        reward_config=reward_config,
        model_config=model_config,
        candidate_pool_config=candidate_pool_config,
        ncdm_item_count=ncdm_item_count,
        q_matrix_item_count=q_matrix_item_count,
        training_seed=training_seed,
        validation_metrics=validation_metrics,
        epoch=epoch,
        strict_item_count_check=strict_item_count_check,
    )
    relative_enabled = bool(model_config.get("use_relative_features", True))
    metadata.update(
        {
            "actor_architecture": SET_ARCHITECTURE,
            "candidate_set_encoder": str(
                model_config.get("candidate_set_encoder", "isab")
            ),
            "num_set_layers": int(model_config.get("num_set_layers", 1)),
            "num_inducing_points": int(
                model_config.get("num_inducing_points", 16)
            ),
            "set_attention_heads": int(
                model_config.get(
                    "set_attention_heads",
                    model_config.get("n_heads", 4),
                )
            ),
            "use_relative_features": relative_enabled,
            "relative_feature_names": (
                list(RELATIVE_FEATURE_NAMES) if relative_enabled else []
            ),
            "relative_feature_dim": 5 if relative_enabled else 0,
            "set_pool_in_value_head": bool(
                model_config.get("set_pool_in_value_head", True)
            ),
            "full_attention_max_candidates": int(
                model_config.get("full_attention_max_candidates", 128)
            ),
            "debug_mode": bool(model_config.get("debug_mode", False)),
            "requested_amp": bool(requested_amp),
            "effective_amp": bool(effective_amp),
        }
    )
    return metadata


def load_set_c3dqn_checkpoint(
    checkpoint_path: str | Path,
    *,
    ncdm,
    q_matrix: torch.Tensor,
    device: str | torch.device = "cpu",
    expected_protocol_config: dict[str, Any] | None = None,
):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    metadata = dict(checkpoint.get("metadata") or {})
    if metadata.get("actor_architecture") != SET_ARCHITECTURE:
        raise ValueError("Set-C3DQN checkpoint architecture mismatch")
    required = [
        "candidate_set_encoder",
        "num_set_layers",
        "num_inducing_points",
        "set_attention_heads",
        "use_relative_features",
        "relative_feature_dim",
        "set_pool_in_value_head",
        "full_attention_max_candidates",
        "debug_mode",
        "candidate_pool_config",
        "alpha_fit",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"Set-C3DQN checkpoint missing metadata fields: {missing}"
        )
    expected = dict(expected_protocol_config or {})
    expected.setdefault("q_matrix_item_count", int(q_matrix.shape[0]))
    expected.setdefault(
        "ncdm_item_count",
        int(ncdm.k_difficulty.num_embeddings),
    )
    expected.setdefault("knowledge_dim", int(q_matrix.shape[1]))
    validate_c3dqn_checkpoint_metadata(metadata, expected)

    config = dict(metadata.get("model_config") or {})
    network = SetConditionedNCDMQNetwork(
        int(metadata["knowledge_dim"]),
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        num_history_layers=int(config.get("num_history_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        candidate_set_encoder=str(metadata["candidate_set_encoder"]),
        num_set_layers=int(metadata["num_set_layers"]),
        num_inducing_points=int(metadata["num_inducing_points"]),
        set_attention_heads=int(metadata["set_attention_heads"]),
        use_relative_features=bool(metadata["use_relative_features"]),
        set_pool_in_value_head=bool(metadata["set_pool_in_value_head"]),
        full_attention_max_candidates=int(
            metadata["full_attention_max_candidates"]
        ),
        debug_mode=bool(metadata["debug_mode"]),
    ).to(device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise ValueError("Set-C3DQN checkpoint missing model_state_dict")
    network.load_state_dict(state, strict=True)
    network.eval()
    return network, metadata


class NCDMC3DQNTrainer(LegacyNCDMC3DQNTrainer):
    """Legacy-compatible trainer with corrected Base/Set execution paths."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        candidate_pool_config = dict(
            kwargs.get("candidate_pool_config")
            or {"prefilter_enabled": True, "prefilter_top_k": 256}
        )
        requested_amp = bool(kwargs.get("requested_amp", False))
        candidate_chunk_size = kwargs.get("candidate_chunk_size")
        super().__init__(*args, **kwargs)
        self.candidate_pool_config = candidate_pool_config
        self.prefilter = (
            NCDMCandidatePrefilter(
                q_matrix=self.q_matrix,
                feature_cache=self.cache,
                ncdm=self.ncdm,
                config=self.candidate_pool_config,
            )
            if self.ncdm is not None
            else None
        )
        self.is_set_model = isinstance(
            self.online_net,
            SetConditionedNCDMQNetwork,
        )
        self.requested_amp = requested_amp
        self.use_amp = bool(
            requested_amp and self.cache.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.candidate_chunk_size = (
            int(candidate_chunk_size)
            if candidate_chunk_size is not None
            else None
        )

    def _fit_alpha(
        self,
        history_items: Sequence[int],
        history_responses: Sequence[float],
        initial_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ncdm is None:
            return torch.zeros(
                (1, self.cache.knowledge_dim),
                device=self.cache.device,
            )
        start_time = time.perf_counter()
        config = dict(self.alpha_fit)
        config.pop("steps", None)
        initial_steps = int(config.pop("initial_steps", 8))
        incremental_steps = int(config.pop("incremental_steps", 3))
        steps = initial_steps if initial_alpha is None else incremental_steps
        alpha = fit_student_alpha(
            self.ncdm,
            self.q_matrix,
            history_items,
            history_responses,
            initial_alpha=initial_alpha,
            steps=steps,
            device=self.cache.device,
            **config,
        )
        self.time_acc["alpha_fit_seconds"] += time.perf_counter() - start_time
        return alpha

    def _filter_candidates(
        self,
        candidates: Sequence[int],
        alpha: torch.Tensor,
        mastery: torch.Tensor,
        coverage_count: torch.Tensor,
    ) -> tuple[list[int], dict[str, Any]]:
        if self.prefilter is None:
            filtered = [int(item_id) for item_id in candidates]
            return filtered, {
                "raw_candidate_count": len(filtered),
                "filtered_candidate_count": len(filtered),
                "candidate_prefilter_seconds": 0.0,
            }
        filtered, summary = self.prefilter.select(
            candidates,
            alpha,
            mastery,
            coverage_count,
        )
        self.time_acc["candidate_prefilter_seconds"] += float(
            summary.get("candidate_prefilter_seconds", 0.0)
        )
        return filtered, summary

    def _select(
        self,
        history_items: Sequence[int],
        history_responses: Sequence[float],
        candidates: Sequence[int],
        alpha: torch.Tensor,
        mastery: torch.Tensor,
        coverage_count: torch.Tensor,
        policy_step: int,
        epsilon: float | None = None,
        *,
        filtered_candidates: Sequence[int] | None = None,
        prefilter_summary: dict[str, Any] | None = None,
    ) -> tuple[int, list[int], dict[str, Any]]:
        if filtered_candidates is None:
            filtered, summary = self._filter_candidates(
                candidates,
                alpha,
                mastery,
                coverage_count,
            )
        else:
            filtered = [int(item_id) for item_id in filtered_candidates]
            summary = dict(prefilter_summary or {})
        if not filtered:
            raise ValueError("candidate prefilter produced an empty action set")
        actual_epsilon = self._epsilon() if epsilon is None else float(epsilon)
        if random.random() < actual_epsilon:
            return int(random.choice(filtered)), filtered, summary

        row = {
            "history_item_ids": list(history_items),
            "history_responses": list(history_responses),
            "candidate_item_ids": filtered,
            "mastery": mastery.detach().cpu().tolist(),
            "coverage_count": coverage_count.detach().cpu().tolist(),
            "coverage": (
                coverage_count / max(1, self.selection_horizon)
            ).clamp(0, 1).detach().cpu().tolist(),
            "policy_step": int(policy_step),
            "selected_item_id": int(filtered[0]),
        }
        feature_start = time.perf_counter()
        batch = pad_c3dqn_batch(
            [row],
            self.cache,
            self.selection_horizon,
            require_exact_coverage=self.is_set_model,
        )
        self.time_acc["feature_build_seconds"] += (
            time.perf_counter() - feature_start
        )
        start_time = time.perf_counter()
        with torch.no_grad():
            q_values, _ = forward_q_network(
                self.online_net,
                batch,
                chunk_size=self.candidate_chunk_size,
            )
        self.time_acc["network_forward_seconds"] += (
            time.perf_counter() - start_time
        )
        index = int(q_values.argmax(dim=1).item())
        return int(filtered[index]), filtered, summary

    def update_once(self) -> dict[str, float] | None:
        if len(self.replay) < self.min_replay_size:
            return None
        transitions = self.replay.sample(self.batch_size)
        feature_start = time.perf_counter()
        batch, next_batch, rewards, dones, indices = transitions_to_batches(
            transitions,
            self.cache,
            self.selection_horizon,
            require_exact_coverage=self.is_set_model,
        )
        self.time_acc["feature_build_seconds"] += (
            time.perf_counter() - feature_start
        )
        self.optim.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", enabled=True)
            if self.use_amp
            else nullcontext()
        )
        start_time = time.perf_counter()
        with amp_context:
            loss, stats = compute_double_dqn_loss(
                self.online_net,
                self.target_net,
                batch,
                next_batch,
                rewards,
                dones,
                self.gamma,
                indices,
                chunk_size=self.candidate_chunk_size,
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            self.gradient_clip,
        )
        self.scaler.step(self.optim)
        self.scaler.update()
        self.learning_steps += 1
        if (
            self.target_update_interval
            and self.learning_steps % self.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            with torch.no_grad():
                for online_parameter, target_parameter in zip(
                    self.online_net.parameters(),
                    self.target_net.parameters(),
                ):
                    target_parameter.data.mul_(1.0 - self.tau).add_(
                        online_parameter.data,
                        alpha=self.tau,
                    )
        self.time_acc["network_update_seconds"] += (
            time.perf_counter() - start_time
        )
        stats["td_loss"] = float(loss.detach().item())
        self.time_acc["update_count"] += 1.0
        self.time_acc["mean_q_sum"] += float(stats["mean_q"])
        self.time_acc["target_q_sum"] += float(stats["target_q_mean"])
        return stats

    def _run_episode(self, split):
        raw_candidates = [int(item) for item in split.candidate_items]
        response_lookup = {
            int(item): float(response)
            for item, response in zip(
                split.support_item_ids,
                split.support_responses,
            )
        }
        history_items = [int(split.warm_start_item)]
        history_responses = [float(split.warm_start_response)]
        alpha = self._fit_alpha(history_items, history_responses)
        mastery = torch.sigmoid(alpha).squeeze(0)
        coverage_count = self._coverage(history_items)
        filtered, summary = self._filter_candidates(
            raw_candidates,
            alpha,
            mastery,
            coverage_count,
        )
        transitions = []
        reward_rows = []
        selected_items = []
        losses = []

        for policy_step in range(
            min(self.selection_horizon, len(raw_candidates))
        ):
            before_nll = nll_score(
                split.query_responses,
                self._predict_query(alpha, split.query_item_ids),
            )
            item, current_filtered, current_summary = self._select(
                history_items,
                history_responses,
                raw_candidates,
                alpha,
                mastery,
                coverage_count,
                policy_step,
                filtered_candidates=filtered,
                prefilter_summary=summary,
            )
            before_items = list(history_items)
            before_responses = list(history_responses)
            before_mastery = mastery.clone()
            before_coverage = coverage_count.clone()
            raw_candidates.remove(item)
            history_items.append(item)
            history_responses.append(response_lookup[item])
            alpha_after = self._fit_alpha(
                history_items,
                history_responses,
                initial_alpha=alpha,
            )
            mastery_after = torch.sigmoid(alpha_after).squeeze(0)
            coverage_after = self._coverage(history_items)
            after_nll = nll_score(
                split.query_responses,
                self._predict_query(alpha_after, split.query_item_ids),
            )
            reward_start = time.perf_counter()
            reward = compute_ncdm_diagnostic_reward(
                before_nll,
                after_nll,
                mastery,
                mastery_after,
                self.cache.q_masks[item],
                before_coverage,
                self.reward_cfg,
            )
            self.time_acc["reward_seconds"] += (
                time.perf_counter() - reward_start
            )
            components = {
                "prediction_gain": reward.prediction_gain,
                "diagnosis_gain": reward.diagnosis_gain,
                "coverage_gain": reward.coverage_gain,
                "total": reward.total,
            }
            done = (
                not raw_candidates
                or policy_step + 1 >= self.selection_horizon
            )
            if done:
                next_filtered, next_summary = [], {}
            else:
                next_filtered, next_summary = self._filter_candidates(
                    raw_candidates,
                    alpha_after,
                    mastery_after,
                    coverage_after,
                )
            transition = C3DQNTransition(
                before_items,
                before_responses,
                list(current_filtered),
                before_mastery.detach().cpu().tolist(),
                (
                    before_coverage / max(1, self.selection_horizon)
                ).clamp(0, 1).detach().cpu().tolist(),
                policy_step,
                item,
                reward.total,
                components,
                list(history_items),
                list(history_responses),
                list(next_filtered),
                mastery_after.detach().cpu().tolist(),
                (
                    coverage_after / max(1, self.selection_horizon)
                ).clamp(0, 1).detach().cpu().tolist(),
                policy_step + 1,
                done,
                before_coverage.detach().cpu().tolist(),
                coverage_after.detach().cpu().tolist(),
                int(
                    current_summary.get(
                        "raw_candidate_count",
                        len(raw_candidates) + 1,
                    )
                ),
                int(
                    current_summary.get(
                        "filtered_candidate_count",
                        len(current_filtered),
                    )
                ),
            )
            self.replay.push(transition)
            transitions.append(transition)
            reward_rows.append(components)
            selected_items.append(item)
            for _ in range(self.updates_per_environment_step):
                update_stats = self.update_once()
                if update_stats:
                    losses.append(update_stats["td_loss"])
            alpha = alpha_after
            mastery = mastery_after
            coverage_count = coverage_after
            filtered = next_filtered
            summary = next_summary
        return transitions, reward_rows, selected_items, losses

    def train(
        self,
        sequences_csv: str | Path,
        *,
        epochs: int = 1,
        train_ratio: float = 0.8,
        max_students: int | None = None,
        query_ratio: float = 0.2,
        min_query_items: int = 2,
    ) -> list[dict[str, float]]:
        loader = CATOfflineEvaluator({"assets": {}}, debug=False)
        sequences = loader._load_sequences(Path(sequences_csv))
        if max_students:
            sequences = sequences[: int(max_students)]
        students = list(sequences)
        random.Random(self.seed).shuffle(students)
        split_index = max(1, int(round(len(students) * float(train_ratio))))
        train_sequences = students[:split_index]
        validation_sequences = students[split_index:] or students[:1]

        history: list[dict[str, float]] = []
        best_nll = float("inf")
        for epoch in range(1, int(epochs) + 1):
            epoch_start = time.perf_counter()
            self.time_acc = defaultdict(float)
            reward_rows: list[dict[str, float]] = []
            selected_all: list[int] = []
            losses: list[float] = []
            for sequence in train_sequences:
                split, _ = make_student_split(
                    sequence.student_id,
                    sequence.item_ids,
                    sequence.responses,
                    seed=self.seed + epoch,
                    valid_count=self.cache.item_count,
                    query_ratio=query_ratio,
                    min_query_items=min_query_items,
                )
                if not split:
                    continue
                _, rewards, selected, episode_losses = self._run_episode(split)
                reward_rows.extend(rewards)
                selected_all.extend(selected)
                losses.extend(episode_losses)

            validation_start = time.perf_counter()
            validation = self.validate(
                validation_sequences,
                seed=self.seed + epoch,
                query_ratio=query_ratio,
                min_query_items=min_query_items,
            )
            self.time_acc["validation_seconds"] += (
                time.perf_counter() - validation_start
            )
            exposure = Counter(selected_all)
            update_count = self.time_acc.get("update_count", 0.0)
            mean_q = (
                self.time_acc.get("mean_q_sum", 0.0) / update_count
                if update_count > 0
                else float("nan")
            )
            target_q_mean = (
                self.time_acc.get("target_q_sum", 0.0) / update_count
                if update_count > 0
                else float("nan")
            )
            row = {
                "epoch": epoch,
                "mean_total_reward": _mean(
                    [reward["total"] for reward in reward_rows]
                ),
                "mean_prediction_reward": _mean(
                    [reward["prediction_gain"] for reward in reward_rows]
                ),
                "mean_diagnosis_reward": _mean(
                    [reward["diagnosis_gain"] for reward in reward_rows]
                ),
                "mean_coverage_reward": _mean(
                    [reward["coverage_gain"] for reward in reward_rows]
                ),
                "td_loss": _mean(losses),
                "mean_q": mean_q,
                "target_q_mean": target_q_mean,
                "epsilon": self._epsilon(),
                "replay_size": len(self.replay),
                "selected_unique_items": len(exposure),
                "item_exposure_gini": gini(exposure.values()),
                **validation,
                "feature_build_seconds": self.time_acc.get(
                    "feature_build_seconds",
                    0.0,
                ),
                "alpha_fit_seconds": self.time_acc.get(
                    "alpha_fit_seconds",
                    0.0,
                ),
                "reward_seconds": self.time_acc.get(
                    "reward_seconds",
                    0.0,
                ),
                "network_forward_seconds": self.time_acc.get(
                    "network_forward_seconds",
                    0.0,
                ),
                "network_update_seconds": self.time_acc.get(
                    "network_update_seconds",
                    0.0,
                ),
                "validation_seconds": self.time_acc.get(
                    "validation_seconds",
                    0.0,
                ),
                "candidate_prefilter_seconds": self.time_acc.get(
                    "candidate_prefilter_seconds",
                    0.0,
                ),
                "mean_raw_candidate_count": _mean(
                    [
                        transition.raw_candidate_count
                        for transition in self.replay._data
                    ]
                ),
                "mean_filtered_candidate_count": _mean(
                    [
                        transition.filtered_candidate_count
                        for transition in self.replay._data
                    ]
                ),
                "total_epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            self._write_history(history)
            if row["validation_query_nll"] <= best_nll:
                best_nll = row["validation_query_nll"]
                self.save_checkpoint(row, epoch)
        return history

    def validate(
        self,
        sequences: Sequence[StudentSequence],
        *,
        seed: int,
        query_ratio: float,
        min_query_items: int,
    ) -> dict[str, float]:
        labels: list[float] = []
        probabilities: list[float] = []
        entropies: list[float] = []
        coverages: list[float] = []
        for sequence in sequences:
            split, _ = make_student_split(
                sequence.student_id,
                sequence.item_ids,
                sequence.responses,
                seed=seed,
                valid_count=self.cache.item_count,
                query_ratio=query_ratio,
                min_query_items=min_query_items,
            )
            if not split:
                continue
            raw_candidates = [int(item) for item in split.candidate_items]
            response_lookup = {
                int(item): float(response)
                for item, response in zip(
                    split.support_item_ids,
                    split.support_responses,
                )
            }
            history_items = [int(split.warm_start_item)]
            history_responses = [float(split.warm_start_response)]
            alpha = self._fit_alpha(history_items, history_responses)
            mastery = torch.sigmoid(alpha).squeeze(0)
            coverage_count = self._coverage(history_items)
            filtered, summary = self._filter_candidates(
                raw_candidates,
                alpha,
                mastery,
                coverage_count,
            )
            for policy_step in range(
                min(self.selection_horizon, len(raw_candidates))
            ):
                item, _, _ = self._select(
                    history_items,
                    history_responses,
                    raw_candidates,
                    alpha,
                    mastery,
                    coverage_count,
                    policy_step,
                    epsilon=0.0,
                    filtered_candidates=filtered,
                    prefilter_summary=summary,
                )
                raw_candidates.remove(item)
                history_items.append(item)
                history_responses.append(response_lookup[item])
                alpha = self._fit_alpha(
                    history_items,
                    history_responses,
                    initial_alpha=alpha,
                )
                mastery = torch.sigmoid(alpha).squeeze(0)
                coverage_count = self._coverage(history_items)
                if raw_candidates:
                    filtered, summary = self._filter_candidates(
                        raw_candidates,
                        alpha,
                        mastery,
                        coverage_count,
                    )
            predicted = self._predict_query(alpha, split.query_item_ids)
            labels.extend(split.query_responses)
            probabilities.extend(predicted)
            entropies.append(float(mastery_entropy(mastery).item()))
            coverages.append(
                float((coverage_count > 0).float().mean().item())
            )
        return {
            "validation_query_nll": nll_score(labels, probabilities),
            "validation_query_auc": auc_score(labels, probabilities),
            "validation_query_brier": brier_score(labels, probabilities),
            "validation_mastery_entropy": _mean(entropies),
            "validation_concept_coverage": _mean(coverages),
        }

    def save_checkpoint(self, metrics: dict[str, float], epoch: int) -> None:
        common = {
            "knowledge_dim": self.cache.knowledge_dim,
            "selection_horizon": self.selection_horizon,
            "warm_start_items": 1,
            "alpha_fit": self.alpha_fit,
            "reward_config": asdict(self.reward_cfg),
            "model_config": self.model_config,
            "candidate_pool_config": self.candidate_pool_config,
            "ncdm_item_count": self.cache.ncdm_item_count,
            "q_matrix_item_count": self.cache.q_matrix_item_count,
            "training_seed": self.seed,
            "validation_metrics": metrics,
            "epoch": epoch,
            "strict_item_count_check": self.cache.strict_item_count_check,
        }
        if self.is_set_model:
            metadata = build_set_checkpoint_metadata(
                **common,
                requested_amp=self.requested_amp,
                effective_amp=self.use_amp,
            )
        else:
            metadata = build_base_checkpoint_metadata(**common)
            metadata["requested_amp"] = self.requested_amp
            metadata["effective_amp"] = self.use_amp
        torch.save(
            {
                "model_state_dict": self.online_net.state_dict(),
                "metadata": metadata,
            },
            self.out_dir / "best_checkpoint.pt",
        )

    def _write_history(self, rows: list[dict[str, float]]) -> None:
        with (self.out_dir / "training_history.csv").open(
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
