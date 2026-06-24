"""Masked Dueling Double DQN trainer for NCDM-native C3DQN."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from evaluation.metrics import auc_score, brier_score, gini, nll_score
from evaluation.offline_eval import CATOfflineEvaluator, StudentSequence
from evaluation.protocol import make_student_split, valid_item_count
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork
from reward.ncdm_diagnostic_reward import NCDMDiagnosticRewardConfig, compute_ncdm_diagnostic_reward, mastery_entropy


def forward_q_network(network, batch, *, chunk_size=None):
    if isinstance(network, SetConditionedNCDMQNetwork):
        if "coverage_count" not in batch:
            raise ValueError("Set-C3DQN batch requires coverage_count")
        if chunk_size is not None:
            return network.forward_chunked(
                history_features=batch["history_features"],
                history_mask=batch["history_mask"],
                candidate_features=batch["candidate_features"],
                candidate_mask=batch["candidate_mask"],
                global_features=batch["global_features"],
                coverage_count=batch["coverage_count"],
                chunk_size=chunk_size,
            )
        return network(
            batch["history_features"],
            batch["history_mask"],
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["global_features"],
            coverage_count=batch["coverage_count"],
        )
    return network(
        batch["history_features"],
        batch["history_mask"],
        batch["candidate_features"],
        batch["candidate_mask"],
        batch["global_features"],
    )


@dataclass
class C3DQNTransition:
    history_item_ids: list[int]
    history_responses: list[float]
    candidate_item_ids: list[int]
    mastery: list[float]
    coverage: list[float]
    policy_step: int
    selected_item_id: int
    reward: float
    reward_components: dict[str, float]
    next_history_item_ids: list[int]
    next_history_responses: list[float]
    next_candidate_item_ids: list[int]
    next_mastery: list[float]
    next_coverage: list[float]
    next_policy_step: int
    done: bool
    coverage_count: list[float] | None = None
    next_coverage_count: list[float] | None = None


class C3DQNReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = int(capacity)
        self._data: list[C3DQNTransition] = []

    def push(self, t: C3DQNTransition) -> None:
        if t.selected_item_id not in t.candidate_item_ids:
            raise ValueError("selected_item_id must belong to candidate_item_ids")
        self._data.append(t)
        if len(self._data) > self.capacity:
            self._data.pop(0)

    def sample(self, batch_size: int) -> list[C3DQNTransition]:
        return random.sample(self._data, min(batch_size, len(self._data)))

    def __len__(self) -> int:
        return len(self._data)

    def state_dict(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self._data]


def masked_argmax(q_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    if not candidate_mask.bool().any(dim=1).all():
        raise ValueError("masked_argmax received a sample with no valid candidates")
    return q_values.masked_fill(~candidate_mask.bool(), -1.0e9).argmax(dim=1)


def _samples_from_transitions(transitions: Sequence[C3DQNTransition], *, next_state: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in transitions:
        cands = t.next_candidate_item_ids if next_state else t.candidate_item_ids
        if not cands:
            if next_state:
                continue
            raise ValueError("current-state C3DQN batch requires at least one candidate")
        rows.append({
            "history_item_ids": t.next_history_item_ids if next_state else t.history_item_ids,
            "history_responses": t.next_history_responses if next_state else t.history_responses,
            "candidate_item_ids": cands,
            "mastery": t.next_mastery if next_state else t.mastery,
            "coverage": t.next_coverage if next_state else t.coverage,
            "coverage_count": (t.next_coverage_count if next_state else t.coverage_count) or (t.next_coverage if next_state else t.coverage),
            "policy_step": t.next_policy_step if next_state else t.policy_step,
            "selected_item_id": cands[0] if next_state else t.selected_item_id,
        })
    return rows


def compute_double_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    batch: dict[str, torch.Tensor],
    next_batch: dict[str, torch.Tensor] | None,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    non_terminal_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_values, _ = forward_q_network(online_net, batch)
    chosen_q = q_values.gather(1, batch["action_index"].view(-1, 1)).squeeze(1)
    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        next_action_mean = 0.0
        if next_batch is not None:
            next_online_q, _ = forward_q_network(online_net, next_batch)
            next_action = masked_argmax(next_online_q, next_batch["candidate_mask"])
            next_action_mean = float(next_action.float().mean().item())
            next_target_q, _ = forward_q_network(target_net, next_batch)
            evaluated = next_target_q.gather(1, next_action.view(-1, 1)).squeeze(1)
            if non_terminal_indices is None:
                non_terminal_indices = torch.nonzero(~dones.bool(), as_tuple=False).flatten()
            next_q[non_terminal_indices.to(rewards.device)] = evaluated
        target = rewards + float(gamma) * next_q
    loss = F.smooth_l1_loss(chosen_q, target)
    return loss, {"mean_q": float(chosen_q.mean().item()), "target_q_mean": float(target.mean().item()), "next_q_mean": float(next_q.mean().item()), "next_action_mean": next_action_mean}


def transitions_to_batches(transitions: Sequence[C3DQNTransition], cache: NCDMItemFeatureCache, selection_horizon: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = pad_c3dqn_batch(_samples_from_transitions(transitions), cache, selection_horizon)
    next_rows = []
    nt_indices = []
    for idx, t in enumerate(transitions):
        if not t.done and t.next_candidate_item_ids:
            nt_indices.append(idx)
            cands = t.next_candidate_item_ids
            next_rows.append({"history_item_ids": t.next_history_item_ids, "history_responses": t.next_history_responses, "candidate_item_ids": cands, "mastery": t.next_mastery, "coverage": t.next_coverage, "coverage_count": t.next_coverage_count, "policy_step": t.next_policy_step, "selected_item_id": cands[0]})
    next_batch = pad_c3dqn_batch(next_rows, cache, selection_horizon) if next_rows else None
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=cache.device)
    dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=cache.device)
    non_terminal_indices = torch.tensor(nt_indices, dtype=torch.long, device=cache.device)
    return batch, next_batch, rewards, dones, non_terminal_indices


def validate_c3dqn_checkpoint_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = ["actor_architecture", "knowledge_dim", "history_feature_dim", "candidate_feature_dim", "global_feature_dim", "selection_horizon", "warm_start_items", "alpha_fit", "candidate_pool_config", "q_matrix_item_count", "ncdm_item_count", "strict_item_count_check"]
    for key in keys:
        if key in expected and metadata.get(key) != expected.get(key):
            raise ValueError(f"C3DQN-NCDM checkpoint protocol mismatch for {key}: {metadata.get(key)!r} != {expected.get(key)!r}")


def build_checkpoint_metadata(*, knowledge_dim: int, selection_horizon: int, warm_start_items: int, alpha_fit: dict, reward_config: dict, model_config: dict, candidate_pool_config: dict, ncdm_item_count: int, q_matrix_item_count: int, training_seed: int, validation_metrics: dict, epoch: int, strict_item_count_check: bool = True) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"
    return {"actor_architecture": "candidate_conditioned_attention_dueling_double_dqn", "knowledge_dim": knowledge_dim, "history_feature_dim": 2 * knowledge_dim + 3, "candidate_feature_dim": 2 * knowledge_dim + 1, "global_feature_dim": 2 * knowledge_dim + 1, "selection_horizon": selection_horizon, "warm_start_items": warm_start_items, "alpha_fit": dict(alpha_fit), "reward_config": dict(reward_config), "model_config": dict(model_config), "candidate_pool_config": dict(candidate_pool_config), "ncdm_item_count": ncdm_item_count, "q_matrix_item_count": q_matrix_item_count, "effective_item_count": min(ncdm_item_count, q_matrix_item_count), "strict_item_count_check": bool(strict_item_count_check), "training_seed": training_seed, "validation_metrics": dict(validation_metrics), "epoch": epoch, "git_commit": git_commit}


def load_c3dqn_checkpoint(checkpoint_path: str | Path, *, ncdm, q_matrix: torch.Tensor, device: str | torch.device = "cpu", expected_protocol_config: dict[str, Any] | None = None) -> tuple[CandidateConditionedNCDMQNetwork, dict[str, Any]]:
    ck = torch.load(Path(checkpoint_path), map_location=device)
    meta = dict(ck.get("metadata") or {})
    if not meta:
        raise ValueError("C3DQN checkpoint missing metadata")
    if meta.get("actor_architecture") == "set_conditioned_candidate_attention_dueling_double_dqn":
        raise ValueError("base C3DQN loader cannot load Set-C3DQN checkpoint")
    expected = dict(expected_protocol_config or {})
    expected.setdefault("q_matrix_item_count", int(q_matrix.shape[0]))
    expected.setdefault("ncdm_item_count", int(ncdm.k_difficulty.num_embeddings))
    expected.setdefault("knowledge_dim", int(q_matrix.shape[1]))
    validate_c3dqn_checkpoint_metadata(meta, expected)
    cfg = dict(meta.get("model_config") or {})
    net = CandidateConditionedNCDMQNetwork(int(meta["knowledge_dim"]), d_model=int(cfg.get("d_model", 64)), n_heads=int(cfg.get("n_heads", 4)), num_history_layers=int(cfg.get("num_history_layers", 1)), dropout=float(cfg.get("dropout", 0.0))).to(device)
    state = ck.get("model_state_dict") or ck.get("state_dict")
    if state is None:
        raise ValueError("C3DQN checkpoint missing model_state_dict")
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, meta


def build_set_checkpoint_metadata(**kwargs: Any) -> dict[str, Any]:
    meta = build_checkpoint_metadata(**kwargs)
    cfg = dict(meta.get("model_config") or {})
    meta["actor_architecture"] = "set_conditioned_candidate_attention_dueling_double_dqn"
    meta.update({
        "candidate_set_encoder": cfg.get("candidate_set_encoder", "isab"),
        "num_set_layers": int(cfg.get("num_set_layers", 1)),
        "num_inducing_points": int(cfg.get("num_inducing_points", 16)),
        "set_attention_heads": int(cfg.get("set_attention_heads", cfg.get("n_heads", 4))),
        "use_relative_features": bool(cfg.get("use_relative_features", True)),
        "relative_feature_names": ["novelty_ratio", "covered_overlap_ratio", "mean_mastery_gap", "weakness_targeting", "concept_count_norm"] if bool(cfg.get("use_relative_features", True)) else [],
        "relative_feature_dim": 5 if bool(cfg.get("use_relative_features", True)) else 0,
        "set_pool_in_value_head": bool(cfg.get("set_pool_in_value_head", True)),
        "full_attention_max_candidates": int(cfg.get("full_attention_max_candidates", 512)),
        "debug_mode": bool(cfg.get("debug_mode", False)),
        "requested_amp": bool(cfg.get("requested_amp", False)),
        "effective_amp": bool(cfg.get("effective_amp", False)),
    })
    return meta


def load_set_c3dqn_checkpoint(checkpoint_path: str | Path, *, ncdm, q_matrix: torch.Tensor, device: str | torch.device = "cpu", expected_protocol_config: dict[str, Any] | None = None) -> tuple[SetConditionedNCDMQNetwork, dict[str, Any]]:
    ck = torch.load(Path(checkpoint_path), map_location=device)
    meta = dict(ck.get("metadata") or {})
    if meta.get("actor_architecture") != "set_conditioned_candidate_attention_dueling_double_dqn":
        raise ValueError("Set-C3DQN loader requires a Set-C3DQN checkpoint")
    expected = dict(expected_protocol_config or {})
    expected.setdefault("q_matrix_item_count", int(q_matrix.shape[0]))
    expected.setdefault("ncdm_item_count", int(ncdm.k_difficulty.num_embeddings))
    expected.setdefault("knowledge_dim", int(q_matrix.shape[1]))
    validate_c3dqn_checkpoint_metadata(meta, expected)
    cfg = dict(meta.get("model_config") or {})
    net = SetConditionedNCDMQNetwork(int(meta["knowledge_dim"]), d_model=int(cfg.get("d_model", 64)), n_heads=int(cfg.get("n_heads", 4)), num_history_layers=int(cfg.get("num_history_layers", 1)), dropout=float(cfg.get("dropout", 0.0)), candidate_set_encoder=str(meta.get("candidate_set_encoder", cfg.get("candidate_set_encoder", "isab"))), num_set_layers=int(meta.get("num_set_layers", cfg.get("num_set_layers", 1))), num_inducing_points=int(meta.get("num_inducing_points", cfg.get("num_inducing_points", 16))), set_attention_heads=int(meta.get("set_attention_heads", cfg.get("set_attention_heads", cfg.get("n_heads", 4)))), use_relative_features=bool(meta.get("use_relative_features", cfg.get("use_relative_features", True))), set_pool_in_value_head=bool(meta.get("set_pool_in_value_head", cfg.get("set_pool_in_value_head", True))), full_attention_max_candidates=int(meta.get("full_attention_max_candidates", cfg.get("full_attention_max_candidates", 512))), debug_mode=bool(meta.get("debug_mode", cfg.get("debug_mode", False)))).to(device)
    state = ck.get("model_state_dict") or ck.get("state_dict")
    if state is None:
        raise ValueError("Set-C3DQN checkpoint missing model_state_dict")
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, meta


class NCDMC3DQNTrainer:
    def __init__(self, online_net: CandidateConditionedNCDMQNetwork, target_net: CandidateConditionedNCDMQNetwork, cache: NCDMItemFeatureCache, selection_horizon: int, out_dir: str | Path, gamma: float = 0.99, lr: float = 1e-3, gradient_clip: float = 5.0, **kwargs: Any) -> None:
        self.online_net = online_net
        self.target_net = target_net
        self.cache = cache
        self.selection_horizon = int(selection_horizon)
        self.out_dir = Path(out_dir)
        self.gamma = float(gamma)
        self.gradient_clip = float(gradient_clip)
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.min_replay_size = int(kwargs.get("min_replay_size", self.batch_size))
        self.updates_per_environment_step = int(kwargs.get("updates_per_environment_step", 1))
        self.tau = float(kwargs.get("tau", 0.01))
        self.target_update_interval = int(kwargs.get("target_update_interval", 0))
        self.epsilon_start = float(kwargs.get("epsilon_start", 1.0))
        self.epsilon_end = float(kwargs.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(kwargs.get("epsilon_decay_steps", 1000))
        self.alpha_fit = dict(kwargs.get("alpha_fit") or {"steps": 8, "lr": 0.05, "early_stop_tol": 1e-5, "grad_clip": 5.0})
        self.reward_cfg = NCDMDiagnosticRewardConfig(**dict(kwargs.get("reward_config") or {}))
        self.model_config = dict(kwargs.get("model_config") or {})
        self.seed = int(kwargs.get("seed", 0))
        random.seed(self.seed)
        requested_amp = bool(kwargs.get("use_amp", False))
        self.requested_amp = requested_amp
        self.use_amp = bool(requested_amp and torch.cuda.is_available() and next(self.online_net.parameters()).is_cuda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optim = torch.optim.Adam(self.online_net.parameters(), lr=float(lr))
        self.replay = C3DQNReplayBuffer(int(kwargs.get("replay_capacity", 10000)))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.learning_steps = 0
        self.time_acc = defaultdict(float)
        self.ncdm = kwargs.get("ncdm")
        self.q_matrix = kwargs.get("q_matrix", self.cache.q_matrix)

    def _epsilon(self) -> float:
        frac = min(1.0, self.learning_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _predict_query(self, alpha: torch.Tensor, items: Sequence[int]) -> list[float]:
        if self.ncdm is None or not items:
            return []
        with torch.no_grad():
            p = self.ncdm.predict_with_alpha(alpha, torch.tensor(items, dtype=torch.long, device=self.cache.device), self.q_matrix)
        return [float(x) for x in p.detach().cpu().tolist()]

    def _fit_alpha(self, hist_i: Sequence[int], hist_r: Sequence[float], initial_alpha=None) -> torch.Tensor:
        if self.ncdm is None:
            return torch.zeros((1, self.cache.knowledge_dim), device=self.cache.device)
        t0 = time.perf_counter()
        alpha = fit_student_alpha(self.ncdm, self.q_matrix, hist_i, hist_r, initial_alpha=initial_alpha, device=self.cache.device, **self.alpha_fit)
        self.time_acc["alpha_fit_seconds"] += time.perf_counter() - t0
        return alpha

    def _coverage(self, items: Sequence[int]) -> torch.Tensor:
        if not items:
            return torch.zeros(self.cache.knowledge_dim, device=self.cache.device)
        return self.cache.q_masks[torch.tensor(items, dtype=torch.long, device=self.cache.device)].sum(dim=0)

    def _select(self, hist_i, hist_r, cand, mastery, coverage, policy_step, epsilon=None) -> int:
        if random.random() < (self._epsilon() if epsilon is None else epsilon):
            return int(random.choice(list(cand)))
        t0 = time.perf_counter()
        row = {"history_item_ids": list(hist_i), "history_responses": list(hist_r), "candidate_item_ids": list(cand), "mastery": mastery.detach().cpu().tolist(), "coverage": coverage.detach().cpu().tolist(), "coverage_count": coverage.detach().cpu().tolist(), "policy_step": int(policy_step), "selected_item_id": int(cand[0])}
        batch = pad_c3dqn_batch([row], self.cache, self.selection_horizon)
        with torch.no_grad():
            q, _ = forward_q_network(self.online_net, batch)
        self.time_acc["network_forward_seconds"] += time.perf_counter() - t0
        return int(list(cand)[int(q.argmax(dim=1).item())])

    def update_once(self) -> dict[str, float] | None:
        if len(self.replay) < self.min_replay_size:
            return None
        transitions = self.replay.sample(self.batch_size)
        t0 = time.perf_counter()
        batch, next_batch, rewards, dones, non_terminal_indices = transitions_to_batches(transitions, self.cache, self.selection_horizon)
        self.optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=getattr(self, "use_amp", False)):
            loss, stats = compute_double_dqn_loss(self.online_net, self.target_net, batch, next_batch, rewards, dones, self.gamma, non_terminal_indices)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.scaler.step(self.optim)
        self.scaler.update()
        self.learning_steps += 1
        if self.target_update_interval and self.learning_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.online_net.parameters(), self.target_net.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        self.time_acc["network_update_seconds"] += time.perf_counter() - t0
        stats["td_loss"] = float(loss.item())
        return stats

    def _run_episode(self, sp) -> tuple[list[C3DQNTransition], list[dict[str, float]], list[int], list[float]]:
        cand = list(sp.candidate_items)
        resp = {int(i): float(r) for i, r in zip(sp.support_item_ids, sp.support_responses)}
        hist_i = [int(sp.warm_start_item)]
        hist_r = [float(sp.warm_start_response)]
        alpha = self._fit_alpha(hist_i, hist_r)
        mastery = torch.sigmoid(alpha).squeeze(0)
        coverage_count = self._coverage(hist_i)
        transitions: list[C3DQNTransition] = []
        rewards: list[dict[str, float]] = []
        selected: list[int] = []
        update_stats: list[float] = []
        for t in range(min(self.selection_horizon, len(cand))):
            q_before = nll_score(sp.query_responses, self._predict_query(alpha, sp.query_item_ids))
            item = self._select(hist_i, hist_r, cand, mastery, coverage_count, t)
            before_i, before_r, before_c, before_m, before_cov = list(hist_i), list(hist_r), list(cand), mastery.clone(), coverage_count.clone()
            cand.remove(item)
            hist_i.append(item)
            hist_r.append(resp[item])
            alpha2 = self._fit_alpha(hist_i, hist_r)
            mastery2 = torch.sigmoid(alpha2).squeeze(0)
            coverage2 = self._coverage(hist_i)
            q_after = nll_score(sp.query_responses, self._predict_query(alpha2, sp.query_item_ids))
            t0 = time.perf_counter()
            r = compute_ncdm_diagnostic_reward(q_before, q_after, mastery, mastery2, self.cache.q_masks[item], before_cov, self.reward_cfg)
            self.time_acc["reward_seconds"] += time.perf_counter() - t0
            comps = {"prediction_gain": r.prediction_gain, "diagnosis_gain": r.diagnosis_gain, "coverage_gain": r.coverage_gain, "total": r.total}
            tr = C3DQNTransition(before_i, before_r, before_c, before_m.detach().cpu().tolist(), before_cov.detach().cpu().tolist(), t, item, r.total, comps, list(hist_i), list(hist_r), list(cand), mastery2.detach().cpu().tolist(), coverage2.detach().cpu().tolist(), t + 1, (not cand) or t + 1 >= self.selection_horizon, before_cov.detach().cpu().tolist(), coverage2.detach().cpu().tolist())
            self.replay.push(tr)
            transitions.append(tr)
            rewards.append(comps)
            selected.append(item)
            for _ in range(self.updates_per_environment_step):
                st = self.update_once()
                if st:
                    update_stats.append(st["td_loss"])
            alpha, mastery, coverage_count = alpha2, mastery2, coverage2
        return transitions, rewards, selected, update_stats

    def train(self, sequences_csv: str | Path, *, epochs: int = 1, train_ratio: float = 0.8, max_students: int | None = None, query_ratio: float = 0.2, min_query_items: int = 2) -> list[dict[str, float]]:
        loader = CATOfflineEvaluator({"assets": {}}, debug=False)
        seqs = loader._load_sequences(Path(sequences_csv))
        if max_students:
            seqs = seqs[: int(max_students)]
        students = list(seqs)
        rng = random.Random(self.seed)
        rng.shuffle(students)
        cut = max(1, int(round(len(students) * float(train_ratio))))
        train_seqs, val_seqs = students[:cut], students[cut:] or students[:1]
        history: list[dict[str, float]] = []
        best = float("inf")
        for epoch in range(1, int(epochs) + 1):
            ep_start = time.perf_counter()
            self.time_acc = defaultdict(float)
            rewards: list[dict[str, float]] = []
            selected_all: list[int] = []
            losses: list[float] = []
            for seq in train_seqs:
                sp, _ = make_student_split(seq.student_id, seq.item_ids, seq.responses, seed=self.seed + epoch, valid_count=self.cache.item_count, query_ratio=query_ratio, min_query_items=min_query_items)
                if not sp:
                    continue
                _, r, sel, ls = self._run_episode(sp)
                rewards.extend(r)
                selected_all.extend(sel)
                losses.extend(ls)
            t0 = time.perf_counter()
            val = self.validate(val_seqs, seed=self.seed + epoch, query_ratio=query_ratio, min_query_items=min_query_items)
            self.time_acc["validation_seconds"] += time.perf_counter() - t0
            cnt = Counter(selected_all)
            row = {"epoch": epoch, "mean_total_reward": _mean([r["total"] for r in rewards]), "mean_prediction_reward": _mean([r["prediction_gain"] for r in rewards]), "mean_diagnosis_reward": _mean([r["diagnosis_gain"] for r in rewards]), "mean_coverage_reward": _mean([r["coverage_gain"] for r in rewards]), "td_loss": _mean(losses), "mean_q": 0.0, "target_q_mean": 0.0, "epsilon": self._epsilon(), "replay_size": len(self.replay), "selected_unique_items": len(cnt), "item_exposure_gini": gini(cnt.values()), **val, "feature_build_seconds": self.time_acc.get("feature_build_seconds", 1e-9), "alpha_fit_seconds": self.time_acc.get("alpha_fit_seconds", 0.0), "reward_seconds": self.time_acc.get("reward_seconds", 0.0), "network_forward_seconds": self.time_acc.get("network_forward_seconds", 0.0), "network_update_seconds": self.time_acc.get("network_update_seconds", 0.0), "validation_seconds": self.time_acc.get("validation_seconds", 0.0), "total_epoch_seconds": time.perf_counter() - ep_start}
            history.append(row)
            self._write_history(history)
            if row["validation_query_nll"] <= best:
                best = row["validation_query_nll"]
                self.save_checkpoint(row, epoch)
        return history

    def validate(self, seqs: Sequence[StudentSequence], *, seed: int, query_ratio: float, min_query_items: int) -> dict[str, float]:
        ys: list[float] = []
        ps: list[float] = []
        ents: list[float] = []
        covs: list[float] = []
        for seq in seqs:
            sp, _ = make_student_split(seq.student_id, seq.item_ids, seq.responses, seed=seed, valid_count=self.cache.item_count, query_ratio=query_ratio, min_query_items=min_query_items)
            if not sp:
                continue
            cand = list(sp.candidate_items)
            resp = {int(i): float(r) for i, r in zip(sp.support_item_ids, sp.support_responses)}
            hist_i = [sp.warm_start_item]
            hist_r = [sp.warm_start_response]
            alpha = self._fit_alpha(hist_i, hist_r)
            for t in range(min(self.selection_horizon, len(cand))):
                mastery = torch.sigmoid(alpha).squeeze(0)
                cov = self._coverage(hist_i)
                item = self._select(hist_i, hist_r, cand, mastery, cov, t, epsilon=0.0)
                cand.remove(item)
                hist_i.append(item)
                hist_r.append(resp[item])
                alpha = self._fit_alpha(hist_i, hist_r)
            probs = self._predict_query(alpha, sp.query_item_ids)
            ys.extend(sp.query_responses)
            ps.extend(probs)
            ents.append(float(mastery_entropy(torch.sigmoid(alpha).squeeze(0)).item()))
            covs.append(float((self._coverage(hist_i) > 0).float().mean().item()))
        return {"validation_query_nll": nll_score(ys, ps), "validation_query_auc": auc_score(ys, ps), "validation_query_brier": brier_score(ys, ps), "validation_mastery_entropy": _mean(ents), "validation_concept_coverage": _mean(covs)}

    def save_checkpoint(self, metrics: dict[str, float], epoch: int) -> None:
        builder = build_set_checkpoint_metadata if isinstance(self.online_net, SetConditionedNCDMQNetwork) else build_checkpoint_metadata
        model_cfg = {**self.model_config, "requested_amp": self.requested_amp, "effective_amp": self.use_amp}
        meta = builder(knowledge_dim=self.cache.knowledge_dim, selection_horizon=self.selection_horizon, warm_start_items=1, alpha_fit=self.alpha_fit, reward_config=asdict(self.reward_cfg), model_config=model_cfg, candidate_pool_config={"max_candidates": None, "prefilter_enabled": False, "prefilter_top_k": 256, "prefilter_mode": "diagnostic_heuristic"}, ncdm_item_count=self.cache.ncdm_item_count, q_matrix_item_count=self.cache.q_matrix_item_count, training_seed=self.seed, validation_metrics=metrics, epoch=epoch, strict_item_count_check=self.cache.strict_item_count_check)
        torch.save({"model_state_dict": self.online_net.state_dict(), "metadata": meta}, self.out_dir / "best_checkpoint.pt")

    def _write_history(self, rows: list[dict[str, float]]) -> None:
        with (self.out_dir / "training_history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def run_synthetic_smoke_epoch(self) -> dict[str, float]:
        start = time.perf_counter()
        k = self.cache.knowledge_dim
        for s in range(8):
            cand = list(range(1, min(self.cache.item_count, 6)))
            sel = cand[s % len(cand)]
            self.replay.push(C3DQNTransition([0], [1.0], cand, [0.5] * k, [0.0] * k, 0, sel, 0.1, {"prediction_gain": 0.1, "diagnosis_gain": 0.0, "coverage_gain": 0.0, "total": 0.1}, [0, sel], [1.0, 1.0], [x for x in cand if x != sel], [0.55] * k, [1.0] * k, 1, False, [0.0] * k, [1.0] * k))
        st = self.update_once() or {"td_loss": 0.0, "mean_q": 0.0, "target_q_mean": 0.0}
        metrics = {"epoch": 1, "mean_total_reward": 0.1, "mean_prediction_reward": 0.1, "mean_diagnosis_reward": 0.0, "mean_coverage_reward": 0.0, "td_loss": st["td_loss"], "mean_q": st.get("mean_q", 0.0), "target_q_mean": st.get("target_q_mean", 0.0), "epsilon": 0.0, "replay_size": len(self.replay), "selected_unique_items": 5, "item_exposure_gini": 0.0, "validation_query_nll": 0.0, "validation_query_auc": 0.5, "validation_query_brier": 0.25, "validation_mastery_entropy": 1.0, "validation_concept_coverage": 0.0, "feature_build_seconds": 1e-9, "alpha_fit_seconds": 0.0, "reward_seconds": 0.0, "network_forward_seconds": 0.0, "network_update_seconds": self.time_acc.get("network_update_seconds", 0.0), "validation_seconds": 0.0, "total_epoch_seconds": time.perf_counter() - start}
        self._write_history([metrics])
        self.save_checkpoint(metrics, 1)
        return metrics


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")
