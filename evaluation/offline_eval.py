"""Offline CAT/RL evaluation utilities for KCQRL-RL.

This module intentionally avoids the old text representation-learning path. It does
not require tokenizers, transformers, json_file_dataset, KC cluster files, or
question/KC map JSON files. The evaluator consumes CAT/RL assets configured in
``configs/default.yaml`` and compares item-selection policies on held-out student
response sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime for friendly errors
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class StudentSequence:
    student_id: str
    item_ids: List[int]
    responses: List[float]


@dataclass
class EvaluationResult:
    policy: str
    students: int
    interactions: int
    accuracy: float
    auc: float
    nll: float
    brier: float
    reward: float


class MissingAssetsError(FileNotFoundError):
    """Raised when configured Google Drive assets are not available."""

    def __init__(self, missing_paths: Sequence[Path]):
        self.missing_paths = [Path(p) for p in missing_paths]
        message = "Missing external assets:\n" + "\n".join(f"  - {p}" for p in self.missing_paths)
        super().__init__(message)


class CATOfflineEvaluator:
    """Minimal offline evaluator for CAT/RL item-selection policies.

    The evaluator replays test response sequences. At each step a policy selects an
    item from candidates with known held-out responses for that student, receives
    the logged response, updates a simple ability estimate, and accrues reward for
    correct prediction of that response.
    """

    REQUIRED_ASSETS = ("q_matrix", "item_bank", "test_sequences")
    OPTIONAL_POLICY_ASSETS = {"DDPG": ("ncdm_checkpoint",), "MIRT-MFI": ("mirt_checkpoint",), "MIRT-KLI": ("mirt_checkpoint",)}

    def __init__(self, config: Mapping, debug: bool = False, ddpg_checkpoint: Optional[str] = None):
        self.config = config
        self.debug = debug
        self.seed = int(config.get("seed", 42))
        self.rng = random.Random(self.seed)
        eval_cfg = config.get("evaluation", {}) or {}
        self.horizon = int(eval_cfg.get("horizon", 20))
        self.max_students = int(eval_cfg.get("max_students", 50 if debug else 1000))
        self.candidate_size = int(eval_cfg.get("candidate_size", 50))
        self.policies = list(eval_cfg.get("policies", ["Random", "MIRT-MFI", "MIRT-KLI", "DDPG", "OneStepOracle"]))
        self.paths = self._resolve_asset_paths(config.get("assets", {}) or {})
        self.ddpg_checkpoint = Path(ddpg_checkpoint or "outputs/ddpg_actor.pt").expanduser()
        self.ddpg_checkpoint = self.ddpg_checkpoint if self.ddpg_checkpoint.is_absolute() else Path.cwd() / self.ddpg_checkpoint
        self.ddpg_actor = None
        self.ddpg_ncdm = None
        self.ddpg_device = torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu") if torch is not None else None
        self._warned_ddpg_fallback = False

    @staticmethod
    def _resolve_asset_paths(asset_cfg: Mapping[str, str]) -> Dict[str, Path]:
        base = Path(asset_cfg.get("base_dir", ".")).expanduser()
        paths: Dict[str, Path] = {}
        for key, value in asset_cfg.items():
            if key == "base_dir" or value is None:
                continue
            p = Path(str(value)).expanduser()
            paths[key] = p if p.is_absolute() else base / p
        return paths

    def missing_required_assets(self) -> List[Path]:
        needed = set(self.REQUIRED_ASSETS)
        for policy in self.policies:
            needed.update(self.OPTIONAL_POLICY_ASSETS.get(policy, ()))
        return [self.paths[name] for name in sorted(needed) if name in self.paths and not self.paths[name].exists()]

    def ensure_assets(self) -> None:
        missing = self.missing_required_assets()
        if missing:
            raise MissingAssetsError(missing)
        if np is None:
            raise RuntimeError("numpy is required to run offline evaluation. Install requirements.txt first.")

    def load(self) -> Tuple["np.ndarray", "np.ndarray", List[StudentSequence]]:
        self.ensure_assets()
        q_matrix = self._load_array(self.paths["q_matrix"]).astype(np.float32)
        item_bank = self._load_array(self.paths["item_bank"]).astype(np.float32)
        sequences = self._load_sequences(self.paths["test_sequences"])
        self._prepare_ddpg(q_matrix, item_bank)
        return q_matrix, item_bank, sequences[: self.max_students]

    @staticmethod
    def _load_array(path: Path):
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".pt":
            if torch is None:
                raise RuntimeError(f"PyTorch is required to load {path}")
            obj = torch.load(path, map_location="cpu")
            if hasattr(obj, "detach"):
                obj = obj.detach().cpu().numpy()
            return np.asarray(obj)
        raise ValueError(f"Unsupported asset type for {path}")

    @staticmethod
    def _parse_list_cell(value: str) -> List[int]:
        value = value.strip()
        if not value:
            return []
        for ch in "[]()":
            value = value.replace(ch, " ")
        return [int(float(x)) for x in value.replace(";", ",").split(",") if x.strip()]


    def _prepare_ddpg(self, q_matrix, item_bank) -> None:
        if "DDPG" not in self.policies or torch is None:
            return
        if not self.ddpg_checkpoint.exists():
            print("DDPG actor checkpoint not found; falling back to heuristic policy.")
            self._warned_ddpg_fallback = True
            return
        from models.actor import LSTMActor
        from models.ncdm import OfficialNCDM, safe_load_ncdm_checkpoint

        q_tensor = torch.tensor(q_matrix, dtype=torch.float32, device=self.ddpg_device)
        ncdm = OfficialNCDM(1, q_matrix.shape[0], q_matrix.shape[1]).to(self.ddpg_device)
        safe_load_ncdm_checkpoint(ncdm, self.paths["ncdm_checkpoint"], self.ddpg_device)
        ncdm.eval()
        for param in ncdm.parameters():
            param.requires_grad = False
        actor = LSTMActor(semantic_dim=item_bank.shape[1], q_dim=q_matrix.shape[1]).to(self.ddpg_device)
        state = torch.load(self.ddpg_checkpoint, map_location=self.ddpg_device)
        actor.load_state_dict(state)
        actor.eval()
        self.ddpg_actor = actor
        self.ddpg_ncdm = ncdm
        self.ddpg_q_tensor = q_tensor
        self.ddpg_item_bank = torch.tensor(item_bank, dtype=torch.float32, device=self.ddpg_device)

    def _load_sequences(self, path: Path) -> List[StudentSequence]:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        sequences: List[StudentSequence] = []
        for idx, row in enumerate(rows):
            sid = row.get("student_id") or row.get("user_id") or str(idx)
            item_cell = row.get("item_ids") or row.get("exer_ids") or row.get("questions") or row.get("question_ids")
            resp_cell = row.get("responses") or row.get("answers") or row.get("correct")
            if item_cell and resp_cell:
                items = self._parse_list_cell(item_cell)
                responses = [float(x) for x in self._parse_list_cell(resp_cell)]
            elif {"item_id", "response"}.issubset(row):
                items = [int(float(row["item_id"]))]
                responses = [float(row["response"])]
            else:
                continue
            if items and len(items) == len(responses):
                sequences.append(StudentSequence(sid, items, responses))
        return sequences

    def evaluate(self) -> List[EvaluationResult]:
        q_matrix, item_bank, sequences = self.load()
        if not sequences:
            raise RuntimeError(f"No valid student sequences found in {self.paths['test_sequences']}")
        return [self._evaluate_policy(policy, q_matrix, item_bank, sequences) for policy in self.policies]

    def _evaluate_policy(self, policy: str, q_matrix, item_bank, sequences: Sequence[StudentSequence]) -> EvaluationResult:
        correct = interactions = 0
        reward_sum = 0.0
        y_true: List[float] = []
        y_score: List[float] = []
        for seq in sequences:
            ability = np.zeros(q_matrix.shape[1], dtype=float)
            seen = set()
            ddpg_hx = ddpg_cx = None
            last_item = last_response = None
            for _ in range(min(self.horizon, len(seq.item_ids))):
                candidates = [i for i in seq.item_ids if i not in seen and i < len(q_matrix)]
                if not candidates:
                    break
                if len(candidates) > self.candidate_size:
                    candidates = candidates[: self.candidate_size]
                item, ddpg_hx, ddpg_cx = self._select(policy, candidates, ability, q_matrix, item_bank, seq, ddpg_hx, ddpg_cx, last_item, last_response)
                seen.add(item)
                response = seq.responses[seq.item_ids.index(item)]
                pred = self._predict(ability, q_matrix[item])
                y_true.append(float(response >= 0.5))
                y_score.append(pred)
                correct += int((pred >= 0.5) == (response >= 0.5))
                reward_sum += 1.0 if (pred >= 0.5) == (response >= 0.5) else 0.0
                ability += (response - pred) * q_matrix[item]
                last_item, last_response = item, response
                interactions += 1
        acc = correct / interactions if interactions else 0.0
        auc = self._auc_score(y_true, y_score) if interactions else float("nan")
        nll = self._binary_cross_entropy(y_true, y_score) if interactions else 0.0
        brier = self._brier_score(y_true, y_score) if interactions else 0.0
        reward = reward_sum / interactions if interactions else 0.0
        return EvaluationResult(policy, len(sequences), interactions, acc, auc, nll, brier, reward)

    def _select(self, policy: str, candidates: Sequence[int], ability, q_matrix, item_bank, seq: StudentSequence, ddpg_hx=None, ddpg_cx=None, last_item=None, last_response=None):
        if policy == "Random":
            return self.rng.choice(list(candidates)), ddpg_hx, ddpg_cx
        if policy == "OneStepOracle":
            return max(candidates, key=lambda item: abs(seq.responses[seq.item_ids.index(item)] - self._predict(ability, q_matrix[item]))), ddpg_hx, ddpg_cx
        probs = [(item, self._predict(ability, q_matrix[item])) for item in candidates]
        if policy == "MIRT-MFI":
            return max(probs, key=lambda x: x[1] * (1.0 - x[1]))[0], ddpg_hx, ddpg_cx
        if policy == "MIRT-KLI":
            return max(probs, key=lambda x: abs(x[1] - 0.5))[0], ddpg_hx, ddpg_cx
        if policy == "DDPG":
            if self.ddpg_actor is not None:
                with torch.no_grad():
                    if ddpg_hx is None or ddpg_cx is None:
                        ddpg_hx, ddpg_cx = self.ddpg_actor.init_hidden(1, self.ddpg_device)
                    if last_item is None:
                        sem = torch.zeros((1, self.ddpg_item_bank.shape[1]), dtype=torch.float32, device=self.ddpg_device)
                        q = torch.zeros((1, self.ddpg_q_tensor.shape[1]), dtype=torch.float32, device=self.ddpg_device)
                        diff = torch.zeros_like(q)
                        disc = torch.zeros((1, 1), dtype=torch.float32, device=self.ddpg_device)
                        resp = torch.zeros(1, dtype=torch.float32, device=self.ddpg_device)
                    else:
                        tid = torch.tensor([last_item], dtype=torch.long, device=self.ddpg_device)
                        sem = self.ddpg_item_bank[last_item].unsqueeze(0)
                        q = self.ddpg_q_tensor[last_item].unsqueeze(0)
                        diff = torch.sigmoid(self.ddpg_ncdm.k_difficulty(tid))
                        disc = torch.sigmoid(self.ddpg_ncdm.e_discrimination(tid))
                        resp = torch.tensor([float(last_response)], dtype=torch.float32, device=self.ddpg_device)
                    ideal, ddpg_hx, ddpg_cx = self.ddpg_actor(sem, q, diff, disc, resp, ddpg_hx, ddpg_cx)
                    cand_ids = torch.tensor(list(candidates), dtype=torch.long, device=self.ddpg_device)
                    cand_vecs = torch.cat([self.ddpg_q_tensor[cand_ids], torch.sigmoid(self.ddpg_ncdm.k_difficulty(cand_ids)), torch.sigmoid(self.ddpg_ncdm.e_discrimination(cand_ids))], dim=-1)
                    loc = int(torch.argmin(torch.cdist(ideal, cand_vecs).squeeze(0)).item())
                    return int(candidates[loc]), ddpg_hx.detach(), ddpg_cx.detach()
            target = ability[: item_bank.shape[1]] if item_bank.ndim == 2 else ability
            return max(candidates, key=lambda item: float(np.dot(item_bank[item][: len(target)], target))), ddpg_hx, ddpg_cx
        return self.rng.choice(list(candidates)), ddpg_hx, ddpg_cx

    @staticmethod
    def _predict(ability, q_vec) -> float:
        denom = max(float(np.linalg.norm(q_vec)), 1.0)
        z = float(np.dot(ability, q_vec) / denom)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

    @staticmethod
    def _auc_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
        """Compute binary ROC AUC with average ranks for tied scores.

        This rank-based implementation is equivalent to the Mann-Whitney U
        statistic. It returns NaN when AUC is undefined because only one class is
        present in ``y_true``.
        """
        positives = sum(1 for y in y_true if y >= 0.5)
        negatives = len(y_true) - positives
        if positives == 0 or negatives == 0:
            return float("nan")

        ranked = sorted(zip(y_score, y_true), key=lambda pair: pair[0])
        pos_rank_sum = 0.0
        i = 0
        while i < len(ranked):
            j = i + 1
            while j < len(ranked) and ranked[j][0] == ranked[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            pos_rank_sum += avg_rank * sum(1 for _, y in ranked[i:j] if y >= 0.5)
            i = j

        return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    @staticmethod
    def _binary_cross_entropy(y_true: Sequence[float], y_score: Sequence[float]) -> float:
        eps = 1e-7
        total = 0.0
        for y, p in zip(y_true, y_score):
            clipped = min(max(float(p), eps), 1.0 - eps)
            total += -(float(y) * math.log(clipped) + (1.0 - float(y)) * math.log(1.0 - clipped))
        return total / len(y_true) if y_true else 0.0

    @staticmethod
    def _brier_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
        eps = 1e-7
        total = 0.0
        for y, p in zip(y_true, y_score):
            clipped = min(max(float(p), eps), 1.0 - eps)
            total += (clipped - float(y)) ** 2
        return total / len(y_true) if y_true else 0.0
