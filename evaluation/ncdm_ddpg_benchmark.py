"""Unified paired benchmark support for NCDM-DDPG."""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from evaluation.benchmark import BenchmarkV2Evaluator
from evaluation.offline_eval import (
    CATOfflineEvaluator,
    MissingAssetsError,
    StudentSequence,
)
from evaluation.policies import NCDMDDPGPolicy
from models.ncdm import OfficialNCDM, safe_load_ncdm_checkpoint


def _load_sequences_limited(
    legacy: CATOfflineEvaluator,
    path: Path,
    limit: int,
) -> list[StudentSequence]:
    """Stream at most ``limit`` students instead of materializing the full CSV."""
    max_sequences = max(1, int(limit))
    sequences: list[StudentSequence] = []
    grouped: "OrderedDict[str, tuple[list[int], list[float]]]" = OrderedDict()

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            sid = str(row.get("student_id") or row.get("user_id") or index)
            item_cell = (
                row.get("item_ids")
                or row.get("exer_ids")
                or row.get("questions")
                or row.get("question_ids")
            )
            response_cell = (
                row.get("responses")
                or row.get("answers")
                or row.get("correct")
            )

            if item_cell and response_cell:
                items = legacy._parse_list_cell(item_cell)
                responses = [
                    float(value)
                    for value in legacy._parse_list_cell(response_cell)
                ]
                if items and len(items) == len(responses):
                    sequences.append(StudentSequence(sid, items, responses))
                    if len(sequences) >= max_sequences:
                        break
                continue

            if {"item_id", "response"}.issubset(row):
                if sid not in grouped and len(grouped) >= max_sequences:
                    break
                grouped.setdefault(sid, ([], []))
                grouped[sid][0].append(int(float(row["item_id"])))
                grouped[sid][1].append(float(row["response"]))

    if not sequences:
        for sid, (items, responses) in grouped.items():
            if items and len(items) == len(responses):
                sequences.append(StudentSequence(sid, items, responses))
                if len(sequences) >= max_sequences:
                    break
    return sequences


class NCDMDDPGBenchmarkEvaluator(BenchmarkV2Evaluator):
    """BenchmarkV2 with bounded-memory real-data loading for NCDM-DDPG."""

    def _load_or_synthetic(self):
        if self.debug:
            return super()._load_or_synthetic()

        legacy = CATOfflineEvaluator(
            self.config,
            debug=False,
            ddpg_checkpoint=str(self.ddpg_checkpoint),
        )
        paths = legacy.paths
        required_keys = ["q_matrix", "ncdm_checkpoint", "test_sequences"]
        if "NCDM-DDPG" in self.policy_names:
            required_keys.append("item_bank")

        missing: list[Path] = []
        for key in required_keys:
            path = paths.get(key)
            if path is None:
                missing.append(Path(f"assets.{key}"))
            elif not path.exists():
                missing.append(path)
        if missing:
            raise MissingAssetsError(missing)

        print("loading Q matrix...", flush=True)
        q = legacy._load_array(paths["q_matrix"]).astype(np.float32)

        print(
            f"streaming at most {self.max_students} test students...",
            flush=True,
        )
        sequences = _load_sequences_limited(
            legacy,
            paths["test_sequences"],
            self.max_students,
        )
        if not sequences:
            raise RuntimeError(
                f"No valid student sequences found in {paths['test_sequences']}"
            )
        print(f"loaded {len(sequences)} test students", flush=True)

        print("loading frozen NCDM...", flush=True)
        ncdm = OfficialNCDM(1, q.shape[0], q.shape[1]).to(self.device)
        safe_load_ncdm_checkpoint(
            ncdm,
            paths["ncdm_checkpoint"],
            self.device,
        )
        ncdm.eval()
        for parameter in ncdm.parameters():
            parameter.requires_grad = False

        if "NCDM-DDPG" in self.policy_names:
            print("loading semantic item bank...", flush=True)
            item_bank = legacy._load_array(paths["item_bank"]).astype(np.float32)
        else:
            item_bank = np.zeros((0, 0), dtype=np.float32)

        common_items = min(
            int(q.shape[0]),
            int(ncdm.k_difficulty.num_embeddings),
            int(ncdm.e_discrimination.num_embeddings),
        )
        if "NCDM-DDPG" in self.policy_names:
            if item_bank.ndim != 2 or item_bank.shape[1] <= 0:
                raise ValueError("NCDM-DDPG requires a rank-2 semantic item bank")
            common_items = min(common_items, int(item_bank.shape[0]))
            # The base benchmark includes the semantic bank in valid_item_count
            # for non-native tracks. This gives Random and NCDM-DDPG the same
            # 0..common_items-1 universe without changing the full Q/NCDM shapes.
            self.track = "benchmark_v2"
            print(
                f"shared NCDM-DDPG item universe: {common_items}",
                flush=True,
            )
        if common_items <= 0:
            raise ValueError("NCDM-DDPG assets have no common valid item IDs")

        self.mirt_model = None
        return q, item_bank, sequences, ncdm, False

    def _policies(self, q, item_bank, ncdm, synthetic, mirt=None):
        requested_names = list(self.policy_names)
        base_names = [name for name in requested_names if name != "NCDM-DDPG"]
        self.policy_names = base_names
        try:
            base_policies = super()._policies(
                q,
                item_bank,
                ncdm,
                synthetic,
                mirt=mirt,
            )
        finally:
            self.policy_names = requested_names

        ddpg_policy = None
        if "NCDM-DDPG" in requested_names:
            ddpg_policy = NCDMDDPGPolicy(
                self.ddpg_checkpoint,
                q_matrix=torch.as_tensor(q, dtype=torch.float32, device=self.device),
                item_bank=torch.as_tensor(
                    item_bank,
                    dtype=torch.float32,
                    device=self.device,
                ),
                ncdm=ncdm,
                device=self.device,
                allow_debug_fallback=self.debug or synthetic,
            )

        by_name = {policy.name: policy for policy in base_policies}
        ordered = []
        for name in requested_names:
            if name == "NCDM-DDPG":
                if ddpg_policy is None:
                    raise RuntimeError("failed to construct NCDM-DDPG")
                ordered.append(ddpg_policy)
            else:
                policy = by_name.get(name)
                if policy is None:
                    raise RuntimeError(
                        f"base benchmark did not construct requested policy: {name}"
                    )
                ordered.append(policy)
        return ordered
