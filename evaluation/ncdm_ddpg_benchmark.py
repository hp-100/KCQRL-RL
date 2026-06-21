"""Unified paired benchmark support for NCDM-DDPG."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from evaluation.benchmark import BenchmarkV2Evaluator
from evaluation.offline_eval import CATOfflineEvaluator, MissingAssetsError
from evaluation.policies import NCDMDDPGPolicy


class NCDMDDPGBenchmarkEvaluator(BenchmarkV2Evaluator):
    """BenchmarkV2 with semantic-bank loading for NCDM-DDPG."""

    def _load_or_synthetic(self):
        q, item_bank, sequences, ncdm, synthetic = super()._load_or_synthetic()
        if "NCDM-DDPG" in self.policy_names and item_bank.size == 0:
            legacy = CATOfflineEvaluator(
                self.config,
                debug=self.debug,
                ddpg_checkpoint=str(self.ddpg_checkpoint),
            )
            item_bank_path = legacy.paths.get("item_bank")
            if item_bank_path is None:
                raise MissingAssetsError([Path("item_bank")])
            if not item_bank_path.exists():
                raise MissingAssetsError([item_bank_path])
            item_bank = legacy._load_array(item_bank_path).astype(np.float32)

        if "NCDM-DDPG" in self.policy_names:
            if item_bank.ndim != 2 or item_bank.shape[1] <= 0:
                raise ValueError("NCDM-DDPG requires a rank-2 semantic item bank")
            common_items = min(
                int(q.shape[0]),
                int(item_bank.shape[0]),
                int(ncdm.k_difficulty.num_embeddings),
                int(ncdm.e_discrimination.num_embeddings),
            )
            if common_items <= 0:
                raise ValueError("NCDM-DDPG assets have no common valid item IDs")
        return q, item_bank, sequences, ncdm, synthetic

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
