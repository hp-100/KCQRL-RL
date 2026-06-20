"""NCDM-native C3DQN evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import random, torch
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork, SetConditionedNCDMQNetwork
from agents.ncdm_c3dqn_trainer import load_c3dqn_checkpoint
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter

class RandomNCDMPolicy(BaseCATPolicy):
    name = "Random-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="random", evaluator_model="NCDM")
    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context); self.rng=random.Random(f"{student_id}:{seed}")
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any]) -> int:
        return int(self.rng.choice(list(candidate_item_ids)))

class C3DQNNCDMPolicy(BaseCATPolicy):
    name = "C3DQN-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="candidate_conditioned_attention_dueling_double_dqn", evaluator_model="NCDM", uses_query_labels=False, uses_privileged_information=False)
    def __init__(self, checkpoint_path, ncdm, q_matrix: torch.Tensor, device: str | torch.device = "cpu", expected_protocol_config: dict[str, Any] | None = None, network: CandidateConditionedNCDMQNetwork | None = None, cache: NCDMItemFeatureCache | None = None, selection_horizon: int | None = None, alpha_fit: dict | None = None) -> None:
        self.device = torch.device(device)
        self.ncdm = ncdm.to(self.device).eval() if hasattr(ncdm, "to") else ncdm
        self.q_matrix = q_matrix.to(self.device).float()
        if network is None:
            self.network, meta = load_c3dqn_checkpoint(checkpoint_path, ncdm=self.ncdm, q_matrix=self.q_matrix, device=self.device, expected_protocol_config=expected_protocol_config)
            self.selection_horizon = int(meta["selection_horizon"])
            self.alpha_fit = dict(meta.get("alpha_fit") or {})
            self.candidate_pool_config = dict(meta.get("candidate_pool_config") or {})
        else:
            self.network = network.to(self.device).eval()
            self.selection_horizon = int(selection_horizon or (expected_protocol_config or {}).get("selection_horizon", 5))
            self.alpha_fit = dict(alpha_fit or {"initial_steps": 8, "incremental_steps": 3, "lr": 0.05, "early_stop_tol": 1e-5})
            self.candidate_pool_config = dict((expected_protocol_config or {}).get("candidate_pool_config") or {})
        self.cache = cache or NCDMItemFeatureCache(self.ncdm, self.q_matrix, self.device)
        self.prefilter = NCDMCandidatePrefilter(q_matrix=self.q_matrix, feature_cache=self.cache, ncdm=self.ncdm, config=self.candidate_pool_config)
        self._alpha = None
        self._history_prefix = ([], [])
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any] | None = None) -> int:
        if not candidate_item_ids: raise ValueError("C3DQN-NCDM requires at least one candidate item")
        forbidden = set((context or {}).keys()) & {"query_item_ids","query_responses","future_responses","query_labels"}
        if forbidden: raise ValueError(f"C3DQN-NCDM policy received privileged context keys: {sorted(forbidden)}")
        with torch.enable_grad():
            alpha = fit_student_alpha(
                self.ncdm,
                self.q_matrix,
                history_item_ids,
                history_responses,
                initial_alpha=self._alpha if list(history_item_ids[:-1]) == self._history_prefix[0] and list(history_responses[:-1]) == self._history_prefix[1] else None,
                steps=int(self.alpha_fit.get("initial_steps", self.alpha_fit.get("steps", 8)) if self._alpha is None else self.alpha_fit.get("incremental_steps", self.alpha_fit.get("steps", 3))),
                lr=float(self.alpha_fit.get("lr", 0.05)),
                early_stop_tol=float(self.alpha_fit.get("early_stop_tol", 1e-5)),
                grad_clip=self.alpha_fit.get("grad_clip", None),
                device=self.device,
            )
        with torch.no_grad():
            self._alpha = alpha.detach(); self._history_prefix = (list(history_item_ids), list(history_responses))
            mastery = torch.sigmoid(alpha).squeeze(0)
            if history_item_ids:
                coverage_count = self.cache.q_masks[torch.as_tensor(history_item_ids, dtype=torch.long, device=self.device)].sum(dim=0)
            else:
                coverage_count = torch.zeros_like(mastery)
            filtered, summary = self.prefilter.select(candidate_item_ids, alpha, mastery, coverage_count)
            coverage = (coverage_count / float(self.selection_horizon)).clamp(0,1)
            hist = self.cache.history(history_item_ids, history_responses, self.selection_horizon).unsqueeze(0)
            if hist.shape[1] == 0:
                raise ValueError("C3DQN-NCDM requires warm-start history before selection")
            hmask = torch.ones((1,hist.shape[1]), dtype=torch.bool, device=self.device)
            cand = self.cache.candidate(filtered).unsqueeze(0)
            cmask = torch.ones((1,len(filtered)), dtype=torch.bool, device=self.device)
            policy_step = int((context or {}).get("policy_step", len(history_item_ids)))
            glob = build_global_feature(mastery, coverage, policy_step, self.selection_horizon).unsqueeze(0)
            q_values,_ = self.network(hist,hmask,cand,cmask,glob)
            return int(list(filtered)[int(q_values.argmax(dim=1).item())])


class SetC3DQNNCDMPolicy(C3DQNNCDMPolicy):
    name = "Set-C3DQN-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="set_c3dqn", evaluator_model="NCDM", uses_query_labels=False, uses_privileged_information=False)
