"""NCDM-native C3DQN evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import random, torch
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork

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
    def __init__(self, network: CandidateConditionedNCDMQNetwork, cache: NCDMItemFeatureCache, ncdm, q_matrix: torch.Tensor, selection_horizon: int, alpha_fit: dict | None = None, device: str | torch.device = "cpu") -> None:
        self.network=network.to(device).eval(); self.cache=cache; self.ncdm=ncdm; self.q_matrix=q_matrix.to(device); self.selection_horizon=int(selection_horizon); self.alpha_fit=dict(alpha_fit or {"steps":8,"lr":0.05}); self.device=torch.device(device)
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any] | None = None) -> int:
        if not candidate_item_ids: raise ValueError("C3DQN-NCDM requires at least one candidate item")
        forbidden = set((context or {}).keys()) & {"query_item_ids","query_responses","future_responses","query_labels"}
        if forbidden: raise ValueError(f"C3DQN-NCDM policy received privileged context keys: {sorted(forbidden)}")
        with torch.no_grad():
            alpha = fit_student_alpha(self.ncdm, self.q_matrix, history_item_ids, history_responses, steps=int(self.alpha_fit.get("steps",8)), lr=float(self.alpha_fit.get("lr",0.05)), device=self.device)
            mastery = torch.sigmoid(alpha).squeeze(0)
            if history_item_ids:
                coverage_count = self.cache.q_masks[torch.as_tensor(history_item_ids, dtype=torch.long, device=self.device)].sum(dim=0)
            else:
                coverage_count = torch.zeros_like(mastery)
            coverage = (coverage_count / float(self.selection_horizon)).clamp(0,1)
            hist = self.cache.history(history_item_ids, history_responses, self.selection_horizon).unsqueeze(0)
            if hist.shape[1] == 0:
                raise ValueError("C3DQN-NCDM requires warm-start history before selection")
            hmask = torch.ones((1,hist.shape[1]), dtype=torch.bool, device=self.device)
            cand = self.cache.candidate(candidate_item_ids).unsqueeze(0)
            cmask = torch.ones((1,len(candidate_item_ids)), dtype=torch.bool, device=self.device)
            glob = build_global_feature(mastery, coverage, len(history_item_ids), self.selection_horizon).unsqueeze(0)
            q_values,_ = self.network(hist,hmask,cand,cmask,glob)
            return int(list(candidate_item_ids)[int(q_values.argmax(dim=1).item())])
