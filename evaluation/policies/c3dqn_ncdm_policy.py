"""NCDM-native C3DQN evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import random, torch
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from agents.ncdm_c3dqn_trainer import load_c3dqn_checkpoint

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
        else:
            self.network = network.to(self.device).eval()
            self.selection_horizon = int(selection_horizon or (expected_protocol_config or {}).get("selection_horizon", 5))
            self.alpha_fit = dict(alpha_fit or {"steps": 8, "lr": 0.05, "early_stop_tol": 1e-5})
        self.cache = cache or NCDMItemFeatureCache(self.ncdm, self.q_matrix, self.device)
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
                steps=int(self.alpha_fit.get("steps", 8)),
                lr=float(self.alpha_fit.get("lr", 0.05)),
                early_stop_tol=float(self.alpha_fit.get("early_stop_tol", 1e-5)),
                grad_clip=self.alpha_fit.get("grad_clip", None),
                device=self.device,
            )
        with torch.no_grad():
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
            policy_step = int((context or {}).get("policy_step", len(history_item_ids)))
            glob = build_global_feature(mastery, coverage, policy_step, self.selection_horizon).unsqueeze(0)
            q_values,_ = self.network(hist,hmask,cand,cmask,glob)
            return int(list(candidate_item_ids)[int(q_values.argmax(dim=1).item())])

class PrivilegeGuard:
    FORBIDDEN={"query_item_ids","query_responses","query_labels","future_responses","candidate_response_lookup","query_loss","future_item_ids"}
    @classmethod
    def check(cls, context: dict[str, Any] | None, policy_name: str):
        bad=set((context or {}).keys()) & cls.FORBIDDEN
        if bad: raise ValueError(f"{policy_name} policy received privileged context keys: {sorted(bad)}")

class SetC3DQNNCDMPolicy(C3DQNNCDMPolicy):
    name = "Set-C3DQN-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="set_conditioned_candidate_attention_dueling_double_dqn", evaluator_model="NCDM", uses_query_labels=False, uses_privileged_information=False)
    def __init__(self, checkpoint_path, ncdm, q_matrix: torch.Tensor, device: str | torch.device = "cpu", expected_protocol_config: dict[str, Any] | None = None, network=None, cache: NCDMItemFeatureCache | None = None, selection_horizon: int | None = None, alpha_fit: dict | None = None):
        from agents.ncdm_c3dqn_trainer import load_set_c3dqn_checkpoint
        self.device=torch.device(device); self.ncdm=ncdm.to(self.device).eval() if hasattr(ncdm,"to") else ncdm; self.q_matrix=q_matrix.to(self.device).float()
        if network is None:
            self.network, meta=load_set_c3dqn_checkpoint(checkpoint_path,ncdm=self.ncdm,q_matrix=self.q_matrix,device=self.device,expected_protocol_config=expected_protocol_config)
            self.selection_horizon=int(meta["selection_horizon"]); self.alpha_fit=dict(meta.get("alpha_fit") or {})
        else:
            self.network=network.to(self.device).eval(); self.selection_horizon=int(selection_horizon or (expected_protocol_config or {}).get("selection_horizon",5)); self.alpha_fit=dict(alpha_fit or {"steps":8,"lr":0.05,"early_stop_tol":1e-5})
        self.cache=cache or NCDMItemFeatureCache(self.ncdm,self.q_matrix,self.device)
        self._cached_items=[]; self._cached_responses=[]; self._cached_alpha=None
    def reset(self, student_id=None, seed=None, context=None):
        super().reset(student_id, seed, context); self._cached_items=[]; self._cached_responses=[]; self._cached_alpha=None
    def select(self, candidate_item_ids, history_item_ids, history_responses, context=None):
        PrivilegeGuard.check(context, self.name)
        if not candidate_item_ids: raise ValueError("Set-C3DQN-NCDM requires at least one candidate item")
        # Use base implementation; base forward now routes coverage_count through kwargs only for Set network via helper would be ideal.
        with torch.enable_grad():
            if self._cached_alpha is not None and list(history_item_ids[:-1])==self._cached_items and list(history_responses[:-1])==self._cached_responses:
                init=self._cached_alpha; steps=int(self.alpha_fit.get("incremental_steps", self.alpha_fit.get("steps",3)))
            else:
                init=None; steps=int(self.alpha_fit.get("initial_steps", self.alpha_fit.get("steps",8)))
            alpha=fit_student_alpha(self.ncdm,self.q_matrix,history_item_ids,history_responses,initial_alpha=init,steps=steps,lr=float(self.alpha_fit.get("lr",0.05)),early_stop_tol=float(self.alpha_fit.get("early_stop_tol",1e-5)),grad_clip=self.alpha_fit.get("grad_clip",None),device=self.device)
            self._cached_alpha=alpha.detach(); self._cached_items=list(history_item_ids); self._cached_responses=list(history_responses)
        with torch.no_grad():
            mastery=torch.sigmoid(alpha).squeeze(0); coverage_count=self.cache.q_masks[torch.as_tensor(history_item_ids,dtype=torch.long,device=self.device)].sum(0) if history_item_ids else torch.zeros_like(mastery)
            hist=self.cache.history(history_item_ids,history_responses,self.selection_horizon).unsqueeze(0); hmask=torch.ones((1,hist.shape[1]),dtype=torch.bool,device=self.device)
            cand=self.cache.candidate(candidate_item_ids).unsqueeze(0); cmask=torch.ones((1,len(candidate_item_ids)),dtype=torch.bool,device=self.device)
            policy_step=int((context or {}).get("policy_step",len(history_item_ids))); glob=build_global_feature(mastery,coverage_count,policy_step,self.selection_horizon).unsqueeze(0)
            q,_=self.network(hist,hmask,cand,cmask,glob,coverage_count=coverage_count.unsqueeze(0)); return int(list(candidate_item_ids)[int(q.argmax(1).item())])
