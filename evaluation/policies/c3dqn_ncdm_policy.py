"""NCDM-native C3DQN evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import random, torch
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork, SetConditionedNCDMQNetwork
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from agents.ncdm_c3dqn_trainer import load_c3dqn_checkpoint, load_set_c3dqn_checkpoint, forward_q_network

class RandomNCDMPolicy(BaseCATPolicy):
    name = "Random-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="random", evaluator_model="NCDM")
    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context); self.rng=random.Random(f"{student_id}:{seed}")
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any]) -> int:
        return int(self.rng.choice(list(candidate_item_ids)))

class PrivilegeGuardMixin:
    def _validate_context(self, context):
        forbidden={"query_item_ids","query_responses","future_responses","query_labels","candidate_response_lookup","query_loss"}
        leaked=forbidden & set((context or {}).keys())
        if leaked: raise ValueError(f"{self.name} policy received privileged context keys: {sorted(leaked)}")

class C3DQNNCDMPolicy(PrivilegeGuardMixin, BaseCATPolicy):
    name = "C3DQN-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="candidate_conditioned_attention_dueling_double_dqn", evaluator_model="NCDM", uses_query_labels=False, uses_privileged_information=False)
    loader = staticmethod(load_c3dqn_checkpoint)
    network_type = CandidateConditionedNCDMQNetwork
    def __init__(self, checkpoint_path=None, ncdm=None, q_matrix: torch.Tensor|None=None, device: str | torch.device = "cpu", expected_protocol_config: dict[str, Any] | None = None, network=None, cache: NCDMItemFeatureCache | None = None, selection_horizon: int | None = None, alpha_fit: dict | None = None, candidate_pool_config: dict | None=None) -> None:
        self.device=torch.device(device); self.ncdm=ncdm.to(self.device).eval() if hasattr(ncdm,"to") else ncdm; self.q_matrix=q_matrix.to(self.device).float()
        if network is None:
            self.network, meta = self.loader(checkpoint_path, ncdm=self.ncdm, q_matrix=self.q_matrix, device=self.device, expected_protocol_config=expected_protocol_config)
            self.selection_horizon=int(meta["selection_horizon"]); self.alpha_fit=dict(meta.get("alpha_fit") or {}); self.candidate_pool_config=dict(candidate_pool_config or meta.get("candidate_pool_config") or {})
        else:
            self.network=network.to(self.device).eval(); self.selection_horizon=int(selection_horizon or (expected_protocol_config or {}).get("selection_horizon",5)); self.alpha_fit=dict(alpha_fit or {"initial_steps":8,"incremental_steps":3,"lr":0.05,"early_stop_tol":1e-5}); self.candidate_pool_config=dict(candidate_pool_config or {})
        self.cache=cache or NCDMItemFeatureCache(self.ncdm,self.q_matrix,self.device)
        self.prefilter=NCDMCandidatePrefilter(self.q_matrix,self.cache,self.ncdm,self.candidate_pool_config)
        self.cached_alpha=None; self.cached_history_item_ids=[]; self.cached_history_responses=[]; self.cached_history_length=0
    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context); self.cached_alpha=None; self.cached_history_item_ids=[]; self.cached_history_responses=[]; self.cached_history_length=0
    def _fit_alpha_cached(self, history_item_ids, history_responses):
        use_inc=(self.cached_alpha is not None and len(history_item_ids)==self.cached_history_length+1 and list(history_item_ids)[:-1]==self.cached_history_item_ids and list(history_responses)[:-1]==self.cached_history_responses)
        init=self.cached_alpha if use_inc else None
        cfg=dict(self.alpha_fit); steps=int(cfg.get("incremental_steps", cfg.get("steps",3))) if init is not None else int(cfg.get("initial_steps", cfg.get("steps",8)))
        with torch.enable_grad():
            alpha=fit_student_alpha(self.ncdm,self.q_matrix,history_item_ids,history_responses,initial_alpha=init,steps=steps,lr=float(cfg.get("lr",0.05)),early_stop_tol=float(cfg.get("early_stop_tol",1e-5)),grad_clip=cfg.get("grad_clip"),device=self.device)
        self.cached_alpha=alpha.detach(); self.cached_history_item_ids=list(history_item_ids); self.cached_history_responses=list(history_responses); self.cached_history_length=len(history_item_ids); return alpha
    def select(self, candidate_item_ids: Sequence[int], history_item_ids: Sequence[int], history_responses: Sequence[float], context: dict[str, Any] | None = None) -> int:
        self._validate_context(context)
        if not candidate_item_ids: raise ValueError(f"{self.name} requires at least one candidate item")
        alpha=self._fit_alpha_cached(history_item_ids, history_responses)
        with torch.no_grad():
            mastery=torch.sigmoid(alpha).squeeze(0); coverage_count=self.cache.q_masks[torch.as_tensor(history_item_ids,dtype=torch.long,device=self.device)].sum(0) if history_item_ids else torch.zeros_like(mastery)
            filtered=self.prefilter.select(candidate_item_ids, alpha=alpha, mastery=mastery, coverage_count=coverage_count)
            coverage=(coverage_count/float(self.selection_horizon)).clamp(0,1); hist=self.cache.history(history_item_ids,history_responses,self.selection_horizon).unsqueeze(0)
            if hist.shape[1]==0: raise ValueError(f"{self.name} requires warm-start history before selection")
            batch={"history_features":hist,"history_mask":torch.ones((1,hist.shape[1]),dtype=torch.bool,device=self.device),"candidate_features":self.cache.candidate(filtered).unsqueeze(0),"candidate_mask":torch.ones((1,len(filtered)),dtype=torch.bool,device=self.device),"global_features":build_global_feature(mastery,coverage,int((context or {}).get("policy_step",len(history_item_ids))),self.selection_horizon).unsqueeze(0),"coverage_count":coverage_count.unsqueeze(0)}
            q_values,_=forward_q_network(self.network,batch)
            return int(filtered[int(q_values.argmax(dim=1).item())])

class SetC3DQNNCDMPolicy(C3DQNNCDMPolicy):
    name = "Set-C3DQN-NCDM"
    metadata = PolicyMetadata(name=name, implementation="ncdm_native", selection_model="set_conditioned_candidate_attention_dueling_double_dqn", evaluator_model="NCDM", uses_query_labels=False, uses_privileged_information=False)
    loader = staticmethod(load_set_c3dqn_checkpoint)
    network_type = SetConditionedNCDMQNetwork
