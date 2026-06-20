"""NCDM-native C3DQN and Set-C3DQN evaluation policies."""
from __future__ import annotations
from typing import Any, Sequence
import random, torch
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from models.ncdm import fit_student_alpha
from models.ncdm_candidate_features import NCDMItemFeatureCache, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from agents.ncdm_c3dqn_trainer import load_c3dqn_checkpoint, load_set_c3dqn_checkpoint

class RandomNCDMPolicy(BaseCATPolicy):
    name="Random-NCDM"; metadata=PolicyMetadata(name=name,implementation="ncdm_native",selection_model="random",evaluator_model="NCDM")
    def reset(self, student_id, seed, context): super().reset(student_id,seed,context); self.rng=random.Random(f"{student_id}:{seed}")
    def select(self,candidate_item_ids,history_item_ids,history_responses,context): return int(self.rng.choice(list(candidate_item_ids)))

class _AlphaProtocolMixin:
    def reset(self, student_id, seed, context):
        super().reset(student_id,seed,context); self.cached_alpha=None; self.cached_history_length=0; self.last_prefiltered_candidate_ids=[]; self.last_raw_candidate_ids=[]; self.last_prefilter_score_summary={}
    def _fit_alpha_cached(self, history_item_ids, history_responses):
        initial=int(self.alpha_fit.get("initial_steps", self.alpha_fit.get("steps",8)))
        incr=int(self.alpha_fit.get("incremental_steps", self.alpha_fit.get("steps",initial)))
        if self.cached_alpha is not None and len(history_item_ids)==self.cached_history_length+1:
            steps=incr; init=self.cached_alpha
        else:
            steps=initial; init=None
        alpha=fit_student_alpha(self.ncdm,self.q_matrix,history_item_ids,history_responses,initial_alpha=init,steps=steps,lr=float(self.alpha_fit.get("lr",0.05)),early_stop_tol=float(self.alpha_fit.get("early_stop_tol",1e-5)),grad_clip=self.alpha_fit.get("grad_clip",None),device=self.device)
        self.cached_alpha=alpha.detach(); self.cached_history_length=len(history_item_ids); self.last_alpha_steps=steps; self.last_alpha_initial_was_cached=init is not None
        return alpha
    def _state_tensors(self,candidate_item_ids,history_item_ids,history_responses,context):
        with torch.enable_grad(): alpha=self._fit_alpha_cached(history_item_ids,history_responses)
        mastery=torch.sigmoid(alpha).squeeze(0)
        coverage_count=self.cache.q_masks[torch.as_tensor(history_item_ids,dtype=torch.long,device=self.device)].sum(0) if history_item_ids else torch.zeros_like(mastery)
        self.last_raw_candidate_ids=list(map(int,candidate_item_ids))
        pre=self.prefilter.select(candidate_item_ids,mastery,coverage_count)
        cids=pre.candidate_ids; self.last_prefiltered_candidate_ids=cids; self.last_prefilter_score_summary=pre.score_summary
        coverage=(coverage_count/float(self.selection_horizon)).clamp(0,1)
        hist=self.cache.history(history_item_ids,history_responses,self.selection_horizon).unsqueeze(0)
        if hist.shape[1]==0: raise ValueError(f"{self.name} requires warm-start history before selection")
        hmask=torch.ones((1,hist.shape[1]),dtype=torch.bool,device=self.device); cand=self.cache.candidate(cids).unsqueeze(0); cmask=torch.ones((1,len(cids)),dtype=torch.bool,device=self.device)
        glob=build_global_feature(mastery,coverage,int((context or {}).get("policy_step",len(history_item_ids))),self.selection_horizon).unsqueeze(0)
        return cids,hist,hmask,cand,cmask,glob,coverage_count.unsqueeze(0)

class C3DQNNCDMPolicy(_AlphaProtocolMixin, BaseCATPolicy):
    name="C3DQN-NCDM"; metadata=PolicyMetadata(name=name,implementation="ncdm_native",selection_model="candidate_conditioned_attention_dueling_double_dqn",evaluator_model="NCDM",uses_query_labels=False,uses_privileged_information=False)
    def __init__(self, checkpoint_path, ncdm, q_matrix:torch.Tensor, device="cpu", expected_protocol_config=None, network:CandidateConditionedNCDMQNetwork|None=None, cache=None, selection_horizon=None, alpha_fit=None):
        self.device=torch.device(device); self.ncdm=ncdm.to(self.device).eval() if hasattr(ncdm,"to") else ncdm; self.q_matrix=q_matrix.to(self.device).float()
        if network is None:
            self.network,meta=load_c3dqn_checkpoint(checkpoint_path,ncdm=self.ncdm,q_matrix=self.q_matrix,device=self.device,expected_protocol_config=expected_protocol_config); self.selection_horizon=int(meta["selection_horizon"]); self.alpha_fit=dict(meta.get("alpha_fit") or {}); self.candidate_pool_config=dict(meta.get("candidate_pool_config") or {})
        else:
            self.network=network.to(self.device).eval(); self.selection_horizon=int(selection_horizon or 5); self.alpha_fit=dict(alpha_fit or {"initial_steps":8,"incremental_steps":3,"lr":0.05}); self.candidate_pool_config={}
        self.cache=cache or NCDMItemFeatureCache(self.ncdm,self.q_matrix,self.device); self.prefilter=NCDMCandidatePrefilter(self.q_matrix,self.candidate_pool_config); self.cached_alpha=None; self.cached_history_length=0
    def select(self,candidate_item_ids,history_item_ids,history_responses,context=None):
        if set((context or {}).keys()) & {"query_item_ids","query_responses","future_responses","query_labels"}: raise ValueError("C3DQN-NCDM policy received privileged context")
        cids,h,hmask,c,cmask,g,cov=self._state_tensors(candidate_item_ids,history_item_ids,history_responses,context)
        with torch.no_grad(): q,_=self.network(h,hmask,c,cmask,g)
        return int(cids[int(q.argmax(1).item())])

class SetC3DQNNCDMPolicy(C3DQNNCDMPolicy):
    name="Set-C3DQN-NCDM"; metadata=PolicyMetadata(name=name,implementation="ncdm_native",selection_model="set_conditioned_candidate_attention_dueling_double_dqn",evaluator_model="NCDM",uses_query_labels=False,uses_privileged_information=False)
    def __init__(self, checkpoint_path, ncdm, q_matrix:torch.Tensor, device="cpu", expected_protocol_config=None, network:SetConditionedNCDMQNetwork|None=None, cache=None, selection_horizon=None, alpha_fit=None):
        self.device=torch.device(device); self.ncdm=ncdm.to(self.device).eval() if hasattr(ncdm,"to") else ncdm; self.q_matrix=q_matrix.to(self.device).float()
        if network is None:
            self.network,meta=load_set_c3dqn_checkpoint(checkpoint_path,ncdm=self.ncdm,q_matrix=self.q_matrix,device=self.device,expected_protocol_config=expected_protocol_config); self.selection_horizon=int(meta["selection_horizon"]); self.alpha_fit=dict(meta.get("alpha_fit") or {}); self.candidate_pool_config=dict(meta.get("candidate_pool_config") or {})
        else:
            self.network=network.to(self.device).eval(); self.selection_horizon=int(selection_horizon or 5); self.alpha_fit=dict(alpha_fit or {"initial_steps":8,"incremental_steps":3,"lr":0.05}); self.candidate_pool_config={}
        self.cache=cache or NCDMItemFeatureCache(self.ncdm,self.q_matrix,self.device); self.prefilter=NCDMCandidatePrefilter(self.q_matrix,self.candidate_pool_config); self.cached_alpha=None; self.cached_history_length=0
    def select(self,candidate_item_ids,history_item_ids,history_responses,context=None):
        cids,h,hmask,c,cmask,g,cov=self._state_tensors(candidate_item_ids,history_item_ids,history_responses,context)
        with torch.no_grad():
            if bool((context or {}).get("chunked",False)): q,_=self.network.forward_chunked(h,hmask,c,cmask,g,cov,int((context or {}).get("chunk_size",128)))
            else: q,_=self.network(h,hmask,c,cmask,g,cov)
        return int(cids[int(q.argmax(1).item())])
