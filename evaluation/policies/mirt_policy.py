from __future__ import annotations
import torch
import numpy as np
from .base import BaseCATPolicy, PolicyMetadata
from models.mirt import fit_student_theta, predict_with_theta

class HeuristicMIRTPolicy(BaseCATPolicy):
    def __init__(self, name: str):
        self.name = name
        self.metadata = PolicyMetadata(name=name, implementation="heuristic", notes="Simplified proxy, not a formal MIRT implementation.")
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        preds = context["predict_history"](history_item_ids, history_responses, candidate_item_ids)
        if self.name == "MIRT-MFI":
            idx = int(np.argmax([p * (1 - p) for p in preds]))
        else:
            idx = int(np.argmax([abs(p - 0.5) for p in preds]))
        return int(list(candidate_item_ids)[idx])

class FormalMIRTPolicy(BaseCATPolicy):
    _IMPL = {"MIRT-Trace-MFI":"formal_mirt_trace_fisher", "MIRT-D-opt":"formal_mirt_d_opt", "MIRT-MKLI":"formal_mirt_mkli_mc", "MIRT-Local-KLI":"local_finite_difference_approximation"}
    def __init__(self, name, model, *, theta_cfg=None, d_opt_ridge=0.01, mkli_samples=16, mkli_scale=0.25, device=None):
        self.name=name; self.model=model; self.theta_cfg=dict(theta_cfg or {}); self.d_opt_ridge=float(d_opt_ridge)
        self.mkli_samples=int(mkli_samples); self.mkli_scale=float(mkli_scale); self.device=torch.device(device or next(model.parameters()).device)
        self.metadata = PolicyMetadata(name=name, implementation=self._IMPL[name], uses_privileged_information=False,
            selection_model="mirt", evaluator_model="ncdm", uses_query_labels=False,
            notes="selection_model=mirt; evaluator_model=ncdm; uses_query_labels=false")
    def _theta(self, h_i, h_r):
        return fit_student_theta(self.model, h_i, h_r, device=self.device, **self.theta_cfg)
    def _cand(self, candidate_item_ids):
        return torch.tensor(list(candidate_item_ids), dtype=torch.long, device=self.device)
    def trace_mfi_scores(self, theta, candidate_item_ids):
        ids=self._cand(candidate_item_ids); a=self.model.disc_emb(ids); p=predict_with_theta(self.model, theta, ids)
        return p*(1-p)*a.pow(2).sum(dim=1)
    def d_opt_scores(self, theta, candidate_item_ids, history_item_ids):
        d=self.model.n_dims; info=self.d_opt_ridge*torch.eye(d,device=self.device)
        if history_item_ids:
            h=torch.tensor(list(history_item_ids),dtype=torch.long,device=self.device); ah=self.model.disc_emb(h); ph=predict_with_theta(self.model, theta, h); w=ph*(1-ph)
            info=info+torch.einsum('n,nd,ne->de', w, ah, ah)
        ids=self._cand(candidate_item_ids); a=self.model.disc_emb(ids); p=predict_with_theta(self.model, theta, ids); w=p*(1-p)
        scores=[]
        for wi,ai in zip(w,a):
            mat=info+wi*torch.outer(ai,ai)
            scores.append(torch.linalg.slogdet(mat).logabsdet)
        return torch.stack(scores)
    def mkli_scores(self, theta, candidate_item_ids):
        ids=self._cand(candidate_item_ids); base=predict_with_theta(self.model, theta, ids).clamp(1e-7,1-1e-7)
        g=torch.Generator(device=self.device).manual_seed(12345); half=max(1,self.mkli_samples//2)
        eps=torch.randn((half,self.model.n_dims),generator=g,device=self.device)*self.mkli_scale; deltas=torch.cat([eps,-eps],dim=0)[:self.mkli_samples]
        vals=[]
        for delta in deltas:
            q=predict_with_theta(self.model, theta+delta, ids).clamp(1e-7,1-1e-7)
            vals.append(base*(base/q).log()+(1-base)*((1-base)/(1-q)).log())
        return torch.stack(vals).mean(dim=0)
    def local_kli_scores(self, theta, candidate_item_ids):
        vals=[]
        for i in range(self.model.n_dims):
            e=torch.zeros_like(theta); e[i]=0.1
            vals.append(self.mkli_scores(theta+e, candidate_item_ids)); vals.append(self.mkli_scores(theta-e, candidate_item_ids))
        return torch.stack(vals).max(dim=0).values
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        theta=self._theta(history_item_ids, history_responses)
        if self.name=="MIRT-Trace-MFI": scores=self.trace_mfi_scores(theta,candidate_item_ids)
        elif self.name=="MIRT-D-opt": scores=self.d_opt_scores(theta,candidate_item_ids,history_item_ids)
        elif self.name=="MIRT-MKLI": scores=self.mkli_scores(theta,candidate_item_ids)
        else: scores=self.local_kli_scores(theta,candidate_item_ids)
        return int(list(candidate_item_ids)[int(torch.argmax(scores).detach().cpu())])
