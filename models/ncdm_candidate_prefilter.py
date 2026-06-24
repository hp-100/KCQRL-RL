"""Shared deterministic NCDM diagnostic candidate prefilter."""
from __future__ import annotations
from typing import Any, Sequence
import torch

class NCDMCandidatePrefilter:
    def __init__(self, q_matrix: torch.Tensor, feature_cache=None, ncdm=None, config: dict[str, Any] | None=None):
        self.q_matrix=q_matrix.float().to(feature_cache.device if feature_cache is not None else q_matrix.device)
        self.feature_cache=feature_cache; self.ncdm=ncdm; self.config=dict(config or {})
        self.last_summary: dict[str, Any] = {}
    def select(self, candidate_item_ids: Sequence[int], alpha=None, mastery=None, coverage_count=None):
        ids=torch.as_tensor(list(map(int,candidate_item_ids)),dtype=torch.long,device=self.q_matrix.device)
        if ids.numel()==0: return []
        top_k=int(self.config.get('prefilter_top_k', self.config.get('max_candidates', ids.numel())) or ids.numel())
        if not self.config.get('prefilter_enabled', True):
            out=ids[:top_k].detach().cpu().tolist(); self.last_summary={'raw_candidate_count':int(ids.numel()),'filtered_candidate_count':len(out)}; return out
        qmask=self.q_matrix[ids].clamp(0,1); cc=qmask.sum(-1).clamp_min(1)
        if mastery is None: mastery=torch.full((self.q_matrix.shape[1],),0.5,device=ids.device)
        mastery=torch.as_tensor(mastery,device=ids.device,dtype=torch.float32).flatten()
        if coverage_count is None: coverage_count=torch.zeros_like(mastery)
        coverage_count=torch.as_tensor(coverage_count,device=ids.device,dtype=torch.float32).flatten()
        if self.ncdm is not None and alpha is not None and hasattr(self.ncdm,'predict_with_alpha'):
            with torch.no_grad(): p=self.ncdm.predict_with_alpha(alpha,ids,self.q_matrix).float().flatten()
            uncertainty=4*p*(1-p)
        else: uncertainty=torch.ones_like(cc)*0.5
        weakness=((1-mastery).unsqueeze(0)*qmask).sum(-1)/cc
        novelty=(qmask*(coverage_count==0).float().unsqueeze(0)).sum(-1)/cc
        if self.feature_cache is not None:
            md=self.feature_cache.masked_difficulties[ids]; dn=self.feature_cache.disc_norms[ids].mean(-1)
            diff_raw=(md - mastery.unsqueeze(0)*qmask).abs().sum(-1)/cc
            difficulty=(1-diff_raw.clamp(0,1))
            discrimination=dn.clamp(0,1)
        else:
            difficulty=torch.ones_like(cc)*0.5; discrimination=torch.ones_like(cc)*0.5
        w=dict(self.config.get('weights') or {})
        score=float(w.get('uncertainty',0.35))*uncertainty+float(w.get('weakness',0.25))*weakness+float(w.get('novelty',0.15))*novelty+float(w.get('difficulty',0.15))*difficulty+float(w.get('discrimination',0.10))*discrimination
        quota=min(int(self.config.get('diversity_quota',0) or 0), top_k, ids.numel())
        primary_slots=min(top_k-quota, ids.numel())
        order=torch.argsort(score,descending=True,stable=True)
        selected=[]; used=set()
        for j in order[:primary_slots].tolist(): selected.append(int(ids[j])); used.add(int(j))
        current=(qmask[[j for j in used]].sum(0) if used else coverage_count.clone())
        remaining=[j for j in order.tolist() if j not in used]
        for _ in range(quota):
            if not remaining: break
            div_scores=[]
            low=(current<=current.min()).float()
            for j in remaining: div_scores.append((float((qmask[j]*low).sum().item()), float(score[j].item()), -int(ids[j].item()), j))
            best=max(div_scores); j=best[3]
            selected.append(int(ids[j])); used.add(j); remaining.remove(j); current=current+qmask[j]
        if len(selected)<min(top_k,ids.numel()):
            for j in order.tolist():
                if j not in used: selected.append(int(ids[j])); used.add(j)
                if len(selected)>=min(top_k,ids.numel()): break
        self.last_summary={'raw_candidate_count':int(ids.numel()),'filtered_candidate_count':len(selected),'score_mean':float(score.mean()),'score_max':float(score.max())}
        return selected
