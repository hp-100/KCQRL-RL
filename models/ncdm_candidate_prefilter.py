"""Shared deterministic NCDM candidate Top-K prefilter for training and evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Any
import torch

@dataclass(frozen=True)
class PrefilterResult:
    candidate_ids:list[int]
    scores:list[float]
    score_summary:dict[str,float]

class NCDMCandidatePrefilter:
    def __init__(self, q_matrix:torch.Tensor, config:dict[str,Any]|None=None):
        self.q_matrix=q_matrix.float()
        self.config=dict(config or {})
    def select(self, candidate_item_ids:Sequence[int], mastery:torch.Tensor, coverage_count:torch.Tensor, *, top_k:int|None=None)->PrefilterResult:
        ids=[int(x) for x in candidate_item_ids]
        if not ids: return PrefilterResult([],[],{"count":0})
        enabled=bool(self.config.get("prefilter_enabled", top_k is not None))
        k=int(top_k or self.config.get("prefilter_top_k", len(ids)))
        if not enabled or k>=len(ids):
            return PrefilterResult(ids,[0.0]*len(ids),{"count":len(ids),"min":0.0,"max":0.0,"mean":0.0})
        dev=mastery.device; q=self.q_matrix.to(dev)[torch.tensor(ids,dtype=torch.long,device=dev)].clamp(0,1)
        mastery=mastery.flatten().to(dev); cov=coverage_count.flatten().to(dev)
        denom=q.sum(-1).clamp_min(1)
        weakness=((1-mastery)*q).sum(-1)/denom
        novelty=(q*(cov==0).float()).sum(-1)/denom
        gap=((mastery.unsqueeze(0)-0.5).abs()*q).sum(-1)/denom
        w=self.config.get("weights",{}) or {}
        score=float(w.get("weakness",1.0))*weakness+float(w.get("novelty",0.5))*novelty-float(w.get("gap",0.1))*gap
        order=torch.argsort(score,descending=True,stable=True).tolist()
        chosen=[]; seen=set()
        for idx in order:
            if ids[idx] not in seen:
                chosen.append(idx); seen.add(ids[idx])
            if len(chosen)>=k: break
        # optional diversity: reserve low-coverage concepts if requested
        div=int(self.config.get("diversity_quota",0) or 0)
        if div>0 and len(chosen)==k:
            low=torch.argsort(cov).tolist()
            for concept in low:
                if cov[concept]>0: break
                hits=[i for i in order if q[i,concept]>0 and i not in chosen]
                if hits:
                    chosen[-1]=hits[0]
                    break
        out_ids=[ids[i] for i in chosen]
        out_scores=[float(score[i].detach().cpu()) for i in chosen]
        return PrefilterResult(out_ids,out_scores,{"count":len(out_ids),"min":min(out_scores),"max":max(out_scores),"mean":sum(out_scores)/len(out_scores)})
