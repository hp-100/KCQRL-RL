"""Student-conditioned Set-C3DQN network for NCDM-native adaptive testing."""
from __future__ import annotations
import torch
import torch.nn as nn
from models.ncdm_candidate_q_network import NEG_INF_Q
from models.set_attention import MultiheadAttentionBlock, InducedSetAttentionBlock

RELATIVE_FEATURE_NAMES=["novelty_ratio","covered_overlap_ratio","mean_mastery_gap","weakness_targeting","concept_count_norm"]

class SetConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim:int, d_model:int=128, n_heads:int=4, num_history_layers:int=2, dropout:float=0.1,
                 candidate_set_encoder:str="isab", num_set_layers:int=1, num_inducing_points:int=16, set_attention_heads:int|None=None,
                 use_relative_features:bool=True, set_pool_in_value_head:bool=True, full_attention_max_candidates:int=128, debug_mode:bool=False):
        super().__init__(); self.knowledge_dim=int(knowledge_dim); self.d_model=int(d_model)
        self.history_feature_dim=2*self.knowledge_dim+3; self.candidate_feature_dim=2*self.knowledge_dim+1; self.global_feature_dim=2*self.knowledge_dim+1
        self.candidate_set_encoder=candidate_set_encoder; self.num_set_layers=int(num_set_layers); self.num_inducing_points=int(num_inducing_points)
        self.set_attention_heads=int(set_attention_heads or n_heads); self.use_relative_features=bool(use_relative_features); self.relative_feature_names=list(RELATIVE_FEATURE_NAMES); self.relative_feature_dim=5 if self.use_relative_features else 0
        self.set_pool_in_value_head=bool(set_pool_in_value_head); self.full_attention_max_candidates=int(full_attention_max_candidates); self.debug_mode=bool(debug_mode)
        self.history_projector=nn.Linear(self.history_feature_dim,d_model); layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dropout=dropout,batch_first=True)
        self.history_encoder=nn.TransformerEncoder(layer,num_layers=num_history_layers); self.candidate_projector=nn.Linear(self.candidate_feature_dim,d_model)
        self.cross_attention=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True); self.global_projector=nn.Linear(self.global_feature_dim,d_model)
        self.cognitive_projector=nn.Linear(3*self.knowledge_dim,d_model); self.relative_projector=nn.Linear(self.relative_feature_dim,d_model) if self.use_relative_features else None
        if candidate_set_encoder=="none": self.set_layers=nn.ModuleList([])
        elif candidate_set_encoder=="full_self_attention": self.set_layers=nn.ModuleList([MultiheadAttentionBlock(d_model,self.set_attention_heads,dropout) for _ in range(self.num_set_layers)])
        elif candidate_set_encoder=="isab": self.set_layers=nn.ModuleList([InducedSetAttentionBlock(d_model,self.set_attention_heads,self.num_inducing_points,dropout) for _ in range(self.num_set_layers)])
        else: raise ValueError(f"unknown candidate_set_encoder: {candidate_set_encoder}")
        vdim=2*d_model+(d_model if self.set_pool_in_value_head else 0); self.value_head=nn.Sequential(nn.Linear(vdim,d_model),nn.ReLU(),nn.Linear(d_model,1))
        self.advantage_head=nn.Sequential(nn.Linear(2*d_model,d_model),nn.ReLU(),nn.Linear(d_model,1)); self.last_debug={}
    def _assert_shapes(self,h,hm,c,cm,g):
        if h.shape[0]!=c.shape[0] or g.shape!=(h.shape[0],self.global_feature_dim) or h.shape[-1]!=self.history_feature_dim or c.shape[-1]!=self.candidate_feature_dim: raise ValueError("Set-C3DQN batch dimensions are inconsistent")
        if not (hm.bool().any(1).all() and cm.bool().any(1).all()): raise ValueError("each sample must have at least one valid history and candidate")
    def _relative_features(self,candidate_features,global_features,coverage_count):
        k=self.knowledge_dim; q=candidate_features[:,:,:k]; diff=candidate_features[:,:,k:2*k]; mastery=global_features[:,:k]
        if coverage_count is None: coverage_count=torch.zeros((candidate_features.shape[0],k),device=candidate_features.device,dtype=candidate_features.dtype)
        coverage_count=coverage_count.to(candidate_features.device).float(); concept_count=q.sum(-1,keepdim=True).clamp_min(1.0)
        novelty=(q*(coverage_count.unsqueeze(1)==0).float()).sum(-1,keepdim=True)/concept_count
        covered=(q*(coverage_count.unsqueeze(1)>0).float()).sum(-1,keepdim=True)/concept_count
        gap=(torch.abs(mastery.unsqueeze(1)-diff)*q).sum(-1,keepdim=True)/concept_count
        weak=((1.0-mastery).unsqueeze(1)*q).sum(-1,keepdim=True)/concept_count
        count=q.sum(-1,keepdim=True)/float(k)
        return torch.cat([novelty,covered,gap,weak,count],-1)
    def _local(self,h,hm,c,cm,g,coverage_count=None):
        key_padding_mask=~hm.bool(); hist=self.history_encoder(self.history_projector(h),src_key_padding_mask=key_padding_mask); cand=self.candidate_projector(c)
        ctx,_=self.cross_attention(cand,hist,hist,key_padding_mask=key_padding_mask,need_weights=False)
        k=self.knowledge_dim; mastery=g[:,:k]; q=c[:,:,:k]; diff=c[:,:,k:2*k]; mastered=mastery.unsqueeze(1)*q; weakness=(1-mastery).unsqueeze(1)*q; difficulty_gap=mastered-diff
        local=cand+ctx+self.global_projector(g).unsqueeze(1)+self.cognitive_projector(torch.cat([mastered,weakness,difficulty_gap],-1))
        rel=None
        if self.use_relative_features:
            rel=self._relative_features(c,g,coverage_count); local=local+self.relative_projector(rel)
        return hist,cand,ctx,local,rel
    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, coverage_count=None, return_attention:bool=False):
        self._assert_shapes(history_features,history_mask,candidate_features,candidate_mask,global_features)
        valid=candidate_mask.bool(); eff=valid.sum(1).max().item()
        if self.candidate_set_encoder=="full_self_attention" and eff>self.full_attention_max_candidates: raise ValueError("full candidate self-attention exceeds configured candidate limit")
        hist,cand,ctx,x,rel=self._local(history_features,history_mask,candidate_features,candidate_mask,global_features,coverage_count)
        for layer in self.set_layers:
            if isinstance(layer,MultiheadAttentionBlock): x=layer(x,key_padding_mask=~valid)
            else: x=layer(x,key_padding_mask=~valid)
        masked_hist=hist*history_mask.unsqueeze(-1).float(); pooled=masked_hist.sum(1)/history_mask.sum(1,keepdim=True).clamp_min(1).float(); glob=self.global_projector(global_features)
        parts=[pooled,glob]
        if self.set_pool_in_value_head: parts.append((x*valid.unsqueeze(-1).float()).sum(1)/valid.sum(1,keepdim=True).clamp_min(1).float())
        value=self.value_head(torch.cat(parts,-1)); raw=self.advantage_head(torch.cat([x,ctx],-1)).squeeze(-1)
        mean=raw.masked_fill(~valid,0).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1).float(); q=value+raw-mean; q=q.masked_fill(~valid,NEG_INF_Q)
        self.last_debug={} if not self.debug_mode else {"local_candidate_representation":x.detach(),"relative_features": None if rel is None else rel.detach(),"raw_advantage":raw.detach()}
        return q, (None if not return_attention else {})
    def forward_chunked(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, coverage_count=None, chunk_size:int=64):
        # Full set encoders are not truly streamable; use full forward to preserve mathematical equivalence.
        return self.forward(history_features,history_mask,candidate_features,candidate_mask,global_features,coverage_count=coverage_count)
