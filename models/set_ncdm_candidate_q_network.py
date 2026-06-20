"""Set-conditioned dueling Q-network for NCDM-native adaptive testing."""
from __future__ import annotations
import torch
import torch.nn as nn
from models.ncdm_candidate_q_network import NEG_INF_Q, CandidateConditionedNCDMQNetwork
from models.set_attention import InducedSetAttentionBlock, MultiheadAttentionBlock

RELATIVE_FEATURE_NAMES=["novelty_ratio","covered_overlap_ratio","mean_mastery_gap","weakness_targeting","concept_count_norm"]

def masked_mean(x, mask, dim=1, keepdim=False):
    m=mask.unsqueeze(-1).to(x.dtype)
    return (x*m).sum(dim=dim,keepdim=keepdim)/m.sum(dim=dim,keepdim=keepdim).clamp_min(1)

class SetConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim:int, d_model:int=128, n_heads:int=4, num_history_layers:int=2, dropout:float=0.1, candidate_set_encoder:str="isab", num_set_layers:int=1, num_inducing_points:int=16, set_attention_heads:int|None=None, use_relative_features:bool=True, set_pool_in_value_head:bool=True, full_attention_max_candidates:int=128):
        super().__init__(); self.knowledge_dim=int(knowledge_dim); self.d_model=d_model
        self.history_feature_dim=2*self.knowledge_dim+3; self.candidate_feature_dim=2*self.knowledge_dim+1; self.global_feature_dim=2*self.knowledge_dim+1
        self.candidate_set_encoder=str(candidate_set_encoder); self.use_relative_features=bool(use_relative_features); self.set_pool_in_value_head=bool(set_pool_in_value_head); self.full_attention_max_candidates=int(full_attention_max_candidates)
        self.history_projector=nn.Linear(self.history_feature_dim,d_model)
        self.history_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dropout=dropout,batch_first=True),num_layers=num_history_layers)
        self.candidate_projector=nn.Linear(self.candidate_feature_dim,d_model); self.cross_attention=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.global_projector=nn.Linear(self.global_feature_dim,d_model)
        rel_dim=5 if self.use_relative_features else 0; self.relative_feature_dim=rel_dim
        self.cognitive_projector=nn.Linear(3*self.knowledge_dim+rel_dim,d_model)
        self.local_norm=nn.LayerNorm(d_model)
        heads=int(set_attention_heads or n_heads)
        layers=[]
        for _ in range(int(num_set_layers)):
            if self.candidate_set_encoder=="isab": layers.append(InducedSetAttentionBlock(d_model,heads,int(num_inducing_points),dropout))
            elif self.candidate_set_encoder=="full_self_attention": layers.append(MultiheadAttentionBlock(d_model,heads,dropout))
            elif self.candidate_set_encoder=="none": pass
            else: raise ValueError("candidate_set_encoder must be none, full_self_attention, or isab")
        self.set_layers=nn.ModuleList(layers)
        v_in=3*d_model if self.set_pool_in_value_head else 2*d_model
        self.value_head=nn.Sequential(nn.Linear(v_in,d_model),nn.ReLU(),nn.Linear(d_model,1))
        self.advantage_head=nn.Sequential(nn.Linear(3*d_model+rel_dim,d_model),nn.ReLU(),nn.Linear(d_model,1))
        self.last_debug={}
    def _relative(self,candidate_features,global_features,coverage_count):
        b,c,_=candidate_features.shape; k=self.knowledge_dim; q=candidate_features[:,:,:k]; diff=candidate_features[:,:,k:2*k]
        mastery=global_features[:,:k]
        cov=coverage_count.to(candidate_features.device).float()
        denom=q.sum(-1,keepdim=True).clamp_min(1)
        novelty=(q*(cov.unsqueeze(1)==0).float()).sum(-1,keepdim=True)/denom
        covered=(q*(cov.unsqueeze(1)>0).float()).sum(-1,keepdim=True)/denom
        gap=((mastery.unsqueeze(1)-diff).abs()*q).sum(-1,keepdim=True)/denom
        weak=((1-mastery).unsqueeze(1)*q).sum(-1,keepdim=True)/denom
        cnt=q.sum(-1,keepdim=True)/float(k)
        out=torch.cat([novelty,covered,gap,weak,cnt],-1)
        return torch.nan_to_num(out,0.0,0.0,0.0)
    def _encode_common(self,hist,hmask,cand,cmask,glob,coverage_count):
        key_padding_mask=~hmask.bool(); eh=self.history_encoder(self.history_projector(hist),src_key_padding_mask=key_padding_mask)
        ce=self.candidate_projector(cand); ctx,_=self.cross_attention(ce,eh,eh,key_padding_mask=key_padding_mask,need_weights=False)
        ge=self.global_projector(glob); pooled=masked_mean(eh,hmask)
        k=self.knowledge_dim; mastery=glob[:,:k]; qmask=cand[:,:,:k]; diff=cand[:,:,k:2*k]
        mastered=mastery.unsqueeze(1)*qmask; weakness=(1-mastery).unsqueeze(1)*qmask; gap=mastered-diff
        rel=self._relative(cand,glob,coverage_count) if self.use_relative_features else cand.new_zeros((*cand.shape[:2],0))
        cog=self.cognitive_projector(torch.cat([mastered,weakness,gap,rel],-1))
        local=self.local_norm(ce+ctx+ge.unsqueeze(1)+cog)
        set_aware=local
        if self.candidate_set_encoder=="full_self_attention" and cand.shape[1]>self.full_attention_max_candidates: raise ValueError("full_self_attention candidate count exceeds full_attention_max_candidates")
        for layer in self.set_layers:
            if self.candidate_set_encoder=="isab": set_aware=layer(set_aware,cmask)
            elif self.candidate_set_encoder=="full_self_attention": set_aware=layer(set_aware,set_aware,set_aware,key_padding_mask=~cmask.bool())*cmask.unsqueeze(-1).float()
        return pooled,ge,ctx,local,set_aware,rel
    def _raw_adv(self,ctx,local,set_aware,rel): return self.advantage_head(torch.cat([local,set_aware,ctx,rel],-1)).squeeze(-1)
    def forward(self,hist,hmask,cand,cmask,glob,coverage_count,return_attention:bool=False):
        pooled,ge,ctx,local,set_aware,rel=self._encode_common(hist,hmask,cand,cmask,glob,coverage_count)
        v_in=[pooled,ge];
        if self.set_pool_in_value_head: v_in.append(masked_mean(set_aware,cmask))
        value=self.value_head(torch.cat(v_in,-1)); adv=self._raw_adv(ctx,local,set_aware,rel); valid=cmask.bool(); mean=(adv.masked_fill(~valid,0).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1).float())
        q=(value+adv-mean).masked_fill(~valid,NEG_INF_Q); self.last_debug={"relative_features":rel.detach(),"set_aware_candidate":set_aware.detach(),"advantage":adv.detach(),"masked_mean_advantage":mean.detach()}; return q,None
    def forward_chunked(self,hist,hmask,cand,cmask,glob,coverage_count,chunk_size:int=128):
        # Scoring chunked only: common candidate-history and set context are still materialized for Top-K candidates.
        pooled,ge,ctx,local,set_aware,rel=self._encode_common(hist,hmask,cand,cmask,glob,coverage_count)
        v_in=[pooled,ge];
        if self.set_pool_in_value_head: v_in.append(masked_mean(set_aware,cmask))
        value=self.value_head(torch.cat(v_in,-1)); chunks=[]
        for s in range(0,cand.shape[1],int(chunk_size)): chunks.append(self._raw_adv(ctx[:,s:s+chunk_size],local[:,s:s+chunk_size],set_aware[:,s:s+chunk_size],rel[:,s:s+chunk_size]))
        adv=torch.cat(chunks,1); valid=cmask.bool(); mean=adv.masked_fill(~valid,0).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1).float()
        return (value+adv-mean).masked_fill(~valid,NEG_INF_Q), None
