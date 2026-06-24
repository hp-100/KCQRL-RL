"""Candidate-conditioned dueling Q-networks for NCDM-native adaptive testing."""
from __future__ import annotations
import torch
import torch.nn as nn

NEG_INF_Q = -1.0e9

class CandidateConditionedNCDMQNetwork(nn.Module):
    def __init__(self, knowledge_dim: int, d_model: int = 128, n_heads: int = 4, num_history_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.d_model = int(d_model); self.n_heads = int(n_heads); self.num_history_layers = int(num_history_layers); self.dropout = float(dropout)
        self.history_feature_dim = 2 * self.knowledge_dim + 3
        self.candidate_feature_dim = 2 * self.knowledge_dim + 1
        self.global_feature_dim = 2 * self.knowledge_dim + 1
        self.history_projector = nn.Linear(self.history_feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_history_layers)
        self.candidate_projector = nn.Linear(self.candidate_feature_dim, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_projector = nn.Linear(self.global_feature_dim, d_model)
        self.value_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.advantage_head = nn.Sequential(nn.Linear(3 * d_model + 3 * self.knowledge_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.last_debug: dict[str, torch.Tensor] = {}

    def _assert_shapes(self, history_features, history_mask, candidate_features, candidate_mask, global_features) -> None:
        b, _, hf = history_features.shape; bc, _, cf = candidate_features.shape
        if b != bc or history_mask.shape[:1] != (b,) or candidate_mask.shape[:1] != (b,) or global_features.shape != (b, self.global_feature_dim):
            raise ValueError("C3DQN batch dimensions are inconsistent")
        if hf != self.history_feature_dim or cf != self.candidate_feature_dim:
            raise ValueError(f"feature dims must be history={self.history_feature_dim}, candidate={self.candidate_feature_dim}")
        if not (history_mask.any(dim=1).all() and candidate_mask.any(dim=1).all()):
            raise ValueError("each sample must have at least one valid history and candidate")

    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, return_attention: bool = False, coverage_count=None):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        if not all(torch.isfinite(x).all() for x in (history_features, candidate_features, global_features)):
            raise ValueError("non-finite C3DQN input")
        key_padding_mask = ~history_mask.bool()
        hist_emb = self.history_projector(history_features)
        encoded_history = self.history_encoder(hist_emb, src_key_padding_mask=key_padding_mask)
        candidate_embeddings = self.candidate_projector(candidate_features)
        candidate_context, attn = self.cross_attention(candidate_embeddings, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=return_attention)
        global_embedding = self.global_projector(global_features)
        masked_hist = encoded_history * history_mask.unsqueeze(-1).float()
        pooled = masked_hist.sum(dim=1) / history_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        value = self.value_head(torch.cat([pooled, global_embedding], dim=-1))
        mastery = global_features[:, :self.knowledge_dim]
        candidate_q_mask = candidate_features[:, :, :self.knowledge_dim]
        candidate_masked_difficulty = candidate_features[:, :, self.knowledge_dim:2*self.knowledge_dim]
        mastered = mastery.unsqueeze(1) * candidate_q_mask
        weakness = (1.0 - mastery).unsqueeze(1) * candidate_q_mask
        difficulty_gap = mastered - candidate_masked_difficulty
        global_b = global_embedding.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        adv_in = torch.cat([candidate_embeddings, candidate_context, global_b, mastered, weakness, difficulty_gap], dim=-1)
        advantage = self.advantage_head(adv_in).squeeze(-1)
        valid = candidate_mask.bool()
        mean_adv = advantage.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        q_values = (value + advantage - mean_adv).masked_fill(~valid, NEG_INF_Q)
        if not torch.isfinite(q_values).all():
            raise ValueError("non-finite q_values")
        self.last_debug = {"value": value.detach()}
        if return_attention:
            self.last_debug.update({"advantage": advantage.detach(), "masked_mean_advantage": mean_adv.detach(), "mastered": mastered.detach(), "weakness": weakness.detach(), "difficulty_gap": difficulty_gap.detach(), "candidate_context": candidate_context.detach()})
        return q_values, (attn if return_attention else None)

class MAB(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout:float=0.0):
        super().__init__(); self.attn=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True); self.ln1=nn.LayerNorm(d_model); self.ff=nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(),nn.Linear(d_model,d_model)); self.ln2=nn.LayerNorm(d_model)
    def forward(self, q,k, key_padding_mask=None):
        h,_=self.attn(q,k,k,key_padding_mask=key_padding_mask,need_weights=False); x=self.ln1(q+h); return self.ln2(x+self.ff(x))

class ISAB(nn.Module):
    def __init__(self, d_model:int, n_heads:int, num_inducing_points:int, dropout:float=0.0):
        super().__init__(); self.inducing=nn.Parameter(torch.randn(num_inducing_points,d_model)*0.02); self.mab1=MAB(d_model,n_heads,dropout); self.mab2=MAB(d_model,n_heads,dropout)
    def forward(self,x,mask):
        b=x.shape[0]; i=self.inducing.unsqueeze(0).expand(b,-1,-1); h=self.mab1(i,x,key_padding_mask=~mask.bool()); return self.mab2(x,h)

class SetConditionedNCDMQNetwork(CandidateConditionedNCDMQNetwork):
    def __init__(self, knowledge_dim:int, d_model:int=128, n_heads:int=4, num_history_layers:int=2, dropout:float=0.1, candidate_set_encoder:str="isab", num_set_layers:int=1, num_inducing_points:int=16, set_attention_heads:int|None=None, use_relative_features:bool=True, set_pool_in_value_head:bool=True, full_attention_max_candidates:int=128, debug_mode:bool=False) -> None:
        super().__init__(knowledge_dim,d_model,n_heads,num_history_layers,dropout)
        self.candidate_set_encoder=candidate_set_encoder; self.num_set_layers=int(num_set_layers); self.num_inducing_points=int(num_inducing_points); self.set_attention_heads=int(set_attention_heads or n_heads); self.use_relative_features=bool(use_relative_features); self.set_pool_in_value_head=bool(set_pool_in_value_head); self.full_attention_max_candidates=int(full_attention_max_candidates); self.debug_mode=bool(debug_mode)
        if candidate_set_encoder not in {"none","full_self_attention","isab"}: raise ValueError(f"unknown candidate_set_encoder: {candidate_set_encoder}")
        if candidate_set_encoder == "full_self_attention":
            layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=self.set_attention_heads,dropout=dropout,batch_first=True)
            self.set_encoder=nn.TransformerEncoder(layer,num_layers=self.num_set_layers)
        elif candidate_set_encoder == "isab":
            self.set_encoder=nn.ModuleList([ISAB(d_model,self.set_attention_heads,self.num_inducing_points,dropout) for _ in range(self.num_set_layers)])
        else: self.set_encoder=nn.Identity()

    def forward(self, history_features, history_mask, candidate_features, candidate_mask, global_features, *, coverage_count=None, return_attention: bool=False, chunk_size:int|None=None):
        self._assert_shapes(history_features, history_mask, candidate_features, candidate_mask, global_features)
        key_padding_mask=~history_mask.bool(); hist_emb=self.history_projector(history_features)
        encoded_history=self.history_encoder(hist_emb,src_key_padding_mask=key_padding_mask)
        cand_emb=self.candidate_projector(candidate_features)
        if self.candidate_set_encoder == "full_self_attention":
            cand_emb=self.set_encoder(cand_emb, src_key_padding_mask=~candidate_mask.bool())
        elif self.candidate_set_encoder == "isab":
            for layer in self.set_encoder: cand_emb=layer(cand_emb,candidate_mask)
        candidate_context, attn = self.cross_attention(cand_emb, encoded_history, encoded_history, key_padding_mask=key_padding_mask, need_weights=return_attention)
        global_embedding=self.global_projector(global_features); masked_hist=encoded_history*history_mask.unsqueeze(-1).float(); pooled=masked_hist.sum(1)/history_mask.sum(1,keepdim=True).clamp_min(1).float()
        if self.set_pool_in_value_head:
            set_pool=(cand_emb*candidate_mask.unsqueeze(-1).float()).sum(1)/candidate_mask.sum(1,keepdim=True).clamp_min(1).float(); pooled=0.5*(pooled+set_pool)
        value=self.value_head(torch.cat([pooled,global_embedding],-1))
        mastery=global_features[:,:self.knowledge_dim]; qmask=candidate_features[:,:,:self.knowledge_dim]; diff=candidate_features[:,:,self.knowledge_dim:2*self.knowledge_dim]
        mastered=mastery.unsqueeze(1)*qmask; weakness=(1-mastery).unsqueeze(1)*qmask; difficulty_gap=mastered-diff
        global_b=global_embedding.unsqueeze(1).expand(-1,candidate_features.shape[1],-1)
        advantage=self.advantage_head(torch.cat([cand_emb,candidate_context,global_b,mastered,weakness,difficulty_gap],-1)).squeeze(-1)
        valid=candidate_mask.bool(); mean_adv=advantage.masked_fill(~valid,0).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1).float(); q=(value+advantage-mean_adv).masked_fill(~valid,NEG_INF_Q)
        self.last_debug={}
        if self.debug_mode or return_attention:
            self.last_debug={"advantage":advantage.detach(),"masked_mean_advantage":mean_adv.detach(),"set_aware_candidate":cand_emb.detach(),"relative_features":difficulty_gap.detach()}
        return q,(attn if return_attention else None)
