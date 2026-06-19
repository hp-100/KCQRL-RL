from __future__ import annotations
from pathlib import Path
import torch
from .base import BaseCATPolicy, PolicyMetadata
from core.mirt_state_builder import ActionNormalizer, build_mirt_state, nearest_item
from models.mirt_recurrent_actor import MIRTRecurrentActor, ACTOR_ARCHITECTURE

class RDPGMIRTPolicy(BaseCATPolicy):
    name='RDPG-MIRT'
    metadata=PolicyMetadata(name=name, implementation='recurrent_deterministic_policy_gradient', selection_model='mirt', evaluator_model='mirt', uses_query_labels=False)
    metadata.actor_architecture='lstm_sequence_bptt'
    metadata.uses_semantic_features=False
    def __init__(self, checkpoint, mirt, theta_cfg=None, device='cpu'):
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.exists(): raise FileNotFoundError(f'RDPG-MIRT checkpoint not found: {self.checkpoint}')
        self.mirt=mirt; self.device=torch.device(device); ck=torch.load(self.checkpoint,map_location=self.device)
        arch=ck.get('actor_architecture')
        if arch != ACTOR_ARCHITECTURE: raise ValueError(f'RDPG-MIRT checkpoint actor architecture {arch!r} is not supported by {ACTOR_ARCHITECTURE!r} policy')
        if 'action_mean' not in ck or 'action_std' not in ck: raise KeyError('RDPG-MIRT checkpoint missing action normalization statistics')
        self.theta_cfg=dict(ck.get('theta_fit') or theta_cfg or {})
        self.normalizer=ActionNormalizer(torch.as_tensor(ck['action_mean']).float(), torch.as_tensor(ck['action_std']).float())
        cfg=(ck.get('training_config') or {}).get('model',{})
        self.actor=MIRTRecurrentActor(state_dim=int(cfg.get('state_dim',75)), hidden_dim=int(ck.get('hidden_dim',cfg.get('hidden_dim',128))), action_dim=int(cfg.get('action_dim',37))).to(self.device)
        self.actor.load_state_dict(ck['actor_state_dict']); self.actor.eval(); self.hidden=None
    def reset(self, student_id=None, seed=None, context=None):
        super().reset(student_id, seed, context or {}); self.hidden=self.actor.init_hidden(1,self.device)
    def select(self,candidate_item_ids,history_item_ids,history_responses,context):
        if self.hidden is None: self.hidden=self.actor.init_hidden(1,self.device)
        st=build_mirt_state(self.mirt,history_item_ids,history_responses,len(history_item_ids),int(context.get('max_steps',20)),self.theta_cfg,self.device)
        with torch.no_grad(): a,self.hidden=self.actor.forward_step(st,self.hidden)
        return nearest_item(a.squeeze(0),candidate_item_ids,self.mirt,self.normalizer,self.device)
