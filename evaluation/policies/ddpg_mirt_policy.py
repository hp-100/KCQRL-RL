from __future__ import annotations
from pathlib import Path
import torch
from .base import BaseCATPolicy, PolicyMetadata
from core.mirt_state_builder import ActionNormalizer, build_mirt_state, nearest_item
from models.mirt_actor import MIRTActor, ACTOR_ARCHITECTURE

class DDPGMIRTPolicy(BaseCATPolicy):
    name='DDPG-MIRT'
    metadata=PolicyMetadata(name=name, implementation='trained_ddpg_mirt_checkpoint', selection_model='mirt', evaluator_model='mirt', uses_query_labels=False)
    def __init__(self, checkpoint, mirt, theta_cfg=None, device='cpu'):
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.exists(): raise FileNotFoundError(f'DDPG-MIRT checkpoint not found: {self.checkpoint}')
        self.mirt=mirt; self.device=torch.device(device)
        ck=torch.load(self.checkpoint,map_location=self.device)
        if 'action_mean' not in ck or 'action_std' not in ck: raise KeyError('DDPG-MIRT checkpoint missing action normalization statistics')
        ck_theta=dict(ck.get('theta_fit') or ck.get('theta_cfg') or {})
        req_theta=dict(theta_cfg or {})
        if ck_theta and req_theta and ck_theta != req_theta:
            print(f"WARNING: DDPG-MIRT theta_fit config differs from checkpoint: checkpoint={ck_theta}, requested={req_theta}; using checkpoint config")
        self.theta_cfg=ck_theta or req_theta
        arch=ck.get('actor_architecture', ACTOR_ARCHITECTURE)
        if arch != ACTOR_ARCHITECTURE:
            raise ValueError(f'DDPG-MIRT checkpoint actor architecture {arch!r} is not supported by current {ACTOR_ARCHITECTURE!r} policy')
        self.normalizer=ActionNormalizer(torch.as_tensor(ck['action_mean']).float(), torch.as_tensor(ck['action_std']).float())
        self.actor=MIRTActor(hidden_dim=int(ck.get('hidden_dim',128))).to(self.device)
        self.actor.load_state_dict(ck.get('actor_state_dict', ck)); self.actor.eval()
    def select(self,candidate_item_ids,history_item_ids,history_responses,context):
        st=build_mirt_state(self.mirt,history_item_ids,history_responses,len(history_item_ids),int(context.get('max_steps',20)),self.theta_cfg,self.device)
        with torch.no_grad(): a=self.actor(st)
        return nearest_item(a.squeeze(0),candidate_item_ids,self.mirt,self.normalizer,self.device)
