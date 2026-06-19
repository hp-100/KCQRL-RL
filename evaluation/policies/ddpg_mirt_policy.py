from __future__ import annotations
from pathlib import Path
import torch
from .base import BaseCATPolicy, PolicyMetadata
from core.mirt_state_builder import ActionNormalizer, build_mirt_state, nearest_item
from evaluation.protocol import assert_theta_fit_equal, canonical_theta_fit
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
        ck_theta=canonical_theta_fit(ck.get('theta_fit') or ck.get('theta_cfg') or {})
        req_theta=canonical_theta_fit(theta_cfg or {})
        assert_theta_fit_equal(ck_theta, req_theta, label_a='DDPG-MIRT checkpoint', label_b='benchmark')
        self.theta_cfg=ck_theta

        if 'selection_horizon' not in ck: raise KeyError(f'{self.name} checkpoint missing selection_horizon')
        self.selection_horizon=int(ck.get('selection_horizon'))
        if ck.get('warm_start_items') != 1: raise ValueError('DDPG-MIRT checkpoint must declare warm_start_items=1')
        arch=ck.get('actor_architecture', ACTOR_ARCHITECTURE)
        if arch != ACTOR_ARCHITECTURE:
            raise ValueError(f'DDPG-MIRT checkpoint actor architecture {arch!r} is not supported by current {ACTOR_ARCHITECTURE!r} policy')
        self.normalizer=ActionNormalizer(torch.as_tensor(ck['action_mean']).float(), torch.as_tensor(ck['action_std']).float())
        self.actor=MIRTActor(hidden_dim=int(ck.get('hidden_dim',128))).to(self.device)
        self.actor.load_state_dict(ck.get('actor_state_dict', ck)); self.actor.eval()
    def select(self,candidate_item_ids,history_item_ids,history_responses,context):
        policy_step=int(context['policy_step']); selection_horizon=int(context['selection_horizon'])
        if selection_horizon != self.selection_horizon: raise ValueError(f'DDPG-MIRT selection_horizon mismatch: checkpoint={self.selection_horizon}, benchmark={selection_horizon}')
        st=build_mirt_state(self.mirt,history_item_ids,history_responses,policy_step,selection_horizon,self.theta_cfg,self.device)
        with torch.no_grad(): a=self.actor(st)
        return nearest_item(a.squeeze(0),candidate_item_ids,self.mirt,self.normalizer,self.device)
