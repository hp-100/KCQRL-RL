import numpy as np
import pytest
import torch
from evaluation.protocol import make_student_split, canonical_theta_fit
from agents.sequence_replay_buffer import SequenceReplayBuffer
from core.mirt_state_builder import compute_action_normalizer
from models.mirt import MIRTModel
from models.mirt_actor import MIRTActor, ACTOR_ARCHITECTURE
from evaluation.policies.ddpg_mirt_policy import DDPGMIRTPolicy


def tiny_mirt(n=9, dim=36):
    m=MIRTModel(2,n,dim)
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()


def test_warm_start_candidate_protocol_and_manifest_fields():
    sp,_=make_student_split('s', list(range(12)), [0,1]*6, seed=0, valid_count=12, min_query_items=3)
    d=sp.to_dict()
    assert sp.warm_start_item == sp.support_item_ids[0]
    assert sp.warm_start_response == sp.support_responses[0]
    assert sp.warm_start_item not in d['candidate_items']
    assert d['query_items'] == sp.query_item_ids


def test_policy_step_zero_with_history_len_one(tmp_path, monkeypatch):
    m=tiny_mirt(); ck=tmp_path/'ck.pt'; theta=canonical_theta_fit({'steps':1})
    torch.save({'actor_state_dict':MIRTActor().state_dict(),'action_mean':torch.zeros(37),'action_std':torch.ones(37),'hidden_dim':128,'actor_architecture':ACTOR_ARCHITECTURE,'theta_fit':theta,'selection_horizon':4,'warm_start_items':1,'action_normalizer_scope':'full_mirt_item_bank','normalizer_item_count':9}, ck)
    seen={}
    def fake_build(mirt, history_item_ids, history_responses, step, max_steps, theta_cfg, device):
        seen['history_len']=len(history_item_ids); seen['step']=step; seen['horizon']=max_steps
        return torch.zeros(75)
    monkeypatch.setattr('evaluation.policies.ddpg_mirt_policy.build_mirt_state', fake_build)
    pol=DDPGMIRTPolicy(ck,m,theta_cfg=theta)
    pol.select([1,2],[0],[1.0],{'policy_step':0,'selection_horizon':4})
    assert seen == {'history_len':1,'step':0,'horizon':4}


def test_checkpoint_horizon_mismatch_errors(tmp_path):
    m=tiny_mirt(); ck=tmp_path/'ck.pt'; theta=canonical_theta_fit({'steps':1})
    torch.save({'actor_state_dict':MIRTActor().state_dict(),'action_mean':torch.zeros(37),'action_std':torch.ones(37),'hidden_dim':128,'actor_architecture':ACTOR_ARCHITECTURE,'theta_fit':theta,'selection_horizon':4,'warm_start_items':1,'action_normalizer_scope':'full_mirt_item_bank','normalizer_item_count':9}, ck)
    pol=DDPGMIRTPolicy(ck,m,theta_cfg=theta)
    with pytest.raises(ValueError, match='selection_horizon mismatch'):
        pol.select([1],[0],[1.0],{'policy_step':0,'selection_horizon':5})


def test_full_mirt_normalizer_uses_all_items():
    m=tiny_mirt(n=11)
    norm=compute_action_normalizer(m, range(m.n_items))
    feats=torch.cat([m.disc_emb.weight, m.diff_emb.weight], dim=1)
    assert torch.allclose(norm.mean, feats.mean(0), atol=1e-6)


def test_short_episode_replay_rejects_all_burnin():
    rb=SequenceReplayBuffer(seed=0)
    rb.add_episode(np.ones((2,75)), np.ones((2,37)), [1,2], np.ones((2,75)), [0,1])
    with pytest.raises(ValueError, match='at least one valid unroll'):
        rb.sample(1,2,5)


def test_sequence_replay_has_valid_unroll_mask():
    rb=SequenceReplayBuffer(seed=0)
    rb.add_episode(np.ones((3,75)), np.ones((3,37)), [1,2,3], np.ones((3,75)), [0,0,1])
    b=rb.sample(2,2,5)
    assert b['valid_mask'][:,2:].sum().item() > 0
