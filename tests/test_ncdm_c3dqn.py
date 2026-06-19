from pathlib import Path
import torch, pytest
from models.ncdm import OfficialNCDM
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch, build_global_feature
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork, NEG_INF_Q
from reward.ncdm_diagnostic_reward import mastery_entropy, compute_ncdm_diagnostic_reward, NCDMDiagnosticRewardConfig
from agents.ncdm_c3dqn_trainer import C3DQNReplayBuffer, C3DQNTransition, compute_double_dqn_loss, validate_c3dqn_checkpoint_metadata, NCDMC3DQNTrainer


def make_cache(k=36, items=8):
    torch.manual_seed(0); n=OfficialNCDM(1, items, k); q=torch.randint(0,2,(items,k)).float(); q[:,0]=1
    return n, q, NCDMItemFeatureCache(n,q)

def sample(k=36):
    return {"history_item_ids":[0,1],"history_responses":[1.0,0.0],"candidate_item_ids":[2,3,4],"mastery":[0.5]*k,"coverage":[0.0]*k,"policy_step":2,"selected_item_id":3}

def test_feature_dims_k36_and_dynamic_non36():
    _,_,c=make_cache(36); assert c.dims.history_feature_dim==75 and c.dims.candidate_feature_dim==73 and c.dims.global_feature_dim==73
    _,_,c2=make_cache(5); assert c2.dims.history_feature_dim==13 and c2.dims.candidate_feature_dim==11 and c2.dims.global_feature_dim==11

def test_response_raw_no_embedding_or_semantic():
    _,_,c=make_cache(4); h=c.history([0],[1.0],5); assert h.shape[-1]==11 and h[0,-2].item()==1.0
    net=CandidateConditionedNCDMQNetwork(4, d_model=16, n_heads=4, num_history_layers=1)
    names="\n".join(n for n,_ in net.named_parameters()).lower(); assert "response" not in names and "semantic" not in names

def test_forward_shape_mask_context_and_debug_shapes():
    _,_,c=make_cache(6); batch=pad_c3dqn_batch([sample(6), {**sample(6),"candidate_item_ids":[2],"selected_item_id":2}], c, 5)
    net=CandidateConditionedNCDMQNetwork(6, d_model=16, n_heads=4, num_history_layers=1, dropout=0.0)
    q_values, attn=net(**{k:v for k,v in batch.items() if k!="action_index"}, return_attention=True)
    assert q_values.shape==(2,3); assert q_values[1,1].item()==pytest.approx(NEG_INF_Q); assert q_values.argmax(1).tolist()[1]==0
    assert net.last_debug["mastered"].shape==(2,3,6) and net.last_debug["weakness"].shape==(2,3,6) and net.last_debug["difficulty_gap"].shape==(2,3,6)
    assert not torch.allclose(net.last_debug["candidate_context"][0,0], net.last_debug["candidate_context"][0,1]) or q_values[0,0] != q_values[0,1]
    adv=net.last_debug["advantage"]; mean=(adv*batch["candidate_mask"].float()).sum(1,keepdim=True)/batch["candidate_mask"].sum(1,keepdim=True)
    assert torch.allclose(net.last_debug["masked_mean_advantage"], mean)

def test_double_dqn_uses_online_selection_target_value():
    _,_,c=make_cache(4); b=pad_c3dqn_batch([sample(4)], c, 5); nb=pad_c3dqn_batch([{**sample(4),"candidate_item_ids":[2,3],"selected_item_id":2}], c, 5)
    online=CandidateConditionedNCDMQNetwork(4,16,4,1); target=CandidateConditionedNCDMQNetwork(4,16,4,1)
    loss, stats=compute_double_dqn_loss(online,target,b,nb,torch.tensor([1.0]),torch.tensor([0.0]),0.9)
    assert loss.isfinite() and "target_q_mean" in stats and "next_action_mean" in stats

def test_reward_entropy_and_components():
    assert mastery_entropy(torch.full((10,),0.5)) > mastery_entropy(torch.full((10,),0.99))
    r=compute_ncdm_diagnostic_reward(0.5,0.4,torch.full((3,),0.5),torch.full((3,),0.9),torch.tensor([1.,0.,1.]),torch.tensor([0.,1.,0.]),NCDMDiagnosticRewardConfig())
    assert r.prediction_gain == pytest.approx(1.0) and r.coverage_gain == pytest.approx(1.0) and torch.isfinite(torch.tensor(r.total))

def test_replay_compact_selected_membership_and_no_dense_features():
    rb=C3DQNReplayBuffer(2); t=C3DQNTransition([0],[1.0],[1,2],[.5]*3,[0]*3,1,2,.1,{},[0,2],[1,1],[1],[.5]*3,[.1]*3,2,False); rb.push(t)
    assert "candidate_features" not in rb.state_dict()[0]
    with pytest.raises(ValueError): rb.push(C3DQNTransition([0],[1],[1],[.5]*3,[0]*3,1,2,.1,{},[],[],[],[.5]*3,[0]*3,2,True))

def test_checkpoint_mismatch_fails_and_policy_privilege():
    meta={"knowledge_dim":3,"history_feature_dim":9,"candidate_feature_dim":7,"global_feature_dim":7,"selection_horizon":5,"warm_start_items":1,"alpha_fit":{},"candidate_pool_config":{}}
    with pytest.raises(ValueError): validate_c3dqn_checkpoint_metadata(meta,{**meta,"knowledge_dim":4})

def test_tiny_synthetic_smoke(tmp_path):
    _,_,c=make_cache(4,8); online=CandidateConditionedNCDMQNetwork(4,16,4,1); target=CandidateConditionedNCDMQNetwork(4,16,4,1)
    m=NCDMC3DQNTrainer(online,target,c,5,tmp_path); metrics=m.run_synthetic_smoke_epoch()
    assert (tmp_path/"best_checkpoint.pt").exists() and (tmp_path/"training_history.csv").exists(); assert metrics["td_loss"]==metrics["td_loss"]
