import pytest, torch
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork
from models.ncdm_candidate_prefilter import NCDMCandidatePrefilter
from agents.ncdm_c3dqn_trainer import build_checkpoint_metadata, build_set_checkpoint_metadata, load_c3dqn_checkpoint, load_set_c3dqn_checkpoint, compute_double_dqn_loss

class TinyNCDM(torch.nn.Module):
    def __init__(self,n=20,k=5): super().__init__(); self.k_difficulty=torch.nn.Embedding(n,k); self.e_discrimination=torch.nn.Embedding(n,1)

def batch(b=2,c=64,k=5):
    torch.manual_seed(0); h=torch.randn(b,3,2*k+3); hm=torch.ones(b,3,dtype=torch.bool); cf=torch.rand(b,c,2*k+1); cf[:,:,:k]=(cf[:,:,:k]>.5).float(); cm=torch.ones(b,c,dtype=torch.bool);
    if b>1: cm[1,-3:]=False
    g=torch.rand(b,2*k+1); cov=torch.randint(0,3,(b,k)).float(); return h,hm,cf,cm,g,cov

def test_set_none_forward_and_relative_no_nan():
    h,hm,c,cm,g,cov=batch(c=8); c[0,0,:5]=0
    net=SetConditionedNCDMQNetwork(5,d_model=16,n_heads=4,num_history_layers=1,candidate_set_encoder="none",dropout=0)
    q,_=net(h,hm,c,cm,g,cov); assert q.shape==(2,8); assert torch.isfinite(q[cm]).all(); assert q[~cm].max()<-1e8
    assert not torch.isnan(net.last_debug["relative_features"]).any()

def test_full_attention_rejects_large_candidate_pool():
    h,hm,c,cm,g,cov=batch(c=9); net=SetConditionedNCDMQNetwork(5,16,4,1,0,candidate_set_encoder="full_self_attention",full_attention_max_candidates=8)
    with pytest.raises(ValueError): net(h,hm,c,cm,g,cov)

def test_isab_padding_invariance_and_permutation_equivariance():
    h,hm,c,cm,g,cov=batch(b=1,c=12); net=SetConditionedNCDMQNetwork(5,16,4,1,0,candidate_set_encoder="isab",num_inducing_points=4).eval()
    q,_=net(h,hm,c,cm,g,cov)
    pad=torch.zeros(1,5,c.shape[-1]); cp=torch.cat([c,pad],1); cmp=torch.cat([cm,torch.zeros(1,5,dtype=torch.bool)],1)
    qp,_=net(h,hm,cp,cmp,g,cov); torch.testing.assert_close(qp[:,:12],q,atol=1e-5,rtol=1e-5); assert qp.argmax(1).item()==q.argmax(1).item()
    perm=torch.randperm(12); q2,_=net(h,hm,c[:,perm],cm[:,perm],g,cov); torch.testing.assert_close(q2,q[:,perm],atol=1e-5,rtol=1e-5)

def test_chunked_matches_full_for_chunk_sizes():
    h,hm,c,cm,g,cov=batch(b=2,c=40); net=SetConditionedNCDMQNetwork(5,16,4,1,0,candidate_set_encoder="isab",num_inducing_points=4).eval()
    q,_=net(h,hm,c,cm,g,cov)
    for cs in [16,32,64,128]:
        qc,_=net.forward_chunked(h,hm,c,cm,g,cov,chunk_size=cs); torch.testing.assert_close(qc,q,atol=1e-6,rtol=1e-6); assert torch.equal(qc.argmax(1),q.argmax(1))

def test_prefilter_topk_deterministic_unique_and_zero_qmask():
    q=torch.eye(5).repeat(4,1); q[0]=0; pf=NCDMCandidatePrefilter(q,{"prefilter_enabled":True,"prefilter_top_k":6})
    out1=pf.select(list(range(20)),torch.zeros(5),torch.zeros(5)); out2=pf.select(list(range(20)),torch.zeros(5),torch.zeros(5))
    assert out1.candidate_ids==out2.candidate_ids; assert len(out1.candidate_ids)<=6; assert len(set(out1.candidate_ids))==len(out1.candidate_ids); assert set(out1.candidate_ids)<=set(range(20)); assert all(torch.isfinite(torch.tensor(out1.scores)))

def test_checkpoint_architecture_isolation(tmp_path):
    ncdm=TinyNCDM(); q=torch.eye(5).repeat(4,1); base=CandidateConditionedNCDMQNetwork(5,16,4,1,0); setnet=SetConditionedNCDMQNetwork(5,16,4,1,0,candidate_set_encoder="none")
    bm=build_checkpoint_metadata(knowledge_dim=5,selection_horizon=3,warm_start_items=1,alpha_fit={},reward_config={},model_config={"d_model":16,"n_heads":4,"num_history_layers":1,"dropout":0},candidate_pool_config={},ncdm_item_count=20,q_matrix_item_count=20,training_seed=0,validation_metrics={},epoch=1)
    sm=build_set_checkpoint_metadata(knowledge_dim=5,selection_horizon=3,warm_start_items=1,alpha_fit={},reward_config={},model_config={"d_model":16,"n_heads":4,"num_history_layers":1,"dropout":0,"candidate_set_encoder":"none"},candidate_pool_config={},ncdm_item_count=20,q_matrix_item_count=20)
    bp=tmp_path/'b.pt'; sp=tmp_path/'s.pt'; torch.save({"model_state_dict":base.state_dict(),"metadata":bm},bp); torch.save({"model_state_dict":setnet.state_dict(),"metadata":sm},sp)
    load_c3dqn_checkpoint(bp,ncdm=ncdm,q_matrix=q)
    with pytest.raises(ValueError): load_c3dqn_checkpoint(sp,ncdm=ncdm,q_matrix=q)
    with pytest.raises(ValueError): load_set_c3dqn_checkpoint(bp,ncdm=ncdm,q_matrix=q)
    load_set_c3dqn_checkpoint(sp,ncdm=ncdm,q_matrix=q)

def test_double_dqn_terminal_next_q_zero():
    h,hm,c,cm,g,cov=batch(b=2,c=4); b={"history_features":h,"history_mask":hm,"candidate_features":c,"candidate_mask":cm,"global_features":g,"coverage_count":cov,"action_index":torch.tensor([0,1])}
    net=CandidateConditionedNCDMQNetwork(5,16,4,1,0); loss,stats=compute_double_dqn_loss(net,net,b,None,torch.ones(2),torch.ones(2,dtype=torch.bool),0.99)
    assert loss.isfinite(); assert stats["next_q_mean"]==0.0
