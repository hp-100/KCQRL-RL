import pytest, torch
from scripts.train_ncdm_c3dqn import build_q_network_from_config
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork, RELATIVE_FEATURE_NAMES
from agents.ncdm_c3dqn_trainer import NCDMCandidatePrefilter


def batch(k=5,c=16):
    h=torch.rand(2,3,2*k+3); hm=torch.ones(2,3,dtype=torch.bool)
    cf=torch.zeros(2,c,2*k+1); cf[:,:,:k]=(torch.rand(2,c,k)>0.5).float(); cf[:,:,0]=1; cf[:,:,k:2*k]=torch.rand(2,c,k)*cf[:,:,:k]; cf[:,:,2*k:]=torch.rand(2,c,1)
    cm=torch.ones(2,c,dtype=torch.bool); g=torch.rand(2,2*k+1); cov=torch.arange(k).float().repeat(2,1)
    return h,hm,cf,cm,g,cov

def test_build_q_network_from_config_base_set_unknown():
    assert isinstance(build_q_network_from_config({'architecture':'base_c3dqn'},4), CandidateConditionedNCDMQNetwork)
    assert isinstance(build_q_network_from_config({'architecture':'set_c3dqn','candidate_set_encoder':'none'},4), SetConditionedNCDMQNetwork)
    with pytest.raises(ValueError): build_q_network_from_config({'architecture':'bogus'},4)

def test_relative_features_values_and_coverage_changes():
    k=3; m=SetConditionedNCDMQNetwork(k,d_model=12,n_heads=3,dropout=0,candidate_set_encoder='none')
    c=torch.zeros(1,2,2*k+1); c[0,0,:k]=torch.tensor([1,1,0]); c[0,1,:k]=0; c[0,:,k:2*k]=0.2
    g=torch.zeros(1,2*k+1); g[0,:k]=torch.tensor([0.1,0.5,0.9])
    r0=m._relative_features(c,g,torch.tensor([[0.,1.,0.]]))
    assert m.relative_feature_names == RELATIVE_FEATURE_NAMES and m.relative_feature_dim == 5
    torch.testing.assert_close(r0[0,0,0], torch.tensor(0.5))
    torch.testing.assert_close(r0[0,0,1], torch.tensor(0.5))
    assert torch.isfinite(r0).all()
    r1=m._relative_features(c,g,torch.zeros(1,k))
    assert not torch.allclose(r0, r1)

def test_use_relative_features_false_no_debug_relative():
    h,hm,c,cm,g,cov=batch(4,8); m=SetConditionedNCDMQNetwork(4,d_model=16,n_heads=2,dropout=0,candidate_set_encoder='none',use_relative_features=False,debug_mode=True)
    q,_=m(h,hm,c,cm,g,coverage_count=cov); assert q.shape==(2,8); assert m.relative_feature_dim==0; assert m.last_debug['relative_features'] is None

def test_full_attention_limit_and_isab_unlimited():
    h,hm,c,cm,g,cov=batch(4,129)
    m=SetConditionedNCDMQNetwork(4,d_model=16,n_heads=2,dropout=0,candidate_set_encoder='full_self_attention',full_attention_max_candidates=128)
    with pytest.raises(ValueError, match='full candidate self-attention exceeds configured candidate limit'): m(h,hm,c,cm,g,coverage_count=cov)
    h,hm,c,cm,g,cov=batch(4,128); m(h,hm,c,cm,g,coverage_count=cov)
    SetConditionedNCDMQNetwork(4,d_model=16,n_heads=2,dropout=0,candidate_set_encoder='isab',num_inducing_points=4)(*batch(4,129)[:5], coverage_count=batch(4,129)[5])

def test_forward_chunked_matches_and_argmax_stable():
    h,hm,c,cm,g,cov=batch(4,40); m=SetConditionedNCDMQNetwork(4,d_model=16,n_heads=2,dropout=0,candidate_set_encoder='isab',num_inducing_points=4).eval()
    q,_=m(h,hm,c,cm,g,coverage_count=cov)
    for cs in [8,16,32,64,128]:
        qc,_=m.forward_chunked(h,hm,c,cm,g,coverage_count=cov,chunk_size=cs)
        torch.testing.assert_close(q,qc); assert torch.equal(q.argmax(1), qc.argmax(1))

def test_prefilter_disabled_enabled_deterministic_and_diversity_history():
    q=torch.eye(6); ids=list(range(6)); mastery=torch.zeros(6); coverage=torch.tensor([1.,1.,1.,0.,0.,0.])
    pf=NCDMCandidatePrefilter(enabled=False,top_k=2); out,_=pf.filter(ids,q_matrix=q,mastery=mastery,coverage_count=coverage); assert out==ids
    pf=NCDMCandidatePrefilter(enabled=True,top_k=4,diversity_quota=2); a,_=pf.filter(ids,q_matrix=q,mastery=mastery,coverage_count=coverage); b,_=pf.filter(ids,q_matrix=q,mastery=mastery,coverage_count=coverage)
    assert a==b and len(a)==4 and len(set(a))==4 and any(x in a for x in [3,4,5])
