import torch
import pytest
from models.ncdm import OfficialNCDM
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch
from models.set_attention import InducedSetAttentionBlock
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork, RELATIVE_FEATURE_NAMES


def cache(k=5, items=12):
    q = torch.randint(0,2,(items,k)).float(); q[:,0]=1
    return NCDMItemFeatureCache(OfficialNCDM(1, items, k), q)


def batch(c, cands=None):
    cands = cands or [1,2,3,4]
    return pad_c3dqn_batch([{"history_item_ids":[0],"history_responses":[1.0],"candidate_item_ids":cands,"mastery":[.5]*c.knowledge_dim,"coverage":[0]*c.knowledge_dim,"coverage_count":[0]*c.knowledge_dim,"policy_step":0,"selected_item_id":cands[0]}], c, 5)


def test_set_modes_and_isab_parameters():
    c=cache()
    n0=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, candidate_set_encoder='none')
    assert len(n0.set_layers)==0
    nf=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, candidate_set_encoder='full_self_attention', num_set_layers=2)
    assert len(nf.set_layers)==2
    ni=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, candidate_set_encoder='isab', num_inducing_points=3)
    assert any(isinstance(m, InducedSetAttentionBlock) for m in ni.modules())
    ni2=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, candidate_set_encoder='isab', num_inducing_points=7)
    assert sum(p.numel() for p in ni2.parameters()) > sum(p.numel() for p in ni.parameters())


def test_relative_features_and_no_nan_for_zero_qmask():
    c=cache(); b=batch(c)
    net=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, use_relative_features=True)
    q,_=net(**{k:v for k,v in b.items() if k!='action_index'})
    assert torch.isfinite(q).all()
    assert net.relative_feature_names == RELATIVE_FEATURE_NAMES
    assert net.last_debug['relative_features'].shape[-1] == 5


def test_permutation_equivariance_and_padding_invariance():
    c=cache(); b=batch(c,[1,2,3,4])
    net=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, dropout=0.0, candidate_set_encoder='isab')
    net.eval()
    q,_=net(**{k:v for k,v in b.items() if k!='action_index'})
    bp=batch(c,[3,1,4,2]); qp,_=net(**{k:v for k,v in bp.items() if k!='action_index'})
    inv=[1,3,0,2]
    assert torch.allclose(q, qp[:,inv], atol=1e-5)
    bpad=batch(c,[1,2,3,4,5]); bpad['candidate_mask'][0,4]=False
    qpad,_=net(**{k:v for k,v in bpad.items() if k!='action_index'})
    assert torch.allclose(q, qpad[:,:4], atol=1e-5)


def test_full_attention_limit_and_chunked_equivalence():
    c=cache(items=20); b=batch(c, list(range(1,11)))
    bad=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, candidate_set_encoder='full_self_attention', full_attention_max_candidates=4)
    with pytest.raises(ValueError):
        bad(**{k:v for k,v in b.items() if k!='action_index'})
    net=SetConditionedNCDMQNetwork(c.knowledge_dim, d_model=16, n_heads=4, dropout=0.0, candidate_set_encoder='isab')
    net.eval()
    q,_=net(**{k:v for k,v in b.items() if k!='action_index'})
    qc,_=net.forward_chunked(chunk_size=3, **{k:v for k,v in b.items() if k!='action_index'})
    assert torch.allclose(q, qc, atol=1e-5)
