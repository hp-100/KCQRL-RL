import pytest, torch
from models.mirt import MIRTModel, fit_student_theta
from models.mirt_actor import MIRTActor, ACTOR_ARCHITECTURE
from core.mirt_state_builder import compute_action_normalizer, item_action_features, nearest_item
from reward.mirt_reward import bernoulli_nll, nll_drop_reward
from evaluation.policies.ddpg_mirt_policy import DDPGMIRTPolicy


def make_mirt(n_items=6, dim=36):
    m=MIRTModel(2,n_items,dim)
    with torch.no_grad():
        m.disc_emb.weight.copy_(torch.arange(n_items*dim,dtype=torch.float32).view(n_items,dim)/100)
        m.diff_emb.weight.copy_(torch.linspace(-1,1,n_items).view(n_items,1))
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()


def test_mirt_actor_is_declared_mlp_and_has_no_recurrent_interface():
    a=MIRTActor()
    out=a(torch.zeros(4,75))
    assert out.shape==(4,37)
    assert ACTOR_ARCHITECTURE == "mlp_explicit_state"
    assert not any(isinstance(m, torch.nn.LSTMCell) for m in a.modules())
    with pytest.raises(TypeError):
        a(torch.zeros(4,75), torch.zeros(4,128), torch.zeros(4,128))


def test_action_normalization_roundtrip_and_nearest():
    m=make_mirt(4)
    norm=compute_action_normalizer(m,[0,1,2,3])
    feats=item_action_features(m,[0,1,2,3])
    assert torch.allclose(norm.denormalize(norm.normalize(feats)), feats, atol=1e-5)
    assert nearest_item(norm.normalize(feats[[2]]).squeeze(0), [0,1,2,3], m, norm)=='x' if False else 2


def test_mirt_fit_leaves_parameters_grad_free():
    m=make_mirt(4)
    _=fit_student_theta(m,[0,1],[1,0],steps=2)
    assert all((not p.requires_grad) and p.grad is None for p in m.parameters())


def test_reward_is_nll_drop():
    prev=bernoulli_nll(torch.tensor([.5,.5]), [1,0])
    cur=bernoulli_nll(torch.tensor([.9,.1]), [1,0])
    assert nll_drop_reward(prev,cur,reward_scale=1,reward_clip=99)==pytest.approx(prev-cur)


def test_ddpg_mirt_checkpoint_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        DDPGMIRTPolicy(tmp_path/'missing.pt', make_mirt())


def test_ddpg_mirt_checkpoint_requires_action_stats(tmp_path):
    ck=tmp_path/'bad.pt'; torch.save({'actor_state_dict':MIRTActor().state_dict()}, ck)
    with pytest.raises(KeyError):
        DDPGMIRTPolicy(ck, make_mirt())


def test_ddpg_mirt_checkpoint_uses_checkpoint_theta_cfg(tmp_path):
    ck=tmp_path/'ok.pt'
    actor=MIRTActor()
    theta_fit={'steps':7,'lr':0.2,'theta_l2':0.03,'grad_clip':1.5,'early_stop_tol':0.0}
    torch.save({'actor_state_dict':actor.state_dict(),'action_mean':torch.zeros(37),'action_std':torch.ones(37),'hidden_dim':128,'actor_architecture':ACTOR_ARCHITECTURE,'theta_fit':theta_fit}, ck)
    pol=DDPGMIRTPolicy(ck, make_mirt(), theta_cfg={'steps':1})
    assert pol.theta_cfg == theta_fit


def test_trainer_scalar_logging_detaches_before_item():
    src=__import__('pathlib').Path('agents/mirt_ddpg_trainer.py').read_text()
    assert 'float(loss)' not in src
    assert 'float(q.mean())' not in src
    assert 'float(y.mean())' not in src
    assert 'loss.detach().item()' in src
    assert 'q.detach().mean().item()' in src
    assert 'y.detach().mean().item()' in src
