import torch, numpy as np, pytest
from models.mirt_recurrent_actor import MIRTRecurrentActor
from models.mirt_recurrent_critic import MIRTRecurrentCritic
from agents.sequence_replay_buffer import SequenceReplayBuffer


def test_actor_sequence_and_step_history_effect():
    torch.manual_seed(0); a=MIRTRecurrentActor(); x=torch.randn(2,4,75); y,h=a.forward_sequence(x)
    assert y.shape==(2,4,37) and h[0].shape==(1,2,128)
    h0=a.init_hidden(1,'cpu'); _,h1=a.forward_step(x[0,0],h0); y2,_=a.forward_step(x[0,1],h1)
    _,h1b=a.forward_step(x[0,0]*0,h0); y2b,_=a.forward_step(x[0,1],h1b)
    assert not torch.allclose(y2,y2b)
    yr,_=a.forward_step(x[0,1],a.init_hidden(1,'cpu'))
    assert not torch.allclose(y2,yr)


def test_sequence_replay_padding_no_hidden_and_no_cross_episode():
    rb=SequenceReplayBuffer(seed=0)
    rb.add_episode(np.ones((2,75))*1, np.ones((2,37)), [1,2], np.ones((2,75))*2, [0,1])
    rb.add_episode(np.ones((8,75))*7, np.ones((8,37))*7, np.ones(8), np.ones((8,75))*8, np.zeros(8))
    assert not hasattr(rb.episodes[0], 'hidden')
    b=rb.sample(4,3,5)
    assert b['states'].shape==(4,8,75) and b['valid_mask'].shape==(4,8,1)
    for seq,mask in zip(b['states'], b['valid_mask']):
        vals=set(seq[mask.squeeze(-1).bool(),0].tolist())
        assert vals in ({1.0},{7.0})


def test_burnin_masked_loss_and_lstm_gradients():
    torch.manual_seed(1); burn,unroll=3,5; B=2
    actor=MIRTRecurrentActor(); critic=MIRTRecurrentCritic(); opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-3)
    s=torch.randn(B,burn+unroll,75); replay_a=torch.randn(B,burn+unroll,37); mask=torch.ones(B,unroll,1); mask[0,-1]=0
    with torch.no_grad(): _,ah=actor.forward_sequence(s[:,:burn]); _,ch=critic.forward_sequence(s[:,:burn],replay_a[:,:burn])
    pa,_=actor.forward_sequence(s[:,burn:],ah); q,_=critic.forward_sequence(s[:,burn:],pa,ch)
    per=(q-torch.ones_like(q)).pow(2); loss=(per*mask).sum()/mask.sum().clamp_min(1)
    opt.zero_grad(); loss.backward()
    assert actor.lstm.weight_ih_l0.grad.abs().sum()>0
    per2=per.clone(); per2[0,-1]=9999
    assert torch.isclose((per*mask).sum()/mask.sum(), (per2*mask).sum()/mask.sum())


def test_target_burnin_shapes():
    ta=MIRTRecurrentActor(); tc=MIRTRecurrentCritic(); B=2; burn=3; unroll=5
    bs=torch.randn(B,burn,75); ba=torch.randn(B,burn,37); ns=torch.randn(B,unroll,75)
    with torch.no_grad():
        _,tah=ta.forward_sequence(bs); _,tch=tc.forward_sequence(bs,ba); na,_=ta.forward_sequence(ns,tah); q,_=tc.forward_sequence(ns,na,tch)
    assert na.shape==(B,unroll,37) and q.shape==(B,unroll,1)


def test_checkpoint_architecture_rejection(tmp_path):
    from evaluation.policies.rdpg_mirt_policy import RDPGMIRTPolicy
    ck=tmp_path/'mlp.pt'; torch.save({'actor_architecture':'mlp_explicit_state','action_mean':torch.zeros(37),'action_std':torch.ones(37),'actor_state_dict':{}},ck)
    with pytest.raises(ValueError): RDPGMIRTPolicy(ck, object())
