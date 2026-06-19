import torch
from models.mirt import MIRTModel, load_mirt_checkpoint, fit_student_theta, predict_with_theta

def test_synthetic_checkpoint_infers_dimension(tmp_path):
    path=tmp_path/'mirt.pt'
    torch.save({'state_dict':{'theta_emb.weight':torch.zeros(7,5),'disc_emb.weight':torch.ones(11,5),'diff_emb.weight':torch.zeros(11,1)}}, path)
    m=load_mirt_checkpoint(path)
    assert (m.n_students,m.n_items,m.n_dims)==(7,11,5)

def test_fit_student_theta_freezes_item_parameters_and_loss_drops():
    m=MIRTModel(2,4,3); m.disc_emb.weight.data=torch.tensor([[2.,0,0],[-2.,0,0],[0,2,0],[0,-2,0]]); m.diff_emb.weight.data.zero_()
    before=(m.disc_emb.weight.detach().clone(), m.diff_emb.weight.detach().clone())
    theta, losses=fit_student_theta(m,[0,1,2,3],[1,0,1,0],steps=40,lr=.1,theta_l2=0.001,return_losses=True)
    assert losses[-1] < losses[0]
    assert torch.allclose(before[0], m.disc_emb.weight)
    assert torch.allclose(before[1], m.diff_emb.weight)
    assert theta.shape == (3,)
