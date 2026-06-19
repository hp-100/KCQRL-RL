import torch, pytest
from models.mirt import load_mirt_checkpoint

def test_checkpoint_direct_state_dict_supported(tmp_path):
    p=tmp_path/'direct.pt'; torch.save({'theta_emb.weight':torch.zeros(2,36),'disc_emb.weight':torch.zeros(9,36),'diff_emb.weight':torch.zeros(9,1)},p)
    assert load_mirt_checkpoint(p).n_dims == 36

def test_checkpoint_validates_required_tensors(tmp_path):
    p=tmp_path/'bad.pt'; torch.save({'state_dict':{'theta_emb.weight':torch.zeros(2,3)}},p)
    with pytest.raises(KeyError): load_mirt_checkpoint(p)
