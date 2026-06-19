import pytest
from evaluation.policies.ddpg_policy import DDPGPolicy

def test_ddpg_missing_checkpoint_errors_without_debug(tmp_path):
    with pytest.raises(FileNotFoundError):
        DDPGPolicy(tmp_path/'no.pt')
