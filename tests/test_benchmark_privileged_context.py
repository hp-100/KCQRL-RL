import json

import pytest
import torch

from evaluation.benchmark import BenchmarkV2Evaluator
from evaluation.policies.base import BaseCATPolicy, PolicyMetadata
from evaluation.policies.ddpg_mirt_policy import DDPGMIRTPolicy
from evaluation.policies.oracle_policy import OneStepOraclePolicy
from evaluation.policies.random_policy import RandomMIRTPolicy, RandomPolicy
from evaluation.policies.rdpg_mirt_policy import RDPGMIRTPolicy
from models.mirt import MIRTModel
from models.mirt_actor import MIRTActor, ACTOR_ARCHITECTURE as DDPG_ARCH
from models.mirt_recurrent_actor import MIRTRecurrentActor, ACTOR_ARCHITECTURE as RDPG_ARCH

PRIVILEGED_KEYS = {
    "query_item_ids",
    "query_responses",
    "candidate_response_lookup",
    "query_nll_after_history",
    "predict_history",
}


class SpyPolicy(BaseCATPolicy):
    name = "Spy"
    metadata = PolicyMetadata(name=name)

    def __init__(self, name="Spy", metadata=None):
        self.name = name
        self.metadata = metadata or PolicyMetadata(name=name)
        self.contexts = []

    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)

    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        self.contexts.append(dict(context))
        return int(list(candidate_item_ids)[0])


def _cfg(tmp_path, policies=None, steps=(0, 1)):
    return {
        "benchmark": {
            "seeds": [3],
            "steps": list(steps),
            "max_students": 3,
            "output_dir": str(tmp_path),
            "policies": policies or ["Random"],
            "selection_horizon": max(steps) if steps else 1,
        },
        "assets": {"base_dir": "/missing", "q_matrix": "q.npy", "item_bank": "i.npy", "test_sequences": "t.csv", "ncdm_checkpoint": "n.pt"},
        "device": "cpu",
    }


def test_non_oracle_select_context_excludes_privileged_keys(tmp_path, monkeypatch):
    spy = SpyPolicy("Random")
    monkeypatch.setattr(BenchmarkV2Evaluator, "_policies", lambda self, *args, **kwargs: [spy])

    BenchmarkV2Evaluator(_cfg(tmp_path), debug=True).run()

    assert spy.contexts
    for ctx in spy.contexts:
        assert set(ctx) == {"policy_step", "selection_horizon"}
        assert PRIVILEGED_KEYS.isdisjoint(ctx)


def test_one_step_oracle_receives_required_privileged_context(tmp_path, monkeypatch):
    oracle_spy = SpyPolicy(
        "OneStepOracle",
        PolicyMetadata(name="OneStepOracle", uses_privileged_information=True, uses_query_labels=True),
    )
    monkeypatch.setattr(BenchmarkV2Evaluator, "_policies", lambda self, *args, **kwargs: [oracle_spy])

    BenchmarkV2Evaluator(_cfg(tmp_path), debug=True).run()

    assert oracle_spy.contexts
    required = {"query_item_ids", "candidate_response_lookup", "query_nll_after_history", "predict_history"}
    for ctx in oracle_spy.contexts:
        assert required.issubset(ctx)
        assert "query_responses" not in ctx


def test_ddpg_and_rdpg_mirt_select_contexts_are_safe(tmp_path, monkeypatch):
    ddpg = SpyPolicy("DDPG-MIRT", PolicyMetadata(name="DDPG-MIRT", uses_privileged_information=False, uses_query_labels=False))
    rdpg = SpyPolicy("RDPG-MIRT", PolicyMetadata(name="RDPG-MIRT", uses_privileged_information=False, uses_query_labels=False))
    monkeypatch.setattr(BenchmarkV2Evaluator, "_policies", lambda self, *args, **kwargs: [ddpg, rdpg])

    BenchmarkV2Evaluator(_cfg(tmp_path, policies=["DDPG-MIRT", "RDPG-MIRT"]), debug=True).run()

    for pol in (ddpg, rdpg):
        assert pol.contexts
        assert all(set(ctx) == {"policy_step", "selection_horizon"} for ctx in pol.contexts)


def test_policy_metadata_privilege_flags():
    ordinary = [RandomPolicy(), RandomMIRTPolicy()]
    for pol in ordinary:
        assert pol.metadata.uses_privileged_information is False
        assert pol.metadata.uses_query_labels is False
    assert OneStepOraclePolicy.metadata.uses_privileged_information is True
    assert OneStepOraclePolicy.metadata.uses_query_labels is True


def _ddpg_checkpoint(path, *, scope="full_mirt_item_bank", count=4):
    actor = MIRTActor(hidden_dim=8)
    torch.save({
        "action_mean": torch.zeros(37),
        "action_std": torch.ones(37),
        "action_normalizer_scope": scope,
        "normalizer_item_count": count,
        "theta_fit": {},
        "selection_horizon": 1,
        "warm_start_items": 1,
        "actor_architecture": DDPG_ARCH,
        "hidden_dim": 8,
        "actor_state_dict": actor.state_dict(),
    }, path)


def _rdpg_checkpoint(path, *, scope="full_mirt_item_bank", count=4):
    actor = MIRTRecurrentActor(hidden_dim=8)
    torch.save({
        "action_mean": torch.zeros(37),
        "action_std": torch.ones(37),
        "action_normalizer_scope": scope,
        "normalizer_item_count": count,
        "theta_fit": {},
        "selection_horizon": 1,
        "warm_start_items": 1,
        "actor_architecture": RDPG_ARCH,
        "hidden_dim": 8,
        "training_config": {"model": {"state_dim": 75, "hidden_dim": 8, "action_dim": 37}},
        "actor_state_dict": actor.state_dict(),
    }, path)


@pytest.mark.parametrize("policy_cls,ckpt_factory", [(DDPGMIRTPolicy, _ddpg_checkpoint), (RDPGMIRTPolicy, _rdpg_checkpoint)])
def test_mirt_rl_checkpoint_normalizer_scope_and_item_count_are_validated(tmp_path, policy_cls, ckpt_factory):
    mirt = MIRTModel(2, 4, 36)

    bad_scope = tmp_path / "bad_scope.pt"
    ckpt_factory(bad_scope, scope="candidate_subset", count=4)
    with pytest.raises(ValueError, match="action_normalizer_scope"):
        policy_cls(bad_scope, mirt, theta_cfg={})

    bad_count = tmp_path / "bad_count.pt"
    ckpt_factory(bad_count, scope="full_mirt_item_bank", count=3)
    with pytest.raises(ValueError, match="normalizer_item_count"):
        policy_cls(bad_count, mirt, theta_cfg={})

    missing_count = tmp_path / "missing_count.pt"
    ckpt_factory(missing_count, scope="full_mirt_item_bank", count=4)
    ck = torch.load(missing_count)
    ck.pop("normalizer_item_count")
    torch.save(ck, missing_count)
    with pytest.raises(KeyError, match="normalizer_item_count"):
        policy_cls(missing_count, mirt, theta_cfg={})


def test_benchmark_step_zero_result_is_unchanged_by_select_context_isolation(tmp_path):
    torch.manual_seed(123)
    cfg = _cfg(tmp_path / "run", policies=["Random"], steps=(0, 1))
    rows = BenchmarkV2Evaluator(cfg, debug=True).run()
    step0_before_selection = [r for r in rows if r["step"] == 0]

    torch.manual_seed(123)
    cfg2 = _cfg(tmp_path / "run2", policies=["Random"], steps=(0,))
    rows2 = BenchmarkV2Evaluator(cfg2, debug=True).run()
    step0_only = [r for r in rows2 if r["step"] == 0]

    for r in step0_before_selection + step0_only:
        r.pop("policy", None)
    assert step0_before_selection == step0_only
