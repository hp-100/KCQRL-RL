import json
import yaml
import torch

from evaluation.benchmark import BenchmarkV2Evaluator
from evaluation.protocol import make_student_split


def test_duplicate_items_never_cross_support_query():
    sp, reason = make_student_split(
        "stu-dup",
        [1, 2, 3, 2, 4, 5, 1, 6, 7, 8, 9, 10],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        seed=7,
        valid_count=20,
        min_query_items=2,
    )
    assert reason is None
    assert sp.duplicate_interactions_removed == 2
    assert sp.original_interactions == 12
    assert sp.valid_interactions == 12
    assert set(sp.support_item_ids).isdisjoint(sp.query_item_ids)


def test_ddpg_policy_receives_l2_normalized_item_bank(tmp_path):
    cfg = {"benchmark": {"policies": ["DDPG"]}}
    evaluator = BenchmarkV2Evaluator(cfg, debug=True, ddpg_checkpoint=str(tmp_path / "missing.pt"))
    q = torch.eye(3).numpy()
    item_bank = torch.tensor([[3.0, 4.0], [0.0, 0.0], [5.0, 12.0]]).numpy()
    policy = evaluator._policies(q, item_bank, ncdm=None, synthetic=True)[0]
    norms = torch.linalg.vector_norm(policy.item_bank, dim=1)
    nonzero = torch.linalg.vector_norm(torch.tensor(item_bank), dim=1) > 0
    assert torch.allclose(norms[nonzero], torch.ones_like(norms[nonzero]), atol=1e-6)
    assert torch.all(norms[~nonzero] == 0)


def test_benchmark_max_students_not_capped_by_legacy_evaluation_limit(tmp_path, monkeypatch):
    import csv
    import numpy as np
    import evaluation.benchmark as benchmark_mod
    from evaluation.offline_eval import CATOfflineEvaluator

    q_path = tmp_path / "q.npy"
    ib_path = tmp_path / "item_bank.npy"
    seq_path = tmp_path / "seq.csv"
    ckpt_path = tmp_path / "ncdm.pt"
    np.save(q_path, np.eye(6, dtype=np.float32))
    np.save(ib_path, np.ones((6, 4), dtype=np.float32))
    ckpt_path.write_bytes(b"placeholder")
    with seq_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "item_ids", "responses"])
        writer.writeheader()
        for sid in range(80):
            writer.writerow({"student_id": str(sid), "item_ids": "0,1,2,3,4,5", "responses": "1,0,1,0,1,0"})

    def fail_legacy_load(self):
        raise AssertionError("benchmark_v2 must not call legacy load()")

    monkeypatch.setattr(CATOfflineEvaluator, "load", fail_legacy_load)
    monkeypatch.setattr(benchmark_mod, "safe_load_ncdm_checkpoint", lambda *args, **kwargs: None)
    cfg = {
        "device": "cpu",
        "evaluation": {"max_students": 50, "policies": ["Random"]},
        "benchmark": {"max_students": 80},
        "assets": {
            "base_dir": str(tmp_path),
            "q_matrix": q_path.name,
            "item_bank": ib_path.name,
            "test_sequences": seq_path.name,
            "ncdm_checkpoint": ckpt_path.name,
        },
    }
    _, _, seqs, _, synthetic = BenchmarkV2Evaluator(cfg, debug=False, max_students=80)._load_or_synthetic()
    assert synthetic is False
    assert len(seqs) == 80


def test_step_level_student_counts_and_policy_filter_run_config(tmp_path):
    cfg = {
        "benchmark": {
            "protocol": "benchmark_v2",
            "seeds": [11],
            "steps": [0, 25],
            "max_students": 4,
            "output_dir": str(tmp_path),
            "policies": ["Random", "MIRT-MFI"],
        },
        "assets": {"base_dir": "/missing", "q_matrix": "q.pt", "item_bank": "i.npy", "test_sequences": "t.csv", "ncdm_checkpoint": "n.pt"},
    }
    rows = BenchmarkV2Evaluator(cfg, debug=True, ddpg_checkpoint=str(tmp_path / "missing.pt"), policies=["Random"]).run()
    assert {r["policy"] for r in rows} == {"Random"}
    by_step = {r["step"]: r for r in rows}
    assert by_step[0]["evaluated_students"] == 4
    assert by_step[25]["evaluated_students"] == 0
    assert by_step[25]["incomplete_students"] == 4
    run_config = yaml.safe_load((tmp_path / "run_config.yaml").read_text())
    assert run_config["benchmark"]["policies"] == ["Random"]
    assert run_config["benchmark"]["steps"] == [0, 25]
    metadata = json.loads((tmp_path / "policy_metadata.json").read_text())
    assert list(metadata) == ["Random"]
