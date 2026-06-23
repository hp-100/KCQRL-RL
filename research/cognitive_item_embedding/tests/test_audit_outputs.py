import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "research" / "cognitive_item_embedding" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import inspect_dataset
import sample_items

class TinySplit(list):
    pass

def synthetic_ds(n=120):
    return {"train": TinySplit({"id": f"tr-{i}", "question": f"题目 $x_{i}$ 中文", "answer": f"{i}", "formal": f"Expr({i})"} for i in range(n))}

def test_synthetic_report_goes_to_tmp_and_not_repo(tmp_path):
    stats, _ = inspect_dataset.audit(synthetic_ds())
    assert stats["total_items"] == 120
    assert stats["status"] == "NOT RUN ON REAL CONIC10K"
    out = tmp_path / "DATA_AUDIT_REPORT.md"
    inspect_dataset.write_report(stats, out)
    text = out.read_text(encoding="utf-8")
    assert "STATUS: NOT RUN ON REAL CONIC10K" in text
    assert "120" not in text
    repo_text = inspect_dataset.REPO_REPORT.read_text(encoding="utf-8")
    assert "STATUS: NOT RUN ON REAL CONIC10K" in repo_text
    assert "120" not in repo_text

def test_refuses_repo_report_without_explicit_real_generation(tmp_path):
    stats, _ = inspect_dataset.audit(synthetic_ds())
    try:
        inspect_dataset.write_report(stats, inspect_dataset.REPO_REPORT)
    except SystemExit as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
