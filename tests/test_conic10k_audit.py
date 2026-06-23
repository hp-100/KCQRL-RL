import csv, json
from pathlib import Path

from conic10k.audit import REQUIRED_FIELDS, REVIEW_COLUMNS, audit_rows, load_dataset, write_outputs


def _write_fixture(root: Path):
    root.mkdir()
    rows = [
        {"text":"A  B","process":"This is a sufficiently long rationale.","answer_expressions":"1","fact_expressions":"F","query_expressions":"Q","fact_spans":"[]","query_spans":"[]"},
        {"text":"A B","process":"short","answer_expressions":"","fact_expressions":"F","query_expressions":"Q","fact_spans":"[]","query_spans":"[]"},
        {"text":"C �","process":"Another sufficiently long rationale.","answer_expressions":"2","fact_expressions":"F","query_expressions":"Q","fact_spans":"[]","query_spans":"[]"},
    ]
    with (root / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_audit_uses_real_conic10k_fields(tmp_path):
    data = tmp_path / "data"
    _write_fixture(data)
    rows = load_dataset(data)
    report = audit_rows(rows)
    assert report["total_items"] == 3
    assert report["split_counts"] == {"train": 3}
    assert all(report["required_fields_present"][field] for field in REQUIRED_FIELDS)
    assert report["quality"]["empty_answers"] == 1
    assert report["quality"]["short_process_lt_20"] == 1
    assert report["quality"]["replacement_character_items"] == 1
    assert report["duplicates"]["exact_text_duplicate_groups"] == 0
    assert report["duplicates"]["whitespace_normalized_duplicate_groups"] == 1


def test_outputs_include_review_columns_and_no_raw_data(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    _write_fixture(data)
    report = write_outputs(load_dataset(data), out, sample_size=2, seed=20260623)
    assert report["total_items"] == 3
    csv_path = out / "conic10k_review_sample_100.csv"
    with csv_path.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    for col in ["item_id", "split", "text", "process", "answer_expressions", "fact_expressions", "query_expressions", "text_length", "process_length", "has_complete_process", *REVIEW_COLUMNS]:
        assert col in header
    assert (out / "conic10k_audit_report.json").exists()
    assert (out / "conic10k_review_sample_100.jsonl").exists()
    assert (out / "conic10k_review_sample_100.html").exists()
