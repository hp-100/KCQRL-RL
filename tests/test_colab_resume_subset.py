from __future__ import annotations

import csv
from pathlib import Path

from scripts.resume_colab_auto_test import _prepare_compact_subset


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_compact_subset_limits_wide_sequence_rows(tmp_path: Path) -> None:
    source = tmp_path / "data/XES3G5M/real_data/XES3G5M/kc_level/test_question_window_sequences.csv"
    rows = [
        {
            "student_id": str(index),
            "item_ids": "1,2,3",
            "responses": "1,0,1",
        }
        for index in range(5)
    ]
    _write_rows(source, ["student_id", "item_ids", "responses"], rows)

    output = _prepare_compact_subset(source, tmp_path, 2)

    with output.open(newline="") as file:
        compact = list(csv.DictReader(file))
    assert [row["student_id"] for row in compact] == ["0", "1"]


def test_prepare_compact_subset_keeps_all_rows_for_first_long_format_students(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data/XES3G5M/real_data/XES3G5M/kc_level/test_question_window_sequences.csv"
    rows = [
        {"student_id": "a", "item_id": "1", "response": "1"},
        {"student_id": "a", "item_id": "2", "response": "0"},
        {"student_id": "b", "item_id": "3", "response": "1"},
        {"student_id": "c", "item_id": "4", "response": "0"},
        {"student_id": "b", "item_id": "5", "response": "1"},
    ]
    _write_rows(source, ["student_id", "item_id", "response"], rows)

    output = _prepare_compact_subset(source, tmp_path, 2)

    with output.open(newline="") as file:
        compact = list(csv.DictReader(file))
    assert [(row["student_id"], row["item_id"]) for row in compact] == [
        ("a", "1"),
        ("a", "2"),
        ("b", "3"),
        ("b", "5"),
    ]
