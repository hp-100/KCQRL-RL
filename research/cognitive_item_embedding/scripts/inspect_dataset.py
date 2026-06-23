#!/usr/bin/env python3
"""Audit Conic10K data without modifying source text fields."""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_ID = "WenyangHui/Conic10K"
SOURCE_URL = "https://huggingface.co/datasets/WenyangHui/Conic10K"
LICENSE = "MIT"
EXPECTED_REAL_MIN_ITEMS = 10000
REPO_REPORT = Path(__file__).resolve().parents[1] / "docs" / "DATA_AUDIT_REPORT.md"

TEXT_KEYS = ("text", "question", "problem", "input")
ANSWER_KEYS = ("answer", "output", "target", "solution")


def load_hf_dataset(dataset_id: str = DATASET_ID):
    from datasets import load_dataset
    return load_dataset(dataset_id)


def iter_rows(ds: Any):
    if hasattr(ds, "keys"):
        for split in ds.keys():
            for idx, row in enumerate(ds[split]):
                yield split, idx, dict(row)
    else:
        for idx, row in enumerate(ds):
            yield "all", idx, dict(row)


def pick(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def audit(ds: Any, dataset_id: str = DATASET_ID) -> tuple[dict, list[dict]]:
    rows, split_counts, missing, lengths, seen, dupes = [], Counter(), Counter(), [], {}, []
    for split, idx, row in iter_rows(ds):
        text = pick(row, TEXT_KEYS)
        answer = pick(row, ANSWER_KEYS)
        formal = str(row.get("formal", row.get("program", row.get("representation", ""))) or "")
        item_id = str(row.get("id", f"{split}-{idx}"))
        split_counts[split] += 1
        for name, val in (("text", text), ("answer", answer), ("formal", formal)):
            if not val:
                missing[name] += 1
        lengths.append(len(text))
        key = (text, answer, formal)
        if key in seen:
            dupes.append({"first_id": seen[key], "duplicate_id": item_id, "split": split})
        else:
            seen[key] = item_id
        rows.append({"id": item_id, "split": split, "text": text, "answer": answer, "formal": formal, "raw": row})
    total = len(rows)
    sl = sorted(lengths)
    def pct(p: float) -> int:
        if not sl: return 0
        return sl[min(len(sl)-1, int(round((len(sl)-1)*p)))]
    stats = {
        "status": "RUN ON REAL CONIC10K" if total >= EXPECTED_REAL_MIN_ITEMS else "NOT RUN ON REAL CONIC10K",
        "dataset_id": dataset_id, "source_url": SOURCE_URL, "license": LICENSE,
        "total_items": total, "split_counts": dict(split_counts),
        "missing_counts": dict(missing), "duplicate_count": len(dupes),
        "text_length": {"min": min(sl) if sl else 0, "p50": pct(.5), "p95": pct(.95), "max": max(sl) if sl else 0},
    }
    return stats, rows


def write_report(stats: dict, output: Path, allow_repo_report: bool = False) -> None:
    if output.resolve() == REPO_REPORT.resolve() and not allow_repo_report:
        raise SystemExit("Refusing to overwrite formal DATA_AUDIT_REPORT.md without --allow-repo-report")
    if stats["total_items"] < EXPECTED_REAL_MIN_ITEMS:
        body = "# Conic10K Data Audit Report\n\nSTATUS: NOT RUN ON REAL CONIC10K\n\nReal Conic10K audit did not complete; synthetic/test statistics are intentionally omitted.\n"
    else:
        body = f"""# Conic10K Data Audit Report

STATUS: RUN ON REAL CONIC10K

- Source: {stats['source_url']}
- Dataset/version: {stats['dataset_id']}
- License: {stats['license']}
- Total items: {stats['total_items']}
- Split counts: {json.dumps(stats['split_counts'], ensure_ascii=False)}
- Missing counts: {json.dumps(stats['missing_counts'], ensure_ascii=False)}
- Duplicate count: {stats['duplicate_count']}
- Text length: {json.dumps(stats['text_length'], ensure_ascii=False)}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/conic10k_audit"))
    ap.add_argument("--report", type=Path)
    ap.add_argument("--allow-repo-report", action="store_true")
    args = ap.parse_args()
    ds = load_hf_dataset(args.dataset_id)
    stats, rows = audit(ds, args.dataset_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = args.report or (args.output_dir / "DATA_AUDIT_REPORT.md")
    write_report(stats, report, args.allow_repo_report)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
