"""Resume the Colab pipeline with an automatically selected compact test set.

The original XES3G5M test CSV can be very large.  Loading the complete file before
slicing to ``max_students`` may exhaust Colab host RAM and the operating system then
kills the benchmark process with return code -9.  This wrapper therefore:

1. selects the KC-level independent test file;
2. verifies duplicate copies by SHA-256;
3. streams a deterministic first-N-student subset to durable Drive storage; and
4. forwards that compact file to ``scripts/colab_pipeline.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_independent_test(drive_root: Path) -> Path:
    data_root = drive_root / "data/XES3G5M"
    candidates = sorted(
        path.resolve()
        for path in data_root.rglob("test_question_window_sequences.csv")
        if path.parent.name == "kc_level"
    )
    if not candidates:
        raise FileNotFoundError(
            "No KC-level independent test file found under "
            f"{data_root}. Expected kc_level/test_question_window_sequences.csv"
        )
    if len(candidates) == 1:
        return candidates[0]

    hashes = {path: _sha256(path) for path in candidates}
    unique_hashes = set(hashes.values())
    print("KC-level independent test candidates:")
    for path, digest in hashes.items():
        print(f" - {path}  sha256={digest}")

    if len(unique_hashes) != 1:
        details = "\n".join(f"{path}: {digest}" for path, digest in hashes.items())
        raise ValueError(
            "Multiple non-identical KC-level test files were found; refusing to "
            "choose silently.\n" + details
        )

    real_data = [path for path in candidates if "real_data" in path.parts]
    selected = real_data[0] if real_data else candidates[0]
    print("Identical duplicates detected; selected:", selected)
    return selected


def _student_id(row: dict[str, str], fallback: int) -> str:
    return str(row.get("student_id") or row.get("user_id") or fallback)


def _prepare_compact_subset(
    source: Path,
    drive_root: Path,
    max_students: int,
) -> Path:
    if max_students <= 0:
        raise ValueError("max_students must be positive")

    source_digest = _sha256(source)[:16]
    output = (
        drive_root
        / "pipeline/cache"
        / f"test_question_window_sequences_first{max_students}_{source_digest}.csv"
    )
    if output.exists() and output.stat().st_size > 0:
        print("Compact independent test subset already exists:", output)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")

    with source.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        fieldnames = list(reader.fieldnames)
        wide_format = any(
            name in fieldnames
            for name in (
                "item_ids",
                "exer_ids",
                "questions",
                "question_ids",
            )
        )

        with temporary.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            if wide_format:
                written = 0
                for row in reader:
                    writer.writerow(row)
                    written += 1
                    if written >= max_students:
                        break
                student_count = written
            else:
                selected_ids: list[str] = []
                selected_set: set[str] = set()
                buffered_rows: list[dict[str, str]] = []
                for index, row in enumerate(reader):
                    sid = _student_id(row, index)
                    if sid not in selected_set and len(selected_ids) < max_students:
                        selected_ids.append(sid)
                        selected_set.add(sid)
                    if sid in selected_set:
                        buffered_rows.append(row)
                writer.writerows(buffered_rows)
                student_count = len(selected_ids)

    if student_count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"No student rows were written from {source}")
    temporary.replace(output)
    print(
        "Prepared compact independent test subset:",
        output,
        f"students={student_count}",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-select and compact the KC-level test set, then resume"
    )
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--max-students", type=int, default=200)
    args, passthrough = parser.parse_known_args()

    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive root does not exist; mount Google Drive first: {drive_root}"
        )

    source_test = _select_independent_test(drive_root)
    compact_test = _prepare_compact_subset(
        source_test,
        drive_root,
        args.max_students,
    )
    command = [
        sys.executable,
        "-u",
        "scripts/colab_pipeline.py",
        "--drive-root",
        str(drive_root),
        "--resume",
        "--test-sequences",
        str(compact_test),
        "--max-students",
        str(args.max_students),
        *passthrough,
    ]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
