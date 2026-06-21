"""Resume the Colab pipeline and auto-select the KC-level independent test file.

This wrapper removes the only remaining manual path choice after a runtime reconnect.
It searches for ``kc_level/test_question_window_sequences.csv`` under the durable
Drive root, deduplicates identical copies by SHA-256, and forwards the selected
path to ``scripts/colab_pipeline.py``.
"""
from __future__ import annotations

import argparse
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

    # Prefer the explicitly archived real-data copy when the files are identical.
    real_data = [path for path in candidates if "real_data" in path.parts]
    selected = real_data[0] if real_data else candidates[0]
    print("Identical duplicates detected; selected:", selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-select the KC-level test set and resume the durable pipeline"
    )
    parser.add_argument("--drive-root", required=True)
    args, passthrough = parser.parse_known_args()

    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive root does not exist; mount Google Drive first: {drive_root}"
        )

    test_sequences = _select_independent_test(drive_root)
    command = [
        sys.executable,
        "-u",
        "scripts/colab_pipeline.py",
        "--drive-root",
        str(drive_root),
        "--resume",
        "--test-sequences",
        str(test_sequences),
        *passthrough,
    ]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
