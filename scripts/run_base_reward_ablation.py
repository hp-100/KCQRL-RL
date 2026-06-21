"""Run the Base-C3DQN reward ablation on durable Google Drive outputs.

The default run trains two new reward variants (prediction-only and
prediction-plus-small-coverage) for seeds 42/43/44. Existing checkpoints are
skipped, so the command is safe to rerun after a Colab reconnect.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from scripts.make_base_reward_ablation_configs import (
    build_reward_ablation_configs,
    write_reward_ablation_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_names(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one experiment")
    return values


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("command:", " ".join(command))
    print("log:", log_path)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"reward-ablation training failed with return_code={return_code}; "
            f"see {log_path}"
        )


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _best_history_row(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    valid = [row for row in rows if row.get("validation_query_nll") not in (None, "")]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row["validation_query_nll"]))


def _write_summary(output_root: Path, experiments: list[str], seeds: list[int]) -> Path:
    summary_rows: list[dict[str, str | int | float]] = []
    for experiment in experiments:
        for seed in seeds:
            run_root = output_root / experiment / f"seed_{seed}"
            best = _best_history_row(_read_history(run_root / "training_history.csv"))
            if best is None:
                continue
            summary_rows.append(
                {
                    "experiment": experiment,
                    "training_seed": seed,
                    "best_epoch": int(float(best["epoch"])),
                    "validation_query_auc": float(best["validation_query_auc"]),
                    "validation_query_nll": float(best["validation_query_nll"]),
                    "validation_query_brier": float(best["validation_query_brier"]),
                    "validation_concept_coverage": float(best["validation_concept_coverage"]),
                    "mean_total_reward": float(best["mean_total_reward"]),
                    "item_exposure_gini": float(best["item_exposure_gini"]),
                    "checkpoint": str(run_root / "best_checkpoint.pt"),
                }
            )

    summary_path = output_root / "validation_summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable Base-C3DQN reward ablations"
    )
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--seeds", type=_parse_ints, default=[42, 43, 44])
    parser.add_argument(
        "--experiments",
        type=_parse_names,
        default=["prediction", "prediction_coverage"],
    )
    parser.add_argument("--max-students", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--selection-horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root).expanduser().resolve()
    data_root = drive_root / "data/XES3G5M"
    q_matrix = data_root / "metadata/q_matrix_multihot_36_expert.pt"
    ncdm_checkpoint = data_root / "metadata/ncdm_model_36d_expert_best.pt"
    train_sequences = data_root / "kc_level/train_valid_sequences.csv"
    for path in (q_matrix, ncdm_checkpoint, train_sequences):
        if not path.exists():
            raise FileNotFoundError(path)

    config_root = drive_root / "configs/base_c3dqn_reward_ablation"
    output_root = drive_root / "outputs/base_c3dqn_reward_ablation"
    log_root = drive_root / "pipeline/logs/base_c3dqn_reward_ablation"

    for seed in args.seeds:
        configs = build_reward_ablation_configs(
            q_matrix=str(q_matrix),
            ncdm_checkpoint=str(ncdm_checkpoint),
            train_valid_sequences=str(train_sequences),
            output_root=str(output_root),
            max_students=args.max_students,
            epochs=args.epochs,
            selection_horizon=args.selection_horizon,
            seed=seed,
            device=args.device,
            use_amp=not args.no_amp,
            top_k=args.top_k,
            experiments=args.experiments,
        )
        config_paths = write_reward_ablation_configs(configs, config_root, seed)
        for experiment in args.experiments:
            run_root = output_root / experiment / f"seed_{seed}"
            checkpoint = run_root / "best_checkpoint.pt"
            if checkpoint.exists() and not args.force:
                print(f"skip {experiment} seed {seed}: {checkpoint}")
                continue
            _run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/train_ncdm_c3dqn.py",
                    "--config",
                    str(config_paths[experiment]),
                ],
                log_root / f"{experiment}_seed_{seed}.log",
            )
            if not checkpoint.exists():
                raise RuntimeError(f"training completed without checkpoint: {checkpoint}")

    summary = _write_summary(output_root, args.experiments, args.seeds)
    state = {
        "experiments": args.experiments,
        "seeds": args.seeds,
        "summary": str(summary),
    }
    state_path = output_root / "COMPLETED.json"
    state_path.write_text(json.dumps(state, indent=2))
    print("reward ablation complete")
    print("summary:", summary)
    print("state:", state_path)


if __name__ == "__main__":
    main()
