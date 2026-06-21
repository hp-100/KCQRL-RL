"""Run Random-NCDM, Base-C3DQN-NCDM and NCDM-DDPG fairly.

The script auto-discovers the real 36D assets under the configured Drive root,
uses the paired benchmark_v2 support/query protocol, and evaluates every
prediction-reward Base-C3DQN checkpoint found for seeds 42/43/44.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.ncdm_ddpg_benchmark import NCDMDDPGBenchmarkEvaluator


def _parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_paths(value: str) -> list[Path]:
    values = [Path(part.strip()).expanduser() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one checkpoint path")
    return values


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def discover_test_sequences(data_root: Path) -> Path:
    preferred = [
        data_root / "real_data/XES3G5M/kc_level/test_question_window_sequences.csv",
        data_root / "XES3G5M/kc_level/test_question_window_sequences.csv",
        data_root / "kc_level/test_question_window_sequences.csv",
        data_root / "kc_level/test.csv",
    ]
    found = _first_existing(preferred)
    if found is None:
        raise FileNotFoundError(
            "No supported independent test sequence file was found under "
            f"{data_root}"
        )
    return found


def _epoch_number(path: Path) -> int:
    match = re.search(r"epoch[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def discover_ddpg_checkpoint(drive_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    data_root = drive_root / "data/XES3G5M"
    exact_names = [
        "ddpg_enhanced_36d_actor_best.pt",
        "ddpg_actor_best.pt",
        "ddpg_actor_final.pt",
    ]
    search_roots = [
        data_root / "metadata",
        drive_root / "outputs",
        drive_root,
    ]
    for name in exact_names:
        exact_matches = []
        for root in search_roots:
            if root.exists():
                exact_matches.extend(root.rglob(name))
        unique = sorted({path.resolve() for path in exact_matches})
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            raise RuntimeError(
                "Multiple DDPG checkpoints matched the same preferred name; "
                "pass --ddpg-checkpoint explicitly:\n"
                + "\n".join(str(path) for path in unique)
            )

    epoch_matches = []
    for root in search_roots:
        if root.exists():
            epoch_matches.extend(root.rglob("ddpg_enhanced_36d_actor_epoch*.pt"))
    unique_epoch_matches = sorted({path.resolve() for path in epoch_matches})
    if unique_epoch_matches:
        return max(unique_epoch_matches, key=lambda path: (_epoch_number(path), str(path)))

    raise FileNotFoundError(
        "No NCDM-DDPG actor checkpoint was found. Pass --ddpg-checkpoint explicitly."
    )


def discover_c3dqn_checkpoints(
    drive_root: Path,
    explicit: list[Path] | None,
) -> list[Path]:
    if explicit:
        resolved = [path.expanduser().resolve() for path in explicit]
        missing = [path for path in resolved if not path.exists()]
        if missing:
            raise FileNotFoundError("\n".join(str(path) for path in missing))
        return resolved

    reward_root = drive_root / "outputs/base_c3dqn_reward_ablation/prediction"
    checkpoints = sorted(reward_root.glob("seed_*/best_checkpoint.pt"))
    if checkpoints:
        return [path.resolve() for path in checkpoints]

    fallback = _first_existing(
        [
            drive_root / "outputs/ncdm_c3dqn/best_checkpoint.pt",
            drive_root / "outputs/c3dqn_topk64/base/seed_42/best_checkpoint.pt",
        ]
    )
    if fallback is not None:
        return [fallback]
    raise FileNotFoundError(
        "No Base-C3DQN checkpoint was found. Pass --c3dqn-checkpoints explicitly."
    )


def _relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_config(
    *,
    data_root: Path,
    test_sequences: Path,
    output_dir: Path,
    eval_seeds: list[int],
    steps: list[int],
    max_students: int,
) -> dict:
    return {
        "seed": 42,
        "device": "auto",
        "assets": {
            "base_dir": str(data_root),
            "q_matrix": "metadata/q_matrix_multihot_36_expert.pt",
            "item_bank": "metadata/item_bank_128d.npy",
            "ncdm_checkpoint": "metadata/ncdm_model_36d_expert_best.pt",
            "test_sequences": _relative_to(test_sequences, data_root),
        },
        "evaluation": {
            "policies": ["Random-NCDM", "C3DQN-NCDM", "NCDM-DDPG"],
        },
        "benchmark": {
            "protocol": "benchmark_v2",
            "track": "ncdm_native",
            "policies": ["Random-NCDM", "C3DQN-NCDM", "NCDM-DDPG"],
            "query_ratio": 0.2,
            "min_query_items": 5,
            "warm_start_items": 1,
            "selection_horizon": max(steps),
            "steps": steps,
            "seeds": eval_seeds,
            "max_students": max_students,
            "candidate_size": None,
            "save_predictions": True,
            "save_traces": True,
            "output_dir": str(output_dir),
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Random/Base-C3DQN/NCDM-DDPG benchmark"
    )
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--ddpg-checkpoint", type=Path, default=None)
    parser.add_argument("--c3dqn-checkpoints", type=_parse_paths, default=None)
    parser.add_argument("--eval-seeds", type=_parse_ints, default=[101, 102, 103])
    parser.add_argument("--steps", type=_parse_ints, default=[0, 1, 3, 5, 10])
    parser.add_argument("--max-students", type=int, default=200)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root).expanduser().resolve()
    data_root = drive_root / "data/XES3G5M"
    required_assets = [
        data_root / "metadata/q_matrix_multihot_36_expert.pt",
        data_root / "metadata/item_bank_128d.npy",
        data_root / "metadata/ncdm_model_36d_expert_best.pt",
    ]
    missing = [path for path in required_assets if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))

    test_sequences = discover_test_sequences(data_root)
    ddpg_checkpoint = discover_ddpg_checkpoint(drive_root, args.ddpg_checkpoint)
    c3dqn_checkpoints = discover_c3dqn_checkpoints(
        drive_root,
        args.c3dqn_checkpoints,
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else drive_root / "outputs/ncdm_ddpg_unified_benchmark"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("NCDM-DDPG actor:", ddpg_checkpoint)
    print("independent test sequences:", test_sequences)
    print("Base-C3DQN checkpoints:")
    for checkpoint in c3dqn_checkpoints:
        print("  ", checkpoint)

    combined_rows: list[dict] = []
    for checkpoint in c3dqn_checkpoints:
        run_name = checkpoint.parent.name
        run_dir = output_root / f"c3dqn_{run_name}"
        aggregate_path = run_dir / "aggregate.csv"
        if aggregate_path.exists() and not args.force:
            print("skip completed benchmark:", run_dir)
        else:
            config = build_config(
                data_root=data_root,
                test_sequences=test_sequences,
                output_dir=run_dir,
                eval_seeds=args.eval_seeds,
                steps=args.steps,
                max_students=args.max_students,
            )
            evaluator = NCDMDDPGBenchmarkEvaluator(
                config,
                ddpg_checkpoint=str(ddpg_checkpoint),
                c3dqn_ncdm_checkpoint=str(checkpoint),
                track="ncdm_native",
                seeds=args.eval_seeds,
                max_students=args.max_students,
                steps=args.steps,
                output_dir=run_dir,
                policies=["Random-NCDM", "C3DQN-NCDM", "NCDM-DDPG"],
            )
            evaluator.run()

        for row in _read_csv(aggregate_path):
            combined_rows.append(
                {
                    "c3dqn_training_run": run_name,
                    "c3dqn_checkpoint": str(checkpoint),
                    "ddpg_checkpoint": str(ddpg_checkpoint),
                    **row,
                }
            )

    combined_path = output_root / "combined_aggregate.csv"
    _write_csv(combined_path, combined_rows)
    state_path = output_root / "COMPLETED.json"
    state_path.write_text(
        json.dumps(
            {
                "ddpg_checkpoint": str(ddpg_checkpoint),
                "c3dqn_checkpoints": [str(path) for path in c3dqn_checkpoints],
                "test_sequences": str(test_sequences),
                "eval_seeds": args.eval_seeds,
                "steps": args.steps,
                "max_students": args.max_students,
                "combined_aggregate": str(combined_path),
            },
            indent=2,
        )
    )
    print("unified benchmark complete")
    print("combined results:", combined_path)
    print("state:", state_path)


if __name__ == "__main__":
    main()
