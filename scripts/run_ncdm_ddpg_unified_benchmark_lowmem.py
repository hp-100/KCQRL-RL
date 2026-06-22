"""Low-memory staged runner for the unified NCDM-DDPG benchmark.

Runs one evaluation seed and one policy group at a time, writes each stage to
Drive, releases Python/GPU memory between stages, and finally combines results.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.ncdm_ddpg_benchmark import NCDMDDPGBenchmarkEvaluator
from scripts.run_ncdm_ddpg_unified_benchmark import (
    _parse_ints,
    _parse_paths,
    build_config,
    discover_c3dqn_checkpoints,
    discover_ddpg_checkpoint,
    discover_test_sequences,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_selection_horizon(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu")
    try:
        metadata = dict(payload.get("metadata") or {})
        if "selection_horizon" not in metadata:
            raise ValueError(
                f"C3DQN checkpoint missing selection_horizon metadata: {checkpoint}"
            )
        horizon = int(metadata["selection_horizon"])
    finally:
        del payload
        release_memory()
    if horizon <= 0:
        raise ValueError(
            f"invalid selection_horizon={horizon} in checkpoint: {checkpoint}"
        )
    return horizon


def run_stage(
    *,
    config: dict,
    ddpg_checkpoint: Path,
    c3dqn_checkpoint: Path,
    eval_seed: int,
    max_students: int,
    steps: list[int],
    output_dir: Path,
    policies: list[str],
    force: bool,
) -> list[dict[str, str]]:
    result_path = output_dir / "per_seed.csv"
    if result_path.exists() and not force:
        print("skip completed stage:", output_dir)
        return read_csv(result_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = NCDMDDPGBenchmarkEvaluator(
        config,
        ddpg_checkpoint=str(ddpg_checkpoint),
        c3dqn_ncdm_checkpoint=str(c3dqn_checkpoint),
        track="ncdm_native",
        seeds=[eval_seed],
        max_students=max_students,
        steps=steps,
        output_dir=output_dir,
        policies=policies,
    )
    try:
        evaluator.run()
    finally:
        del evaluator
        release_memory()
    return read_csv(result_path)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys = sorted({(row["training_run"], row["policy"], int(row["step"])) for row in rows})
    aggregated: list[dict] = []
    for training_run, policy, step in keys:
        group = [
            row
            for row in rows
            if row["training_run"] == training_run
            and row["policy"] == policy
            and int(row["step"]) == step
        ]
        output = {
            "training_run": training_run,
            "policy": policy,
            "step": step,
        }
        metric_names = [
            name
            for name in group[0]
            if name not in {"training_run", "policy", "seed", "step"}
        ]
        for name in metric_names:
            values = []
            for row in group:
                try:
                    value = float(row[name])
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    values.append(value)
            output[name + "_mean"] = (
                sum(values) / len(values) if values else float("nan")
            )
            output[name + "_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregated.append(output)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory staged Random/Base-C3DQN/NCDM-DDPG benchmark"
    )
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--ddpg-checkpoint", type=Path, default=None)
    parser.add_argument("--c3dqn-checkpoints", type=_parse_paths, default=None)
    parser.add_argument("--eval-seeds", type=_parse_ints, default=[101, 102, 103])
    parser.add_argument("--steps", type=_parse_ints, default=[0, 1, 3, 5, 10])
    parser.add_argument("--max-students", type=int, default=200)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-diverse-ddpg",
        action="store_true",
        help="also evaluate the frozen-actor conservative exposure reranker",
    )
    parser.add_argument(
        "--skip-c3dqn",
        action="store_true",
        help="run only Random/original DDPG/diverse DDPG shared stages",
    )
    parser.add_argument("--ddpg-top-k", type=int, default=16)
    parser.add_argument("--ddpg-exposure-weight", type=float, default=0.005)
    parser.add_argument("--ddpg-novelty-weight", type=float, default=0.0)
    parser.add_argument("--ddpg-coverage-weight", type=float, default=0.0)
    parser.add_argument("--ddpg-distance-margin-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ddpg-distance-mode",
        choices=["euclidean", "block_mse"],
        default="euclidean",
    )
    parser.add_argument("--ddpg-q-distance-weight", type=float, default=1.0)
    parser.add_argument("--ddpg-difficulty-distance-weight", type=float, default=1.0)
    parser.add_argument("--ddpg-discrimination-distance-weight", type=float, default=1.0)
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
    checkpoint_horizons = {
        checkpoint: checkpoint_selection_horizon(checkpoint)
        for checkpoint in c3dqn_checkpoints
    }
    if not args.skip_c3dqn:
        for checkpoint, horizon in checkpoint_horizons.items():
            if max(args.steps) > horizon:
                raise ValueError(
                    f"requested step {max(args.steps)} exceeds checkpoint "
                    f"selection_horizon={horizon}: {checkpoint}"
                )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else drive_root / "outputs/ncdm_ddpg_unified_benchmark_lowmem"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    diversity_config = {
        "top_k": args.ddpg_top_k,
        "exposure_weight": args.ddpg_exposure_weight,
        "novelty_weight": args.ddpg_novelty_weight,
        "coverage_weight": args.ddpg_coverage_weight,
        "distance_margin_ratio": args.ddpg_distance_margin_ratio,
        "distance_mode": args.ddpg_distance_mode,
        "q_distance_weight": args.ddpg_q_distance_weight,
        "difficulty_distance_weight": args.ddpg_difficulty_distance_weight,
        "discrimination_distance_weight": args.ddpg_discrimination_distance_weight,
    }

    print("NCDM-DDPG actor:", ddpg_checkpoint)
    print("independent test sequences:", test_sequences)
    print("Base-C3DQN checkpoints:")
    for checkpoint in c3dqn_checkpoints:
        print("  ", checkpoint, "selection_horizon=", checkpoint_horizons[checkpoint])
    if args.include_diverse_ddpg:
        print("NCDM-DDPG-Diverse config:", diversity_config)

    all_rows: list[dict] = []

    baseline_checkpoint = c3dqn_checkpoints[0]
    shared_group = (
        "shared_baselines_with_diverse"
        if args.include_diverse_ddpg
        else "shared_baselines"
    )
    shared_policies = ["Random-NCDM", "NCDM-DDPG"]
    if args.include_diverse_ddpg:
        shared_policies.append("NCDM-DDPG-Diverse")

    for eval_seed in args.eval_seeds:
        stage_dir = output_root / shared_group / f"eval_seed_{eval_seed}"
        config = build_config(
            data_root=data_root,
            test_sequences=test_sequences,
            output_dir=stage_dir,
            eval_seeds=[eval_seed],
            steps=args.steps,
            max_students=args.max_students,
        )
        config["benchmark"]["selection_horizon"] = max(args.steps)
        config["benchmark"]["save_predictions"] = False
        config["benchmark"]["save_traces"] = False
        config["benchmark"]["ddpg_diversity"] = diversity_config
        rows = run_stage(
            config=config,
            ddpg_checkpoint=ddpg_checkpoint,
            c3dqn_checkpoint=baseline_checkpoint,
            eval_seed=eval_seed,
            max_students=args.max_students,
            steps=args.steps,
            output_dir=stage_dir,
            policies=shared_policies,
            force=args.force,
        )
        for row in rows:
            all_rows.append({"training_run": "shared", **row})

    if not args.skip_c3dqn:
        for checkpoint in c3dqn_checkpoints:
            training_run = checkpoint.parent.name
            checkpoint_horizon = checkpoint_horizons[checkpoint]
            for eval_seed in args.eval_seeds:
                stage_dir = (
                    output_root
                    / f"c3dqn_{training_run}"
                    / f"eval_seed_{eval_seed}"
                )
                config = build_config(
                    data_root=data_root,
                    test_sequences=test_sequences,
                    output_dir=stage_dir,
                    eval_seeds=[eval_seed],
                    steps=args.steps,
                    max_students=args.max_students,
                )
                config["benchmark"]["selection_horizon"] = checkpoint_horizon
                config["benchmark"]["save_predictions"] = False
                config["benchmark"]["save_traces"] = False
                rows = run_stage(
                    config=config,
                    ddpg_checkpoint=ddpg_checkpoint,
                    c3dqn_checkpoint=checkpoint,
                    eval_seed=eval_seed,
                    max_students=args.max_students,
                    steps=args.steps,
                    output_dir=stage_dir,
                    policies=["C3DQN-NCDM"],
                    force=args.force,
                )
                for row in rows:
                    all_rows.append({"training_run": training_run, **row})

    per_seed_path = output_root / "combined_per_seed.csv"
    aggregate_path = output_root / "combined_aggregate.csv"
    write_csv(per_seed_path, all_rows)
    write_csv(aggregate_path, aggregate_rows(all_rows))

    state_path = output_root / "COMPLETED.json"
    state_path.write_text(
        json.dumps(
            {
                "ddpg_checkpoint": str(ddpg_checkpoint),
                "c3dqn_checkpoints": [str(path) for path in c3dqn_checkpoints],
                "c3dqn_selection_horizons": {
                    str(path): checkpoint_horizons[path]
                    for path in c3dqn_checkpoints
                },
                "test_sequences": str(test_sequences),
                "eval_seeds": args.eval_seeds,
                "steps": args.steps,
                "max_students": args.max_students,
                "include_diverse_ddpg": args.include_diverse_ddpg,
                "skip_c3dqn": args.skip_c3dqn,
                "ddpg_diversity": diversity_config,
                "combined_per_seed": str(per_seed_path),
                "combined_aggregate": str(aggregate_path),
            },
            indent=2,
        )
    )
    print("low-memory unified benchmark complete")
    print("combined per-seed results:", per_seed_path)
    print("combined aggregate results:", aggregate_path)
    print("state:", state_path)


if __name__ == "__main__":
    main()
