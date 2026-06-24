"""One-command, resumable Colab pipeline for the C3DQN-NCDM study.

The script assumes Google Drive is already mounted and the repository has already
been cloned or updated. All durable outputs, logs, configs and state are written
under ``--drive-root`` so a Colab runtime restart does not lose progress.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.ncdm import OfficialNCDM, load_q_matrix, safe_load_ncdm_checkpoint
from scripts.make_c3dqn_topk64_configs import build_paired_configs


DEFAULT_TRAINING_SEEDS = (42, 43, 44)
DEFAULT_EVALUATION_SEEDS = (101, 102, 103)
DEFAULT_STEPS = (0, 1, 3, 5, 10)


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tasks": {}, "created_at": _utc_timestamp()}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"pipeline state is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"pipeline state must be an object: {path}")
    loaded.setdefault("tasks", {})
    return loaded


def _mark_task(
    state: dict[str, Any],
    state_path: Path,
    task_name: str,
    status: str,
    **details: Any,
) -> None:
    state.setdefault("tasks", {})[task_name] = {
        "status": status,
        "updated_at": _utc_timestamp(),
        **details,
    }
    state["git_head"] = _git_head()
    state["updated_at"] = _utc_timestamp()
    _atomic_json_write(state_path, state)


def _run_logged(
    *,
    name: str,
    command: list[str],
    log_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    log_path = log_dir / f"{safe_name}.log"
    print(f"\n=== {name} ===")
    print("command:", " ".join(command))
    print("log:", log_path)
    _mark_task(state, state_path, name, "running", log=str(log_path))

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=merged_env,
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
        _mark_task(
            state,
            state_path,
            name,
            "failed",
            log=str(log_path),
            return_code=return_code,
        )
        raise RuntimeError(
            f"task failed: {name}; return_code={return_code}; log={log_path}"
        )
    _mark_task(
        state,
        state_path,
        name,
        "completed",
        log=str(log_path),
        return_code=return_code,
    )


def _required_files_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() and path.is_file() for path in paths)


def _discover_test_sequence_candidates(data_root: Path) -> list[Path]:
    exact_names = {
        "test_question_window_sequences.csv",
        "test_sequences.csv",
    }
    exact: list[Path] = []
    other: list[Path] = []
    for path in sorted(data_root.rglob("*.csv")):
        lower_name = path.name.lower()
        lower_text = str(path).lower()
        if "train_valid" in lower_text or "training" in lower_name:
            continue
        if lower_name in exact_names:
            exact.append(path)
        elif "test" in lower_name and "sequence" in lower_name:
            other.append(path)
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in exact + other:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduplicated.append(path)
    return deduplicated


def _validate_real_assets(q_path: Path, ncdm_path: Path) -> dict[str, int]:
    q_matrix = load_q_matrix(q_path, "cpu")
    if q_matrix.ndim != 2:
        raise ValueError(f"Q matrix must be 2D, got {tuple(q_matrix.shape)}")
    item_count, knowledge_dim = map(int, q_matrix.shape)
    if knowledge_dim != 36:
        raise ValueError(
            f"expected the 36D expert Q matrix, got knowledge_dim={knowledge_dim}"
        )
    ncdm = OfficialNCDM(1, item_count, knowledge_dim)
    safe_load_ncdm_checkpoint(ncdm, ncdm_path, "cpu")
    difficulty_shape = tuple(ncdm.k_difficulty.weight.shape)
    discrimination_shape = tuple(ncdm.e_discrimination.weight.shape)
    if difficulty_shape != (item_count, knowledge_dim):
        raise ValueError(
            "NCDM difficulty shape mismatch: "
            f"{difficulty_shape} != {(item_count, knowledge_dim)}"
        )
    if discrimination_shape != (item_count, 1):
        raise ValueError(
            "NCDM discrimination shape mismatch: "
            f"{discrimination_shape} != {(item_count, 1)}"
        )
    return {"item_count": item_count, "knowledge_dim": knowledge_dim}


def _pilot_checkpoint_paths(
    output_root: Path,
    training_seed: int,
) -> tuple[Path, Path]:
    seed_root = output_root / f"seed_{training_seed}"
    return (
        seed_root / "base_c3dqn_topk64/best_checkpoint.pt",
        seed_root / "set_c3dqn_topk64/best_checkpoint.pt",
    )


def _pilot_history_paths(
    output_root: Path,
    training_seed: int,
) -> tuple[Path, Path]:
    seed_root = output_root / f"seed_{training_seed}"
    return (
        seed_root / "base_c3dqn_topk64/training_history.csv",
        seed_root / "set_c3dqn_topk64/training_history.csv",
    )


def _write_paired_configs(
    *,
    config_root: Path,
    output_root: Path,
    training_seed: int,
    q_matrix: Path,
    ncdm_checkpoint: Path,
    train_sequences: Path,
    max_students: int,
    epochs: int,
    selection_horizon: int,
    top_k: int,
    device: str,
) -> tuple[Path, Path]:
    seed_config_root = config_root / f"seed_{training_seed}"
    seed_output_root = output_root / f"seed_{training_seed}"
    base_config, set_config = build_paired_configs(
        q_matrix=str(q_matrix),
        ncdm_checkpoint=str(ncdm_checkpoint),
        train_valid_sequences=str(train_sequences),
        output_root=str(seed_output_root),
        max_students=max_students,
        epochs=epochs,
        selection_horizon=selection_horizon,
        seed=training_seed,
        device=device,
        use_amp=(device != "cpu"),
        top_k=top_k,
    )
    for config in (base_config, set_config):
        config["training"].update(
            {
                "batch_size": 16,
                "min_replay_size": 32,
                "replay_capacity": 10000,
                "epsilon_decay_steps": 3000,
                "updates_per_environment_step": 1,
                "candidate_chunk_size": top_k,
                "use_amp": device != "cpu",
            }
        )
    seed_config_root.mkdir(parents=True, exist_ok=True)
    base_path = seed_config_root / "base_c3dqn_topk64.yaml"
    set_path = seed_config_root / "set_c3dqn_topk64.yaml"
    base_path.write_text(yaml.safe_dump(base_config, sort_keys=False))
    set_path.write_text(yaml.safe_dump(set_config, sort_keys=False))
    return base_path, set_path


def _benchmark_required_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / "aggregate.csv",
        output_dir / "per_seed.csv",
        output_dir / "per_student.csv",
        output_dir / "policy_metadata.json",
        output_dir / "run_config.yaml",
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _validate_step_zero(output_dir: Path, tolerance: float = 1.0e-12) -> None:
    rows = _read_csv(output_dir / "per_seed.csv")
    metric_names = (
        "accuracy_micro",
        "auc_micro",
        "nll_micro",
        "brier_micro",
        "accuracy_macro",
        "auc_macro",
        "nll_macro",
        "brier_macro",
    )
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["step"]) == 0:
            grouped.setdefault(int(row["seed"]), []).append(row)
    expected_policies = {
        "Random-NCDM",
        "C3DQN-NCDM",
        "Set-C3DQN-NCDM",
    }
    for seed, seed_rows in grouped.items():
        policy_rows = {row["policy"]: row for row in seed_rows}
        missing = expected_policies.difference(policy_rows)
        if missing:
            raise ValueError(f"step-0 rows missing at evaluation seed {seed}: {missing}")
        reference = policy_rows["Random-NCDM"]
        for policy in ("C3DQN-NCDM", "Set-C3DQN-NCDM"):
            row = policy_rows[policy]
            for metric in metric_names:
                if abs(float(row[metric]) - float(reference[metric])) > tolerance:
                    raise ValueError(
                        f"step-0 mismatch at seed={seed}, policy={policy}, metric={metric}"
                    )
            if row["evaluated_students"] != reference["evaluated_students"]:
                raise ValueError(
                    f"step-0 evaluated student mismatch at seed={seed}, policy={policy}"
                )

    metadata = json.loads((output_dir / "policy_metadata.json").read_text())
    for policy in expected_policies:
        policy_metadata = dict(metadata.get(policy) or {})
        if bool(policy_metadata.get("uses_privileged_information", False)):
            raise ValueError(f"ordinary policy uses privileged information: {policy}")


def _write_combined_benchmark_summary(
    benchmark_root: Path,
    training_seeds: Iterable[int],
) -> Path:
    combined_rows: list[dict[str, str | int]] = []
    fieldnames: list[str] | None = None
    for training_seed in training_seeds:
        aggregate_path = benchmark_root / f"train_seed_{training_seed}" / "aggregate.csv"
        if not aggregate_path.exists():
            continue
        for row in _read_csv(aggregate_path):
            augmented: dict[str, str | int] = {
                "training_seed": training_seed,
                **row,
            }
            combined_rows.append(augmented)
            if fieldnames is None:
                fieldnames = list(augmented.keys())
    output_path = benchmark_root / "combined_aggregate_by_training_seed.csv"
    if combined_rows and fieldnames:
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
    return output_path


def _print_status(
    *,
    drive_root: Path,
    training_seeds: list[int],
    pilot_output_root: Path,
    profiler_output: Path,
    benchmark_root: Path,
    test_candidates: list[Path],
) -> None:
    print("\n===== durable pipeline status =====")
    print("drive root:", drive_root)
    print("git head:", _git_head())
    for seed in training_seeds:
        base_checkpoint, set_checkpoint = _pilot_checkpoint_paths(
            pilot_output_root,
            seed,
        )
        base_history, set_history = _pilot_history_paths(pilot_output_root, seed)
        print(
            f"training seed {seed}: ",
            f"Base checkpoint={base_checkpoint.exists()}, ",
            f"Set checkpoint={set_checkpoint.exists()}, ",
            f"Base history={base_history.exists()}, ",
            f"Set history={set_history.exists()}",
        )
        benchmark_dir = benchmark_root / f"train_seed_{seed}"
        print(
            f"  benchmark complete={_required_files_exist(_benchmark_required_outputs(benchmark_dir))}"
        )
    print("profile complete:", profiler_output.exists(), profiler_output)
    print("independent test candidates:")
    if test_candidates:
        for candidate in test_candidates:
            print(" -", candidate)
    else:
        print(" - none found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the C3DQN-NCDM Colab study from durable Drive outputs"
    )
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--rerun-tests", action="store_true")
    parser.add_argument("--test-sequences")
    parser.add_argument("--allow-train-valid-fallback", action="store_true")
    parser.add_argument(
        "--training-seeds",
        type=_parse_int_list,
        default=list(DEFAULT_TRAINING_SEEDS),
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=_parse_int_list,
        default=list(DEFAULT_EVALUATION_SEEDS),
    )
    parser.add_argument(
        "--steps",
        type=_parse_int_list,
        default=list(DEFAULT_STEPS),
    )
    parser.add_argument("--max-students", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--selection-horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-benchmark", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive root does not exist; mount Google Drive first: {drive_root}"
        )

    data_root = drive_root / "data/XES3G5M"
    q_matrix_path = data_root / "metadata/q_matrix_multihot_36_expert.pt"
    ncdm_checkpoint_path = data_root / "metadata/ncdm_model_36d_expert_best.pt"
    train_sequences_path = data_root / "kc_level/train_valid_sequences.csv"
    for required_path in (
        q_matrix_path,
        ncdm_checkpoint_path,
        train_sequences_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    durable_root = drive_root / "pipeline"
    log_dir = durable_root / "logs"
    state_path = durable_root / "state.json"
    state = _load_state(state_path)
    config_root = drive_root / "configs/c3dqn_topk64_pilot200"
    pilot_output_root = drive_root / "outputs/c3dqn_topk64_pilot200"
    profiler_output = (
        drive_root
        / "results/c3dqn_topk64_profile/topk64_profile.csv"
    )
    benchmark_root = (
        drive_root
        / "results/c3dqn_topk64_independent_pilot200"
    )
    benchmark_config_root = (
        drive_root
        / "configs/c3dqn_topk64_independent_pilot200"
    )

    asset_summary = _validate_real_assets(q_matrix_path, ncdm_checkpoint_path)
    print("validated 36D assets:", asset_summary)
    _mark_task(
        state,
        state_path,
        "verify_36d_assets",
        "completed",
        **asset_summary,
    )

    test_candidates = _discover_test_sequence_candidates(data_root)
    _print_status(
        drive_root=drive_root,
        training_seeds=args.training_seeds,
        pilot_output_root=pilot_output_root,
        profiler_output=profiler_output,
        benchmark_root=benchmark_root,
        test_candidates=test_candidates,
    )
    if args.status_only:
        return
    if not args.resume:
        print("No work executed. Pass --resume to run missing stages.")
        return

    current_head = _git_head()
    tests_task = dict((state.get("tasks") or {}).get("pytest") or {})
    tests_already_valid = (
        tests_task.get("status") == "completed"
        and tests_task.get("git_head") == current_head
    )
    if not args.skip_tests and (args.rerun_tests or not tests_already_valid):
        _run_logged(
            name="pytest",
            command=[sys.executable, "-m", "pytest", "-q"],
            log_dir=log_dir,
            state=state,
            state_path=state_path,
        )
        state["tasks"]["pytest"]["git_head"] = current_head
        _atomic_json_write(state_path, state)
    else:
        print("pytest skipped: already completed for this git head or --skip-tests used")

    for training_seed in args.training_seeds:
        base_config_path, set_config_path = _write_paired_configs(
            config_root=config_root,
            output_root=pilot_output_root,
            training_seed=training_seed,
            q_matrix=q_matrix_path,
            ncdm_checkpoint=ncdm_checkpoint_path,
            train_sequences=train_sequences_path,
            max_students=args.max_students,
            epochs=args.epochs,
            selection_horizon=args.selection_horizon,
            top_k=args.top_k,
            device=args.device,
        )
        base_checkpoint, set_checkpoint = _pilot_checkpoint_paths(
            pilot_output_root,
            training_seed,
        )
        for architecture, config_path, checkpoint_path in (
            ("base", base_config_path, base_checkpoint),
            ("set", set_config_path, set_checkpoint),
        ):
            task_name = f"train_{architecture}_seed_{training_seed}"
            if checkpoint_path.exists():
                print(f"skip {task_name}: {checkpoint_path}")
                _mark_task(
                    state,
                    state_path,
                    task_name,
                    "completed",
                    skipped_existing=True,
                    checkpoint=str(checkpoint_path),
                )
                continue
            _run_logged(
                name=task_name,
                command=[
                    sys.executable,
                    "-u",
                    "scripts/train_ncdm_c3dqn.py",
                    "--config",
                    str(config_path),
                ],
                log_dir=log_dir,
                state=state,
                state_path=state_path,
            )
            if not checkpoint_path.exists():
                raise RuntimeError(
                    f"training completed without checkpoint: {checkpoint_path}"
                )

    if not profiler_output.exists():
        reference_seed = args.training_seeds[0]
        base_checkpoint, set_checkpoint = _pilot_checkpoint_paths(
            pilot_output_root,
            reference_seed,
        )
        profiler_output.parent.mkdir(parents=True, exist_ok=True)
        _run_logged(
            name="profile_topk64",
            command=[
                sys.executable,
                "-u",
                "scripts/run_profile_c3dqn_ncdm_topk64.py",
                "--q-matrix",
                str(q_matrix_path),
                "--ncdm-checkpoint",
                str(ncdm_checkpoint_path),
                "--base-checkpoint",
                str(base_checkpoint),
                "--set-checkpoint",
                str(set_checkpoint),
                "--output",
                str(profiler_output),
                "--top-k",
                str(args.top_k),
                "--raw-candidates",
                "256",
                "--batch-size",
                "8",
                "--history-length",
                "5",
                "--warmup",
                "10",
                "--repeats",
                "30",
                "--chunk-size",
                str(args.top_k),
                "--device",
                args.device,
            ],
            log_dir=log_dir,
            state=state,
            state_path=state_path,
        )
    else:
        print("skip profile_topk64: existing output", profiler_output)
        _mark_task(
            state,
            state_path,
            "profile_topk64",
            "completed",
            skipped_existing=True,
            output=str(profiler_output),
        )

    if args.test_sequences:
        test_sequences_path = Path(args.test_sequences).expanduser().resolve()
        if not test_sequences_path.exists():
            raise FileNotFoundError(test_sequences_path)
    elif len(test_candidates) == 1:
        test_sequences_path = test_candidates[0]
    elif not test_candidates and args.allow_train_valid_fallback:
        test_sequences_path = train_sequences_path
        print("WARNING: using train_valid_sequences.csv as a non-independent fallback")
    else:
        message = (
            "Independent test evaluation is blocked. "
            "Provide --test-sequences PATH. Candidates: "
            + (", ".join(map(str, test_candidates)) if test_candidates else "none")
        )
        print("\n", message)
        _mark_task(
            state,
            state_path,
            "independent_benchmark",
            "blocked",
            reason=message,
        )
        return

    print("independent test sequences:", test_sequences_path)
    benchmark_config_root.mkdir(parents=True, exist_ok=True)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for training_seed in args.training_seeds:
        base_checkpoint, set_checkpoint = _pilot_checkpoint_paths(
            pilot_output_root,
            training_seed,
        )
        output_dir = benchmark_root / f"train_seed_{training_seed}"
        required_outputs = _benchmark_required_outputs(output_dir)
        task_name = f"benchmark_train_seed_{training_seed}"
        if _required_files_exist(required_outputs) and not args.force_benchmark:
            print(f"skip {task_name}: output already complete")
            _validate_step_zero(output_dir)
            _mark_task(
                state,
                state_path,
                task_name,
                "completed",
                skipped_existing=True,
                output_dir=str(output_dir),
            )
            continue
        config = {
            "device": args.device,
            "assets": {
                "q_matrix": str(q_matrix_path),
                "ncdm_checkpoint": str(ncdm_checkpoint_path),
                "test_sequences": str(test_sequences_path),
            },
            "benchmark": {
                "track": "ncdm_native",
                "policies": [
                    "Random-NCDM",
                    "C3DQN-NCDM",
                    "Set-C3DQN-NCDM",
                ],
                "seeds": list(args.evaluation_seeds),
                "max_students": int(args.max_students),
                "steps": list(args.steps),
                "selection_horizon": int(args.selection_horizon),
                "query_ratio": 0.2,
                "min_query_items": 5,
                "save_predictions": True,
                "save_traces": True,
                "output_dir": str(output_dir),
                "c3dqn_ncdm_checkpoint": str(base_checkpoint),
                "set_c3dqn_ncdm_checkpoint": str(set_checkpoint),
            },
        }
        config_path = benchmark_config_root / f"train_seed_{training_seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        _run_logged(
            name=task_name,
            command=[
                sys.executable,
                "-u",
                "scripts/run_benchmark_config.py",
                "--config",
                str(config_path),
            ],
            log_dir=log_dir,
            state=state,
            state_path=state_path,
        )
        if not _required_files_exist(required_outputs):
            raise RuntimeError(
                f"benchmark completed without all required outputs: {output_dir}"
            )
        _validate_step_zero(output_dir)
        completion = {
            "training_seed": training_seed,
            "evaluation_seeds": list(args.evaluation_seeds),
            "steps": list(args.steps),
            "test_sequences": str(test_sequences_path),
            "completed_at": _utc_timestamp(),
        }
        _atomic_json_write(output_dir / "COMPLETED.json", completion)

    combined_summary = _write_combined_benchmark_summary(
        benchmark_root,
        args.training_seeds,
    )
    _mark_task(
        state,
        state_path,
        "independent_benchmark",
        "completed",
        combined_summary=str(combined_summary),
        test_sequences=str(test_sequences_path),
    )
    print("\nPipeline completed.")
    print("state:", state_path)
    print("combined benchmark summary:", combined_summary)


if __name__ == "__main__":
    main()
