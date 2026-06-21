"""Generate controlled Base-C3DQN reward-ablation configs.

Only the reward configuration and output directory differ across experiments. The
network, NCDM assets, Top-K prefilter, training data, seed and Double-DQN target are
kept identical.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml

from scripts.make_c3dqn_topk64_configs import build_paired_configs


REWARD_EXPERIMENTS: dict[str, dict] = {
    "legacy": {
        "mode": "legacy",
        "prediction_weight": 1.0,
        "diagnosis_weight": 0.2,
        "coverage_weight": 0.05,
        "prediction_scale": 10.0,
        "reward_clip": 5.0,
    },
    "prediction": {
        "mode": "prediction",
        "prediction_weight": 1.0,
        "diagnosis_weight": 0.0,
        "coverage_weight": 0.0,
        "prediction_scale": 5.0,
        "reward_clip": 5.0,
    },
    "prediction_coverage": {
        "mode": "prediction_coverage",
        "prediction_weight": 1.0,
        "diagnosis_weight": 0.0,
        "coverage_weight": 0.02,
        "prediction_scale": 5.0,
        "reward_clip": 5.0,
    },
}


def build_reward_ablation_configs(
    *,
    q_matrix: str,
    ncdm_checkpoint: str,
    train_valid_sequences: str,
    output_root: str,
    max_students: int,
    epochs: int,
    selection_horizon: int,
    seed: int,
    device: str,
    use_amp: bool,
    top_k: int = 64,
    experiments: Iterable[str] = ("prediction", "prediction_coverage"),
) -> dict[str, dict]:
    base_config, _ = build_paired_configs(
        q_matrix=q_matrix,
        ncdm_checkpoint=ncdm_checkpoint,
        train_valid_sequences=train_valid_sequences,
        output_root=output_root,
        max_students=max_students,
        epochs=epochs,
        selection_horizon=selection_horizon,
        seed=seed,
        device=device,
        use_amp=use_amp,
        top_k=top_k,
    )

    configs: dict[str, dict] = {}
    for experiment in experiments:
        if experiment not in REWARD_EXPERIMENTS:
            raise ValueError(f"unknown reward experiment: {experiment}")
        config = deepcopy(base_config)
        config["reward"] = deepcopy(REWARD_EXPERIMENTS[experiment])
        config["training"]["output_dir"] = str(
            Path(output_root) / experiment / f"seed_{seed}"
        )
        config["experiment"] = {
            "family": "base_c3dqn_reward_ablation",
            "name": experiment,
            "training_seed": int(seed),
        }
        configs[experiment] = config
    return configs


def write_reward_ablation_configs(
    configs: dict[str, dict],
    config_root: str | Path,
    seed: int,
) -> dict[str, Path]:
    root = Path(config_root)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for experiment, config in configs.items():
        path = root / experiment / f"seed_{seed}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[experiment] = path
    return written


def _parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one experiment")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Base-C3DQN reward-ablation YAML files"
    )
    parser.add_argument("--q-matrix", required=True)
    parser.add_argument("--ncdm-checkpoint", required=True)
    parser.add_argument("--train-valid-sequences", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--max-students", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--selection-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--experiments", type=_parse_csv, default=["prediction", "prediction_coverage"])
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    configs = build_reward_ablation_configs(
        q_matrix=args.q_matrix,
        ncdm_checkpoint=args.ncdm_checkpoint,
        train_valid_sequences=args.train_valid_sequences,
        output_root=args.output_root,
        max_students=args.max_students,
        epochs=args.epochs,
        selection_horizon=args.selection_horizon,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
        top_k=args.top_k,
        experiments=args.experiments,
    )
    for path in write_reward_ablation_configs(configs, args.config_root, args.seed).values():
        print(path)


if __name__ == "__main__":
    main()
