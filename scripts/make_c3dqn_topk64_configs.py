"""Generate paired Base/Set C3DQN-NCDM configs with identical Top-K=64 settings."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def build_paired_configs(
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
) -> tuple[dict, dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_students <= 0:
        raise ValueError("max_students must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if selection_horizon <= 0:
        raise ValueError("selection_horizon must be positive")

    shared = {
        "device": str(device),
        "candidate_pool": {
            "prefilter_enabled": True,
            "prefilter_top_k": int(top_k),
            "diversity_quota": min(8, int(top_k)),
            "weights": {
                "uncertainty": 0.35,
                "weakness": 0.25,
                "novelty": 0.15,
                "difficulty": 0.15,
                "discrimination": 0.10,
            },
        },
        "alpha_fit": {
            "initial_steps": 8,
            "incremental_steps": 3,
            "lr": 0.05,
            "early_stop_tol": 1.0e-5,
            "grad_clip": 5.0,
        },
        "training": {
            "seed": int(seed),
            "epochs": int(epochs),
            "max_students": int(max_students),
            "train_ratio": 0.8,
            "query_ratio": 0.2,
            "min_query_items": 2,
            "selection_horizon": int(selection_horizon),
            "batch_size": 8,
            "replay_capacity": 5000,
            "min_replay_size": 8,
            "updates_per_environment_step": 1,
            "learning_rate": 1.0e-3,
            "gamma": 0.99,
            "gradient_clip": 5.0,
            "tau": 0.01,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 500,
            "use_amp": bool(use_amp),
            "candidate_chunk_size": int(top_k),
        },
        "paths": {
            "q_matrix": str(q_matrix),
            "ncdm_checkpoint": str(ncdm_checkpoint),
            "train_valid_sequences": str(train_valid_sequences),
        },
    }

    base = deepcopy(shared)
    base["model"] = {
        "architecture": "base_c3dqn",
        "d_model": 64,
        "n_heads": 4,
        "num_history_layers": 1,
        "dropout": 0.0,
    }
    base["training"]["output_dir"] = str(
        Path(output_root) / "base_c3dqn_topk64"
    )

    set_cfg = deepcopy(shared)
    set_cfg["model"] = {
        "architecture": "set_c3dqn",
        "d_model": 64,
        "n_heads": 4,
        "num_history_layers": 1,
        "dropout": 0.0,
        "candidate_set_encoder": "isab",
        "num_set_layers": 1,
        "num_inducing_points": 8,
        "set_attention_heads": 4,
        "use_relative_features": True,
        "set_pool_in_value_head": True,
        "full_attention_max_candidates": int(top_k),
        "debug_mode": False,
    }
    set_cfg["training"]["output_dir"] = str(
        Path(output_root) / "set_c3dqn_topk64"
    )
    return base, set_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired Base/Set C3DQN configs with shared Top-K=64"
    )
    parser.add_argument("--q-matrix", required=True)
    parser.add_argument("--ncdm-checkpoint", required=True)
    parser.add_argument("--train-valid-sequences", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--max-students", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--selection-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    base_cfg, set_cfg = build_paired_configs(
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
    )

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    base_path = config_dir / "base_c3dqn_topk64.yaml"
    set_path = config_dir / "set_c3dqn_topk64.yaml"
    base_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False))
    set_path.write_text(yaml.safe_dump(set_cfg, sort_keys=False))
    print(base_path)
    print(set_path)


if __name__ == "__main__":
    main()
