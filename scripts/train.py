#!/usr/bin/env python
"""Minimal training entry point for the KCQRL-RL framework.

The full DDPG training loop depends on external Google Drive datasets and model
checkpoints. This script validates configuration and assets first so Colab/cloud
runs fail with actionable messages instead of import errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.offline_eval import CATOfflineEvaluator, MissingAssetsError
from utils.config import load_yaml_config


def load_config(path: Path) -> dict:
    return load_yaml_config(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train KCQRL-RL policy.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = load_config(config_path)
        evaluator = CATOfflineEvaluator(config, debug=True)
        evaluator.ensure_assets()
    except MissingAssetsError as exc:
        print("KCQRL-RL training cannot start because external Google Drive assets are missing.")
        print("Mount Google Drive in Colab and verify configs/default.yaml points to these files:")
        for p in exc.missing_paths:
            print(f"  - {p}")
        return 2
    except Exception as exc:
        print(f"KCQRL-RL training setup failed: {exc}", file=sys.stderr)
        return 1

    train_cfg = config.get("training", {}) or {}
    print("Assets found. Minimal training scaffold is ready.")
    print(f"Configured episodes: {train_cfg.get('episodes', 10)}")
    print("Implement full DDPG optimization here or plug in agents/trainer.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
