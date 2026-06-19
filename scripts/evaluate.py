#!/usr/bin/env python
"""Run offline CAT/RL evaluation for KCQRL-RL."""
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
    parser = argparse.ArgumentParser(description="Evaluate CAT/RL policies offline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--debug", action="store_true", help="Use small debug limits and print friendly diagnostics.")
    parser.add_argument("--ddpg-checkpoint", default="outputs/ddpg_actor.pt", help="Path to trained DDPG actor checkpoint.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = load_config(config_path)
        evaluator = CATOfflineEvaluator(config, debug=args.debug, ddpg_checkpoint=args.ddpg_checkpoint)
        results = evaluator.evaluate()
    except MissingAssetsError as exc:
        print("KCQRL-RL evaluation cannot run because external Google Drive assets are missing.")
        print("Mount Google Drive in Colab and verify configs/default.yaml points to these files:")
        for p in exc.missing_paths:
            print(f"  - {p}")
        return 0 if args.debug else 2
    except Exception as exc:
        print(f"KCQRL-RL evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("policy,students,interactions,accuracy,reward")
    for r in results:
        print(f"{r.policy},{r.students},{r.interactions},{r.accuracy:.4f},{r.reward:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
