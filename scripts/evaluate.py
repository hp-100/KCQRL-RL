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
from evaluation.benchmark import BenchmarkV2Evaluator
from utils.config import load_yaml_config


def load_config(path: Path) -> dict:
    return load_yaml_config(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CAT/RL policies offline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--debug", action="store_true", help="Use small debug limits and print friendly diagnostics.")
    parser.add_argument("--ddpg-checkpoint", default="outputs/ddpg_actor.pt", help="Path to trained DDPG actor checkpoint.")
    parser.add_argument("--ddpg-mirt-checkpoint", default=None, help="Path to trained DDPG-MIRT actor checkpoint.")
    parser.add_argument("--track", default=None, help="Evaluation track, e.g. mirt_native.")
    parser.add_argument("--protocol", default=None, help="Evaluation protocol: legacy or benchmark_v2.")
    parser.add_argument("--seeds", default=None, help="Comma-separated benchmark seeds.")
    parser.add_argument("--max-students", type=int, default=None, help="Maximum students for benchmark_v2.")
    parser.add_argument("--steps", default=None, help="Comma-separated benchmark checkpoints.")
    parser.add_argument("--output-dir", default=None, help="benchmark_v2 output directory.")
    parser.add_argument("--policies", default=None, help="Comma-separated benchmark_v2 policies to run.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = load_config(config_path)
        protocol = args.protocol or (config.get("benchmark", {}) or {}).get("protocol")
        if protocol == "benchmark_v2":
            seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
            steps = [int(x) for x in args.steps.split(",")] if args.steps else None
            policies = [x.strip() for x in args.policies.split(",") if x.strip()] if args.policies else None
            evaluator = BenchmarkV2Evaluator(config, debug=args.debug, ddpg_checkpoint=args.ddpg_checkpoint, ddpg_mirt_checkpoint=args.ddpg_mirt_checkpoint, track=args.track, seeds=seeds, max_students=args.max_students, steps=steps, output_dir=args.output_dir, policies=policies)
            rows = evaluator.run()
            print(f"benchmark_v2 complete: wrote outputs to {evaluator.output_dir}")
            print("policy,seed,step,students,accuracy_micro,auc_micro,nll_micro,brier_micro")
            for r in rows:
                print(f"{r['policy']},{r['seed']},{r['step']},{r['students']},{float(r['accuracy_micro']):.4f},{float(r['auc_micro']):.4f},{float(r['nll_micro']):.4f},{float(r['brier_micro']):.4f}")
            return 0
        print("Legacy evaluation protocol is not recommended for paper results. Use --protocol benchmark_v2.")
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

    print("policy,students,interactions,accuracy,auc,nll,brier,reward")
    for r in results:
        print(f"{r.policy},{r.students},{r.interactions},{r.accuracy:.4f},{r.auc:.4f},{r.nll:.4f},{r.brier:.4f},{r.reward:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
