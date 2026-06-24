"""Run one benchmark YAML file with the repository root on ``sys.path``."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import BenchmarkV2Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BenchmarkV2Evaluator from YAML")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    rows = BenchmarkV2Evaluator(config).run()
    print(f"benchmark rows: {len(rows)}")
    print(f"benchmark output: {(config.get('benchmark') or {}).get('output_dir')}")


if __name__ == "__main__":
    main()
