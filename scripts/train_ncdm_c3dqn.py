from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ncdm_c3dqn_app import (
    build_q_network_from_config,
    build_trainer_from_config,
    main,
)

__all__ = [
    "build_q_network_from_config",
    "build_trainer_from_config",
    "main",
]


if __name__ == "__main__":
    main()
