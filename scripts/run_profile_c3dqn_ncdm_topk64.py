"""Execute the Top-K=64 profiler with the repository root on sys.path."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.profile_c3dqn_ncdm_topk64 import main


if __name__ == "__main__":
    main()
