#!/usr/bin/env python
"""Train the DDPG CAT policy from configured Google Drive assets."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from evaluation.offline_eval import MissingAssetsError
from utils.config import load_yaml_config


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Train KCQRL-RL DDPG policy."); parser.add_argument("--config",default="configs/default.yaml")
    args=parser.parse_args(argv); cfg_path=Path(args.config); cfg_path=cfg_path if cfg_path.is_absolute() else ROOT/cfg_path
    try:
        cfg=load_yaml_config(cfg_path)
        assets=cfg.get("assets",{}) or {}; base=Path(assets.get("base_dir",".")).expanduser()
        required=["q_matrix","item_bank","ncdm_checkpoint","train_sequences"]
        missing=[]
        for k in required:
            v=assets.get(k)
            if not v:
                missing.append(base / f"<missing {k}>"); continue
            path=Path(str(v)).expanduser(); path=path if path.is_absolute() else base/path
            if not path.exists(): missing.append(path)
        if missing: raise MissingAssetsError(missing)
        from agents.trainer import DDPGTrainer
        dev_cfg=cfg.get("device","auto"); device=torch.device("cuda" if dev_cfg=="auto" and torch.cuda.is_available() else ("cpu" if dev_cfg=="auto" else dev_cfg))
        trainer=DDPGTrainer(cfg, device)
        logs, out=trainer.train(); print(f"Saved actor checkpoint: {out}"); print(f"Final metrics: {logs[-1] if logs else {}}")
        return 0
    except MissingAssetsError as exc:
        print("KCQRL-RL training cannot start because external Google Drive assets are missing.")
        print("Mount Google Drive in Colab and verify configs/default.yaml points to these files:")
        for p in exc.missing_paths: print(f"  - {p}")
        return 2
    except Exception as exc:
        print(f"KCQRL-RL training failed: {exc}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
