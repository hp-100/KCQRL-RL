#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.config import load_yaml_config
from agents.mirt_ddpg_trainer import MIRTDDPGTrainer

def main(argv=None):
    p=argparse.ArgumentParser(description='Train independent DDPG-MIRT policy')
    p.add_argument('--config',default='configs/mirt_ddpg.yaml'); args=p.parse_args(argv)
    cfg=load_yaml_config(ROOT/args.config if not Path(args.config).is_absolute() else Path(args.config))
    return 0 if MIRTDDPGTrainer(cfg).train() is not None else 1
if __name__=='__main__': raise SystemExit(main())
