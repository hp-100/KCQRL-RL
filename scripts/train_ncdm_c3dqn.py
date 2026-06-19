from __future__ import annotations
import argparse, yaml, torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.ncdm import OfficialNCDM, load_q_matrix, safe_load_ncdm_checkpoint
from models.ncdm_candidate_features import NCDMItemFeatureCache
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from agents.ncdm_c3dqn_trainer import NCDMC3DQNTrainer

def main() -> None:
    p=argparse.ArgumentParser(description="Train C3DQN-NCDM (NCDM-native adaptive item selection)")
    p.add_argument("--config", default="configs/ncdm_c3dqn_smoke.yaml"); p.add_argument("--synthetic-smoke", action="store_true")
    args=p.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    out=cfg.get("output_dir","outputs/ncdm_c3dqn_smoke"); k=int(cfg.get("knowledge_dim",36)); items=int(cfg.get("synthetic_item_count",16))
    if args.synthetic_smoke or cfg.get("synthetic", False):
        q_matrix=torch.randint(0,2,(items,k)).float(); q_matrix[:,0]=1
        ncdm=OfficialNCDM(1, items, k)
    else:
        q_matrix=load_q_matrix(cfg["paths"]["q_matrix"]); ncdm=OfficialNCDM(1, q_matrix.shape[0], q_matrix.shape[1]); safe_load_ncdm_checkpoint(ncdm, cfg["paths"]["ncdm_checkpoint"])
    for param in ncdm.parameters(): param.requires_grad_(False)
    cache=NCDMItemFeatureCache(ncdm,q_matrix)
    model_cfg=cfg.get("model",{})
    online=CandidateConditionedNCDMQNetwork(cache.knowledge_dim, d_model=int(model_cfg.get("d_model",64)), n_heads=int(model_cfg.get("n_heads",4)), num_history_layers=int(model_cfg.get("num_history_layers",1)))
    target=CandidateConditionedNCDMQNetwork(cache.knowledge_dim, d_model=int(model_cfg.get("d_model",64)), n_heads=int(model_cfg.get("n_heads",4)), num_history_layers=int(model_cfg.get("num_history_layers",1)))
    trainer=NCDMC3DQNTrainer(online,target,cache,int(cfg.get("selection_horizon",5)),out,gamma=float(cfg.get("gamma",0.99)),lr=float(cfg.get("learning_rate",0.001)),gradient_clip=float(cfg.get("gradient_clip",5.0)))
    metrics=trainer.run_synthetic_smoke_epoch(); print("C3DQN-NCDM timing breakdown:", {k:v for k,v in metrics.items() if k.endswith("seconds")})
if __name__ == "__main__": main()
