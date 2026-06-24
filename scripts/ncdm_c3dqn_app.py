"""CLI construction helpers for Base and Set C3DQN-NCDM."""
from __future__ import annotations

from pathlib import Path
import argparse
import time

import torch
import yaml

from agents.ncdm_c3dqn_trainer import C3DQNTransition
from agents.ncdm_c3dqn_trainer_v2 import NCDMC3DQNTrainer
from models.ncdm import OfficialNCDM, load_q_matrix, safe_load_ncdm_checkpoint
from models.ncdm_candidate_features import NCDMItemFeatureCache
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.set_ncdm_candidate_q_network import SetConditionedNCDMQNetwork


def _device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_q_network_from_config(model_cfg: dict, knowledge_dim: int):
    cfg = dict(model_cfg or {})
    architecture = str(cfg.get("architecture", "base_c3dqn"))
    common = {
        "d_model": int(cfg.get("d_model", 64)),
        "n_heads": int(cfg.get("n_heads", 4)),
        "num_history_layers": int(cfg.get("num_history_layers", 1)),
        "dropout": float(cfg.get("dropout", 0.0)),
    }
    if architecture == "base_c3dqn":
        return CandidateConditionedNCDMQNetwork(knowledge_dim, **common)
    if architecture == "set_c3dqn":
        return SetConditionedNCDMQNetwork(
            knowledge_dim,
            **common,
            candidate_set_encoder=str(cfg.get("candidate_set_encoder", "isab")),
            num_set_layers=int(cfg.get("num_set_layers", 1)),
            num_inducing_points=int(cfg.get("num_inducing_points", 8)),
            set_attention_heads=int(cfg.get("set_attention_heads", common["n_heads"])),
            use_relative_features=bool(cfg.get("use_relative_features", True)),
            set_pool_in_value_head=bool(cfg.get("set_pool_in_value_head", True)),
            full_attention_max_candidates=int(cfg.get("full_attention_max_candidates", 128)),
            debug_mode=bool(cfg.get("debug_mode", False)),
        )
    raise ValueError(f"unknown C3DQN architecture: {architecture}")


def build_trainer_from_config(cfg: dict, *, synthetic_smoke: bool = False):
    device = _device(cfg.get("device", "auto"))
    train_cfg = dict(cfg.get("training") or {})
    model_cfg = dict(cfg.get("model") or {})
    if synthetic_smoke:
        out_dir = train_cfg.get("output_dir", "outputs/ncdm_c3dqn_smoke")
        knowledge_dim = int(cfg.get("knowledge_dim", 36))
        item_count = int(cfg.get("synthetic_item_count", 16))
        q_matrix = torch.randint(0, 2, (item_count, knowledge_dim), device=device).float()
        q_matrix[:, 0] = 1
        ncdm = OfficialNCDM(1, item_count, knowledge_dim).to(device)
        paths = {}
    else:
        paths = dict(cfg.get("paths") or {})
        required = ["q_matrix", "ncdm_checkpoint", "train_valid_sequences"]
        missing = [name for name in required if not paths.get(name)]
        if missing:
            raise ValueError(f"real C3DQN-NCDM training requires paths.{missing}")
        q_matrix = load_q_matrix(paths["q_matrix"], device)
        ncdm = OfficialNCDM(1, q_matrix.shape[0], q_matrix.shape[1]).to(device)
        safe_load_ncdm_checkpoint(ncdm, paths["ncdm_checkpoint"], device)
        out_dir = train_cfg["output_dir"]
    for parameter in ncdm.parameters():
        parameter.requires_grad_(False)
    ncdm.eval()
    cache = NCDMItemFeatureCache(
        ncdm,
        q_matrix,
        device,
        allow_item_count_intersection=bool(train_cfg.get("allow_item_count_intersection", False)),
    )
    online = build_q_network_from_config(model_cfg, cache.knowledge_dim).to(device)
    target = build_q_network_from_config(model_cfg, cache.knowledge_dim).to(device)
    trainer = NCDMC3DQNTrainer(
        online,
        target,
        cache,
        int(train_cfg.get("selection_horizon", cfg.get("selection_horizon", 5))),
        out_dir,
        gamma=float(train_cfg.get("gamma", 0.99)),
        lr=float(train_cfg.get("learning_rate", 0.001)),
        gradient_clip=float(train_cfg.get("gradient_clip", 5.0)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        replay_capacity=int(train_cfg.get("replay_capacity", 10000)),
        min_replay_size=int(train_cfg.get("min_replay_size", 32)),
        updates_per_environment_step=int(train_cfg.get("updates_per_environment_step", 1)),
        tau=float(train_cfg.get("tau", 0.01)),
        target_update_interval=int(train_cfg.get("target_update_interval", 0)),
        epsilon_start=float(train_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(train_cfg.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(train_cfg.get("epsilon_decay_steps", 1000)),
        alpha_fit=dict(cfg.get("alpha_fit") or train_cfg.get("alpha_fit") or {}),
        reward_config=dict(cfg.get("reward") or train_cfg.get("reward") or {}),
        model_config=model_cfg,
        candidate_pool_config=dict(cfg.get("candidate_pool") or train_cfg.get("candidate_pool") or {}),
        requested_amp=bool(train_cfg.get("use_amp", False)),
        candidate_chunk_size=train_cfg.get("candidate_chunk_size"),
        seed=int(train_cfg.get("seed", 42)),
        ncdm=ncdm,
        q_matrix=q_matrix,
    )
    return trainer, ncdm, q_matrix, paths


def run_synthetic_smoke(trainer: NCDMC3DQNTrainer) -> dict[str, float]:
    start = time.perf_counter()
    knowledge_dim = trainer.cache.knowledge_dim
    candidate_ids = list(range(1, min(trainer.cache.item_count, 7)))
    if len(candidate_ids) < 2:
        raise ValueError("synthetic smoke requires at least three cached items")
    for sample_index in range(max(8, trainer.min_replay_size)):
        selected = candidate_ids[sample_index % len(candidate_ids)]
        next_candidates = [item for item in candidate_ids if item != selected]
        current_count = [0.0] * knowledge_dim
        next_count = [1.0 if index == 0 else 0.0 for index in range(knowledge_dim)]
        trainer.replay.push(
            C3DQNTransition(
                history_item_ids=[0],
                history_responses=[1.0],
                candidate_item_ids=list(candidate_ids),
                mastery=[0.5] * knowledge_dim,
                coverage=[0.0] * knowledge_dim,
                policy_step=0,
                selected_item_id=selected,
                reward=0.1,
                reward_components={"prediction_gain": 0.1, "diagnosis_gain": 0.0, "coverage_gain": 0.0, "total": 0.1},
                next_history_item_ids=[0, selected],
                next_history_responses=[1.0, 1.0],
                next_candidate_item_ids=next_candidates,
                next_mastery=[0.55] * knowledge_dim,
                next_coverage=[value / max(1, trainer.selection_horizon) for value in next_count],
                next_policy_step=1,
                done=False,
                coverage_count=current_count,
                next_coverage_count=next_count,
                raw_candidate_count=len(candidate_ids),
                filtered_candidate_count=len(candidate_ids),
            )
        )
    stats = trainer.update_once() or {"td_loss": 0.0, "mean_q": 0.0, "target_q_mean": 0.0}
    metrics = {
        "epoch": 1,
        "mean_total_reward": 0.1,
        "mean_prediction_reward": 0.1,
        "mean_diagnosis_reward": 0.0,
        "mean_coverage_reward": 0.0,
        "td_loss": stats["td_loss"],
        "mean_q": stats.get("mean_q", 0.0),
        "target_q_mean": stats.get("target_q_mean", 0.0),
        "epsilon": 0.0,
        "replay_size": len(trainer.replay),
        "selected_unique_items": len(candidate_ids),
        "item_exposure_gini": 0.0,
        "validation_query_nll": 0.0,
        "validation_query_auc": 0.5,
        "validation_query_brier": 0.25,
        "validation_mastery_entropy": 1.0,
        "validation_concept_coverage": 0.0,
        "feature_build_seconds": 0.0,
        "alpha_fit_seconds": 0.0,
        "reward_seconds": 0.0,
        "network_forward_seconds": 0.0,
        "network_update_seconds": trainer.time_acc.get("network_update_seconds", 0.0),
        "validation_seconds": 0.0,
        "candidate_prefilter_seconds": 0.0,
        "mean_raw_candidate_count": float(len(candidate_ids)),
        "mean_filtered_candidate_count": float(len(candidate_ids)),
        "total_epoch_seconds": time.perf_counter() - start,
    }
    trainer._write_history([metrics])
    trainer.save_checkpoint(metrics, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C3DQN-NCDM")
    parser.add_argument("--config", default="configs/ncdm_c3dqn_smoke.yaml")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    trainer, _ncdm, _q_matrix, paths = build_trainer_from_config(cfg, synthetic_smoke=args.synthetic_smoke)
    train_cfg = dict(cfg.get("training") or {})
    if args.synthetic_smoke:
        metrics = run_synthetic_smoke(trainer)
    else:
        history = trainer.train(
            paths["train_valid_sequences"],
            epochs=int(train_cfg.get("epochs", 1)),
            train_ratio=float(train_cfg.get("train_ratio", 0.8)),
            max_students=train_cfg.get("max_students"),
            query_ratio=float(train_cfg.get("query_ratio", 0.2)),
            min_query_items=int(train_cfg.get("min_query_items", 2)),
        )
        metrics = history[-1]
    print(
        "C3DQN-NCDM timing breakdown:",
        {key: value for key, value in metrics.items() if key.endswith("seconds")},
    )
