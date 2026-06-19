"""Minimal Masked Dueling Double DQN utilities and smoke trainer for C3DQN-NCDM."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, random, time, subprocess
from typing import Any, Sequence
import torch
import torch.nn.functional as F
from torch import nn
from models.ncdm_candidate_q_network import CandidateConditionedNCDMQNetwork
from models.ncdm_candidate_features import NCDMItemFeatureCache, pad_c3dqn_batch

@dataclass
class C3DQNTransition:
    history_item_ids: list[int]; history_responses: list[float]; candidate_item_ids: list[int]
    mastery: list[float]; coverage: list[float]; policy_step: int; selected_item_id: int
    reward: float; reward_components: dict[str, float]
    next_history_item_ids: list[int]; next_history_responses: list[float]; next_candidate_item_ids: list[int]
    next_mastery: list[float]; next_coverage: list[float]; next_policy_step: int; done: bool

class C3DQNReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = int(capacity); self._data: list[C3DQNTransition] = []
    def push(self, t: C3DQNTransition) -> None:
        if t.selected_item_id not in t.candidate_item_ids:
            raise ValueError("selected_item_id must belong to candidate_item_ids")
        self._data.append(t)
        if len(self._data) > self.capacity: self._data.pop(0)
    def sample(self, batch_size: int) -> list[C3DQNTransition]:
        return random.sample(self._data, min(batch_size, len(self._data)))
    def __len__(self) -> int: return len(self._data)
    def state_dict(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self._data]


def masked_argmax(q_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(~candidate_mask.bool(), -1.0e9).argmax(dim=1)


def compute_double_dqn_loss(online_net: nn.Module, target_net: nn.Module, batch: dict[str, torch.Tensor], next_batch: dict[str, torch.Tensor], rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> tuple[torch.Tensor, dict[str, float]]:
    q_values, _ = online_net(batch["history_features"], batch["history_mask"], batch["candidate_features"], batch["candidate_mask"], batch["global_features"])
    chosen_q = q_values.gather(1, batch["action_index"].view(-1,1)).squeeze(1)
    with torch.no_grad():
        next_online_q, _ = online_net(next_batch["history_features"], next_batch["history_mask"], next_batch["candidate_features"], next_batch["candidate_mask"], next_batch["global_features"])
        next_action = masked_argmax(next_online_q, next_batch["candidate_mask"])
        next_target_q, _ = target_net(next_batch["history_features"], next_batch["history_mask"], next_batch["candidate_features"], next_batch["candidate_mask"], next_batch["global_features"])
        next_q = next_target_q.gather(1, next_action.view(-1,1)).squeeze(1)
        target = rewards + float(gamma) * (1.0 - dones.float()) * next_q
    loss = F.smooth_l1_loss(chosen_q, target)
    return loss, {"mean_q": float(chosen_q.mean().item()), "target_q_mean": float(target.mean().item()), "next_action_mean": float(next_action.float().mean().item())}


def validate_c3dqn_checkpoint_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = ["knowledge_dim", "history_feature_dim", "candidate_feature_dim", "global_feature_dim", "selection_horizon", "warm_start_items", "alpha_fit", "candidate_pool_config"]
    for key in keys:
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"C3DQN-NCDM checkpoint protocol mismatch for {key}: {metadata.get(key)!r} != {expected.get(key)!r}")

def build_checkpoint_metadata(*, knowledge_dim: int, selection_horizon: int, warm_start_items: int, alpha_fit: dict, reward_config: dict, model_config: dict, candidate_pool_config: dict, ncdm_item_count: int, q_matrix_item_count: int, training_seed: int, validation_metrics: dict, epoch: int) -> dict[str, Any]:
    try: git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: git_commit = "unknown"
    return {"actor_architecture":"candidate_conditioned_attention_dueling_double_dqn", "knowledge_dim":knowledge_dim, "history_feature_dim":2*knowledge_dim+3, "candidate_feature_dim":2*knowledge_dim+1, "global_feature_dim":2*knowledge_dim+1, "selection_horizon":selection_horizon, "warm_start_items":warm_start_items, "alpha_fit":dict(alpha_fit), "reward_config":dict(reward_config), "model_config":dict(model_config), "candidate_pool_config":dict(candidate_pool_config), "ncdm_item_count":ncdm_item_count, "q_matrix_item_count":q_matrix_item_count, "q_matrix_knowledge_dim":knowledge_dim, "training_seed":training_seed, "validation_metrics":dict(validation_metrics), "epoch":epoch, "git_commit":git_commit}

class NCDMC3DQNTrainer:
    """Small synthetic-capable trainer; real CSV orchestration can reuse these primitives."""
    def __init__(self, online_net: CandidateConditionedNCDMQNetwork, target_net: CandidateConditionedNCDMQNetwork, cache: NCDMItemFeatureCache, selection_horizon: int, out_dir: str | Path, gamma: float = 0.99, lr: float = 1e-3, gradient_clip: float = 5.0) -> None:
        self.online_net=online_net; self.target_net=target_net; self.cache=cache; self.selection_horizon=selection_horizon; self.out_dir=Path(out_dir); self.gamma=gamma; self.gradient_clip=gradient_clip
        self.optim=torch.optim.Adam(self.online_net.parameters(), lr=lr); self.replay=C3DQNReplayBuffer(1000); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_net.load_state_dict(self.online_net.state_dict())
    def run_synthetic_smoke_epoch(self) -> dict[str, float]:
        start=time.perf_counter(); k=self.cache.knowledge_dim; samples=[]
        for s in range(8):
            cand=list(range(1, min(self.cache.item_count, 6))); sel=cand[s % len(cand)]
            samples.append({"history_item_ids":[0],"history_responses":[1.0],"candidate_item_ids":cand,"mastery":[0.5]*k,"coverage":[0.0]*k,"policy_step":1,"selected_item_id":sel})
            self.replay.push(C3DQNTransition([0],[1.0],cand,[0.5]*k,[0.0]*k,1,sel,0.1,{"prediction_gain":0.1,"diagnosis_gain":0.0,"coverage_gain":0.0},[0,sel],[1.0,1.0],[x for x in cand if x!=sel],[0.55]*k,[0.1]*k,2,False))
        batch=pad_c3dqn_batch(samples,self.cache,self.selection_horizon); next_batch=pad_c3dqn_batch([{**x,"history_item_ids":[0,x["selected_item_id"]],"history_responses":[1.0,1.0],"candidate_item_ids":[c for c in x["candidate_item_ids"] if c!=x["selected_item_id"]] or [x["selected_item_id"]],"selected_item_id":([c for c in x["candidate_item_ids"] if c!=x["selected_item_id"]] or [x["selected_item_id"]])[0],"policy_step":2} for x in samples], self.cache, self.selection_horizon)
        loss, stats=compute_double_dqn_loss(self.online_net,self.target_net,batch,next_batch,torch.full((len(samples),),0.1),torch.zeros(len(samples)),self.gamma)
        self.optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip); self.optim.step()
        metrics={"epoch":1,"mean_total_reward":0.1,"mean_prediction_reward":0.1,"mean_diagnosis_reward":0.0,"mean_coverage_reward":0.0,"td_loss":float(loss.item()),"mean_q":stats["mean_q"],"target_q_mean":stats["target_q_mean"],"epsilon":0.0,"replay_size":len(self.replay),"selected_unique_items":len(set(x["selected_item_id"] for x in samples)),"item_exposure_gini":0.0,"validation_query_nll":0.0,"validation_query_auc":0.5,"validation_query_brier":0.25,"validation_mastery_entropy":1.0,"validation_concept_coverage":0.0,"feature_build_seconds":0.0,"alpha_fit_seconds":0.0,"reward_seconds":0.0,"network_forward_seconds":0.0,"network_update_seconds":0.0,"validation_seconds":0.0,"total_epoch_seconds":time.perf_counter()-start}
        with (self.out_dir/"training_history.csv").open("w", newline="") as f: w=csv.DictWriter(f, fieldnames=list(metrics)); w.writeheader(); w.writerow(metrics)
        torch.save({"model_state_dict":self.online_net.state_dict(),"metadata":build_checkpoint_metadata(knowledge_dim=k,selection_horizon=self.selection_horizon,warm_start_items=1,alpha_fit={"steps":8,"lr":0.05,"early_stop_tol":1e-5,"warm_start_from_previous_alpha":False},reward_config={},model_config={},candidate_pool_config={"max_candidates":None,"prefilter_enabled":False,"prefilter_top_k":256,"prefilter_mode":"diagnostic_heuristic"},ncdm_item_count=self.cache.item_count,q_matrix_item_count=self.cache.q_matrix.shape[0],training_seed=0,validation_metrics=metrics,epoch=1)}, self.out_dir/"best_checkpoint.pt")
        return metrics
