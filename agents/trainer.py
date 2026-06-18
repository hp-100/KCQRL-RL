"""Real DDPG training pipeline for KCQRL-RL."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from agents.replay_buffer import ReplayBuffer
from core.state_builder import candidate_item_vectors, clean_sequence, find_sequence_columns
from models.ddpg import DDPGAgent, soft_update
from models.ncdm import OfficialNCDM, fit_student_alpha, load_q_matrix, predict_remaining, safe_load_ncdm_checkpoint
from reward.reward_fn import entropy_from_predictions, information_gain_reward


class DDPGTrainer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config; self.device = device
        self.seed = int(config.get("seed", 42)); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.paths = self._paths(config.get("assets", {}) or {})
        tc = config.get("training", {}) or {}
        self.episodes = int(tc.get("episodes", 5)); self.batch_size = int(tc.get("batch_size", 64))
        self.gamma = float(tc.get("gamma", 0.99)); self.tau = float(tc.get("tau", 0.005))
        self.horizon = int(tc.get("horizon", 10)); self.max_students = int(tc.get("max_students", 300))
        self.output_dir = Path(tc.get("output_dir", "outputs")); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = ReplayBuffer(int(tc.get("replay_capacity", 20000)), self.seed)

    @staticmethod
    def _paths(asset_cfg):
        base = Path(asset_cfg.get("base_dir", ".")).expanduser()
        return {k: (Path(v).expanduser() if Path(str(v)).is_absolute() else base / str(v)) for k, v in asset_cfg.items() if k != "base_dir" and v is not None}

    def missing_assets(self):
        required = ["q_matrix", "item_bank", "ncdm_checkpoint", "train_sequences"]
        return [self.paths[k] for k in required if k not in self.paths or not self.paths[k].exists()]

    def _load(self):
        q_matrix = load_q_matrix(self.paths["q_matrix"], self.device)
        item_bank = torch.tensor(np.load(self.paths["item_bank"]), dtype=torch.float32, device=self.device)
        item_bank = nn.functional.normalize(item_bank, p=2, dim=1)
        max_item = min(q_matrix.shape[0], item_bank.shape[0])
        df = pd.read_csv(self.paths["train_sequences"])
        q_col, r_col = find_sequence_columns(df)
        ncdm = OfficialNCDM(max(1, len(df)), q_matrix.shape[0], q_matrix.shape[1]).to(self.device)
        safe_load_ncdm_checkpoint(ncdm, self.paths["ncdm_checkpoint"], self.device)
        ncdm.eval()
        for p in ncdm.parameters(): p.requires_grad = False
        agent = DDPGAgent(q_dim=q_matrix.shape[1], semantic_dim=item_bank.shape[1], device=self.device)
        return q_matrix, item_bank, max_item, df.sample(n=min(self.max_students, len(df)), random_state=self.seed), q_col, r_col, ncdm, agent

    def select_candidate(self, ideal, avail_i, q_matrix, ncdm):
        cand = candidate_item_vectors(avail_i, q_matrix, ncdm)
        return int(torch.argmin(torch.cdist(ideal, cand).squeeze(0)).item())

    def _entropy(self, ncdm, q_matrix, hist_i, hist_r, targets):
        if not targets: return 0.0
        return float(entropy_from_predictions(predict_remaining(ncdm, q_matrix, hist_i, hist_r, targets, device=self.device)).item())

    def optimize(self, agent: DDPGAgent):
        if len(self.buffer) <= self.batch_size: return None
        b = self.buffer.sample(self.batch_size, self.device)
        h0,c0,sem,q,diff,disc,resp,act,rew,nsem,nq,ndiff,ndisc,nresp,done = b
        rew = rew.unsqueeze(1); done = done.unsqueeze(1)
        ideal, h1, c1 = agent.actor(sem, q, diff, disc, resp, h0, c0)
        with torch.no_grad():
            nideal, nh, _ = agent.target_actor(nsem, nq, ndiff, ndisc, nresp, h1.detach(), c1.detach())
            target_q = rew + self.gamma * agent.target_critic(nh, nideal) * (1.0 - done)
        critic_loss = nn.SmoothL1Loss()(agent.critic(h1.detach(), act), target_q)
        agent.critic_optimizer.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0); agent.critic_optimizer.step()
        actor_loss = -agent.critic(h1, ideal).mean()
        agent.actor_optimizer.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0); agent.actor_optimizer.step()
        soft_update(agent.actor, agent.target_actor, self.tau); soft_update(agent.critic, agent.target_critic, self.tau)
        return float(actor_loss.item()), float(critic_loss.item())

    def train(self):
        q_matrix, item_bank, max_item, rows, q_col, r_col, ncdm, agent = self._load()
        logs = []
        for epoch in range(self.episodes):
            rewards=[]; aloss=[]; closs=[]; noise_std=max(0.05, 0.5 - epoch * 0.05)
            for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"DDPG epoch {epoch+1}/{self.episodes}"):
                items, responses = clean_sequence(row[q_col], row[r_col], max_item)
                if len(items) < 4: continue
                order = list(range(len(items))); random.shuffle(order); split = max(1, int(len(order)*0.7))
                avail_i=[items[i] for i in order[:split]]; avail_r=[responses[i] for i in order[:split]]; val_i=[items[i] for i in order[split:]]
                seed_idx=random.randrange(len(avail_i)); cur_i=avail_i.pop(seed_idx); cur_r=avail_r.pop(seed_idx)
                hist_i=[cur_i]; hist_r=[cur_r]; hx,cx=agent.actor.init_hidden(1,self.device)
                for step in range(self.horizon):
                    if not avail_i: break
                    prev_ent=self._entropy(ncdm,q_matrix,hist_i,hist_r,val_i)
                    tid=torch.tensor([cur_i],device=self.device)
                    sem=item_bank[cur_i]; q=q_matrix[cur_i]; diff=torch.sigmoid(ncdm.k_difficulty(tid)).squeeze(0); disc=torch.sigmoid(ncdm.e_discrimination(tid)).squeeze(0)
                    with torch.no_grad():
                        ideal,nh,nc=agent.actor(sem.unsqueeze(0),q.unsqueeze(0),diff.unsqueeze(0),disc.unsqueeze(0),torch.tensor([cur_r],device=self.device),hx,cx)
                        noisy=torch.clamp(ideal + torch.randn_like(ideal)*noise_std,0,1)
                    loc=self.select_candidate(noisy,avail_i,q_matrix,ncdm); next_i=avail_i.pop(loc); next_r=avail_r.pop(loc)
                    hist_i.append(next_i); hist_r.append(next_r)
                    curr_ent=self._entropy(ncdm,q_matrix,hist_i,hist_r,val_i); reward=information_gain_reward(prev_ent,curr_ent); rewards.append(reward)
                    nt=torch.tensor([next_i],device=self.device); ndiff=torch.sigmoid(ncdm.k_difficulty(nt)).squeeze(0); ndisc=torch.sigmoid(ncdm.e_discrimination(nt)).squeeze(0)
                    done=float(step == self.horizon-1 or not avail_i)
                    self.buffer.push(hx.squeeze(0).detach().cpu().numpy(), cx.squeeze(0).detach().cpu().numpy(), sem.cpu().numpy(), q.cpu().numpy(), diff.cpu().numpy(), disc.cpu().numpy(), cur_r, noisy.squeeze(0).cpu().numpy(), reward, item_bank[next_i].cpu().numpy(), q_matrix[next_i].cpu().numpy(), ndiff.cpu().numpy(), ndisc.cpu().numpy(), next_r, done)
                    hx,cx=nh.detach(),nc.detach(); cur_i,cur_r=next_i,next_r
                    metrics=self.optimize(agent)
                    if metrics: aloss.append(metrics[0]); closs.append(metrics[1])
                    if done: break
            log={"epoch":epoch+1,"reward":float(np.mean(rewards)) if rewards else 0.0,"actor_loss":float(np.mean(aloss)) if aloss else 0.0,"critic_loss":float(np.mean(closs)) if closs else 0.0}
            logs.append(log); print(f"Epoch {log['epoch']} reward={log['reward']:.4f} actor_loss={log['actor_loss']:.4f} critic_loss={log['critic_loss']:.4f}")
        out=self.output_dir/"ddpg_actor.pt"; torch.save(agent.actor.state_dict(), out)
        return logs, out
