from __future__ import annotations
from pathlib import Path
import torch
from .base import BaseCATPolicy, PolicyMetadata

class DDPGPolicy(BaseCATPolicy):
    name = "DDPG"
    metadata = PolicyMetadata(name=name, implementation="checkpoint")
    def __init__(self, checkpoint: str | Path, actor=None, q_matrix=None, item_bank=None, ncdm=None, device="cpu", allow_debug_fallback: bool=False):
        self.checkpoint = Path(checkpoint)
        self.actor = actor
        self.q_matrix = q_matrix
        self.item_bank = item_bank
        self.ncdm = ncdm
        self.device = device
        self.allow_debug_fallback = allow_debug_fallback
        if actor is None and not self.checkpoint.exists():
            if allow_debug_fallback:
                self.metadata = PolicyMetadata(name=self.name, implementation="explicit_debug_fallback", notes=f"Checkpoint missing: {self.checkpoint}")
            else:
                raise FileNotFoundError(f"DDPG actor checkpoint not found: {self.checkpoint}")
    def reset(self, student_id, seed, context):
        super().reset(student_id, seed, context)
        self.hx = self.cx = None
    def select(self, candidate_item_ids, history_item_ids, history_responses, context):
        if self.actor is None:
            return int(list(candidate_item_ids)[0])
        # nearest-neighbor mapping from actor's 73-d ideal point
        with torch.no_grad():
            if self.hx is None or self.cx is None:
                self.hx, self.cx = self.actor.init_hidden(1, self.device)
            if history_item_ids:
                last = int(history_item_ids[-1]); resp = float(history_responses[-1])
                sem = self.item_bank[last].unsqueeze(0); q = self.q_matrix[last].unsqueeze(0)
                tid = torch.tensor([last], dtype=torch.long, device=self.device)
                diff = torch.sigmoid(self.ncdm.k_difficulty(tid)); disc = torch.sigmoid(self.ncdm.e_discrimination(tid))
                rv = torch.tensor([resp], dtype=torch.float32, device=self.device)
            else:
                sem = torch.zeros((1, self.item_bank.shape[1]), device=self.device); q = torch.zeros((1, self.q_matrix.shape[1]), device=self.device)
                diff = torch.zeros_like(q); disc = torch.zeros((1,1), device=self.device); rv = torch.zeros(1, device=self.device)
            ideal, self.hx, self.cx = self.actor(sem, q, diff, disc, rv, self.hx, self.cx)
            cids = torch.tensor(list(candidate_item_ids), dtype=torch.long, device=self.device)
            cvec = torch.cat([self.q_matrix[cids], torch.sigmoid(self.ncdm.k_difficulty(cids)), torch.sigmoid(self.ncdm.e_discrimination(cids))], dim=-1)
            return int(list(candidate_item_ids)[int(torch.argmin(torch.cdist(ideal, cvec).squeeze(0)).item())])
