"""State and item feature builders for the DDPG CAT agent."""
from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
import torch


def build_mastery_state(history: Iterable[Tuple[int, float]], q_matrix, knowledge_dim=None):
    knowledge_dim = knowledge_dim or q_matrix.shape[1]
    mastery = np.zeros(knowledge_dim, dtype=np.float32)
    counts = np.zeros(knowledge_dim, dtype=np.float32)
    for item, response in history:
        q = np.asarray(q_matrix[item], dtype=np.float32)[:knowledge_dim]
        mastery += q * float(response)
        counts += q
    return mastery / np.maximum(counts, 1.0)


def clean_sequence(row_q, row_r, max_item_id: int) -> tuple[list[int], list[float]]:
    q_strs = str(row_q).replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
    r_strs = str(row_r).replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
    items, responses = [], []
    for q, r in zip(q_strs, r_strs):
        if not q.strip() or not r.strip():
            continue
        item = int(float(q.strip()))
        resp = float(r.strip())
        if 0 <= item < max_item_id and resp >= 0.0:
            items.append(item); responses.append(resp)
    return items, responses


def find_sequence_columns(df):
    q_col = next((c for c in df.columns if "question" in c.lower() or "item" in c.lower()), df.columns[1])
    r_col = next((c for c in df.columns if "response" in c.lower() or "score" in c.lower() or "correct" in c.lower()), df.columns[2])
    return q_col, r_col


def candidate_item_vectors(item_ids: Sequence[int], q_matrix: torch.Tensor, ncdm) -> torch.Tensor:
    ids = torch.tensor(item_ids, dtype=torch.long, device=q_matrix.device)
    return torch.cat([q_matrix[ids], torch.sigmoid(ncdm.k_difficulty(ids)), torch.sigmoid(ncdm.e_discrimination(ids))], dim=-1)
