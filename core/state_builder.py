"""Utilities for converting interaction history into numeric CAT/RL state."""
from typing import Iterable, Tuple

import numpy as np


def build_mastery_state(history: Iterable[Tuple[int, float]], q_matrix, knowledge_dim=None):
    q = np.asarray(q_matrix)
    dim = int(knowledge_dim or q.shape[1])
    state = np.zeros(dim, dtype=float)
    counts = np.zeros(dim, dtype=float)
    for item_id, response in history:
        if 0 <= int(item_id) < len(q):
            mask = q[int(item_id)][:dim]
            state += float(response) * mask
            counts += mask
    return np.divide(state, np.maximum(counts, 1.0))
