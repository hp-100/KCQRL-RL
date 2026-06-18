"""Replay buffer for DDPG transitions."""
from __future__ import annotations
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 42):
        self.memory = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.memory)

    def push(self, *transition):
        self.memory.append(tuple(transition))

    def sample(self, batch_size: int, device):
        batch = self.rng.sample(self.memory, batch_size)
        cols = list(zip(*batch))
        tensors = []
        for i, col in enumerate(cols):
            dtype = torch.float32
            arr = np.array(col)
            tensors.append(torch.tensor(arr, dtype=dtype, device=device))
        return tensors
