"""Episode-based replay buffer for recurrent off-policy RL."""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch

@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

class SequenceReplayBuffer:
    def __init__(self, capacity_episodes:int=5000, seed:int=42):
        self.capacity_episodes=int(capacity_episodes); self.rng=random.Random(seed); self.episodes=[]
    def __len__(self): return len(self.episodes)
    def add_episode(self, states, actions, rewards, next_states, dones):
        ep=Episode(np.asarray(states,dtype=np.float32), np.asarray(actions,dtype=np.float32), np.asarray(rewards,dtype=np.float32), np.asarray(next_states,dtype=np.float32), np.asarray(dones,dtype=np.float32))
        if ep.states.ndim!=2 or ep.actions.ndim!=2: raise ValueError('episode arrays must be time-major')
        self.episodes.append(ep)
        if len(self.episodes)>self.capacity_episodes: self.episodes.pop(0)
    def sample(self, batch_sequences:int, burn_in_length:int, unroll_length:int, device='cpu'):
        if not self.episodes: raise ValueError('cannot sample empty SequenceReplayBuffer')
        total=int(burn_in_length)+int(unroll_length); batch=[]
        eligible=[ep for ep in self.episodes if len(ep.states)>int(burn_in_length)]
        if not eligible:
            raise ValueError('no SequenceReplayBuffer episodes contain at least one valid unroll timestep')
        for _ in range(int(batch_sequences)):
            ep=self.rng.choice(eligible); T=len(ep.states)
            max_start=max(0, T-int(burn_in_length)-1)
            start=0 if max_start==0 else self.rng.randint(0, max_start)
            end=min(T, start+total); valid=end-start; pad=total-valid
            def sl(x, tail_shape):
                arr=x[start:end]
                if pad: arr=np.concatenate([arr, np.zeros((pad,)+tail_shape,dtype=np.float32)], axis=0)
                return arr
            batch.append((sl(ep.states,(ep.states.shape[1],)), sl(ep.actions,(ep.actions.shape[1],)), sl(ep.rewards.reshape(-1,1),(1,)), sl(ep.next_states,(ep.next_states.shape[1],)), sl(ep.dones.reshape(-1,1),(1,)), np.concatenate([np.ones((valid,1),np.float32), np.zeros((pad,1),np.float32)],0)))
        cols=list(zip(*batch))
        names=['states','actions','rewards','next_states','dones','valid_mask']
        out={n:torch.as_tensor(np.stack(c),device=device,dtype=torch.float32) for n,c in zip(names,cols)}
        if out['valid_mask'][:, int(burn_in_length):].sum().item() == 0:
            raise ValueError('sampled batch has zero valid unroll timesteps')
        return out
