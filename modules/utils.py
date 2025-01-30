# modules/utils.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# -------------------------------------------------------------------------
#  Replay Buffer for Full Episodes (or partial) with sequence sampling
# -------------------------------------------------------------------------
class SequenceReplayBuffer:
    """
    Stores full episodes in a list, then samples multi-step slices (seq_length)
    for Dreamer-like RSSM training.
    """
    def __init__(self, capacity=1000, seq_length=50):
        self.capacity = capacity
        self.seq_length = seq_length
        self.buffer = []
        self.position = 0

    def store_episode(self, ep_obs, ep_actions, ep_rewards, ep_dones):
        """
        Store a complete episode in the buffer.
        """
        episode = {
            'obs': ep_obs,
            'actions': ep_actions,
            'rewards': ep_rewards,
            'dones': ep_dones
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(episode)
        else:
            self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of episodes, then randomly pick a seq_length slice from each.
        Returns: obs [B, L, obs_dim], actions [B, L, ...], rewards [B, L], dones [B, L]
        """
        valid_episodes = [ep for ep in self.buffer if ep is not None]
        episodes = np.random.choice(valid_episodes, size=batch_size)

        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for ep in episodes:
            ep_len = len(ep['obs'])
            # If ep_len < seq_length, just start at 0 or skip
            if ep_len <= self.seq_length:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, ep_len - self.seq_length)

            seq_obs = ep['obs'][start_idx : start_idx + self.seq_length]
            seq_actions = ep['actions'][start_idx : start_idx + self.seq_length]
            seq_rewards = ep['rewards'][start_idx : start_idx + self.seq_length]
            seq_dones   = ep['dones'][start_idx : start_idx + self.seq_length]

            obs_batch.append(seq_obs)
            act_batch.append(seq_actions)
            rew_batch.append(seq_rewards)
            done_batch.append(seq_dones)

        obs_batch = torch.FloatTensor(np.array(obs_batch))
        act_batch = torch.FloatTensor(np.array(act_batch))
        rew_batch = torch.FloatTensor(np.array(rew_batch))
        done_batch= torch.FloatTensor(np.array(done_batch))

        return obs_batch, act_batch, rew_batch, done_batch

    def __len__(self):
        return len([ep for ep in self.buffer if ep is not None])


# -------------------------------------------------------------------------
#  Running Mean/Std for Return Normalization
# -------------------------------------------------------------------------
class RunningMeanStd:
    def __init__(self, epsilon=1e-5):
        self.mean = 0
        self.var = 1
        self.count = 0
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.numel()

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (total_count + 1e-8)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (total_count + 1e-8)
        new_var = M2 / (total_count + 1e-8)

        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, x: torch.Tensor):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


# -------------------------------------------------------------------------
#  Symlog & Two-Hot
# -------------------------------------------------------------------------
def symlog(x):
    """Symmetric log transform used in Dreamer."""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def twohot_encode(x, bins):
    """Two-hot encoding for reward as used by Dreamer."""
    # x: shape [B], bins: shape [num_bins]
    distances = torch.abs(x.unsqueeze(-1) - bins)
    # Typically scale distance to get sharp twohot, e.g. * 10
    weights = torch.softmax(-distances * 10.0, dim=-1)
    return weights
