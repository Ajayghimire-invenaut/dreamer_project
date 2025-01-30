# modules/agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import SequenceReplayBuffer, RunningMeanStd
from .world_model import WorldModel


class Actor(nn.Module):
    """
    - If discrete_actions=True: outputs logits for Categorical dist
    - If discrete_actions=False: outputs (mu, log_std) for Normal dist
    """
    def __init__(self, latent_dim, action_dim, discrete_actions=True):
        super().__init__()
        self.discrete_actions = discrete_actions
        hidden = 300
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU()
        )
        if self.discrete_actions:
            self.logits = nn.Linear(hidden, action_dim)
        else:
            self.mu = nn.Linear(hidden, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, feat):
        x = self.net(feat)
        if self.discrete_actions:
            return self.logits(x)  # [B, action_dim]
        else:
            mu = self.mu(x)
            log_std = torch.clamp(self.log_std, -5, 2)
            return mu, log_std


class Critic(nn.Module):
    """Value function approximator."""
    def __init__(self, latent_dim):
        super().__init__()
        hidden = 300
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)  # [B]


class DreamerAgent:
    """
    Advanced Dreamer-like agent:
      - Sequence replay
      - Multi-step RSSM training
      - Imagination for actor-critic
    """
    def __init__(self,
                 world_model: WorldModel,
                 obs_dim: int,
                 action_dim: int,
                 discrete_actions: bool = True,
                 seq_length: int = 50,
                 buffer_capacity: int = 1000,
                 batch_size: int = 16,
                 imagination_horizon: int = 15,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 free_bits: float = 1.0,
                 wm_lr: float = 1e-4,
                 actor_lr: float = 4e-5,
                 critic_lr: float = 4e-5,
                 device: str = 'cpu'):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lam = lam
        self.free_bits = free_bits
        self.device = device

        # Replay
        self.replay_buffer = SequenceReplayBuffer(
            capacity=buffer_capacity,
            seq_length=seq_length
        )

        # World Model
        self.world_model = world_model.to(device)

        # Actor & Critic
        latent_dim = world_model.rssm.deter_dim + world_model.rssm.stoch_dim
        self.actor = Actor(latent_dim, action_dim, discrete_actions).to(device)
        self.critic = Critic(latent_dim).to(device)

        # Optimizers
        self.world_opt = optim.Adam(self.world_model.parameters(), lr=wm_lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Return Normalization
        self.ret_rms = RunningMeanStd()

        # Temporary storage for current episode
        self.ep_obs = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_dones = []

        # Epsilon (for discrete)
        self.epsilon = 0.10
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

    def store_transition(self, obs, action, reward, done):
        self.ep_obs.append(obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_dones.append(done)

        if done:
            ep_obs = np.array(self.ep_obs, dtype=np.float32)
            if self.discrete_actions:
                ep_actions = np.array(self.ep_actions, dtype=np.int32)
            else:
                ep_actions = np.array(self.ep_actions, dtype=np.float32)
            ep_rewards = np.array(self.ep_rewards, dtype=np.float32)
            ep_dones = np.array(self.ep_dones, dtype=np.float32)

            self.replay_buffer.store_episode(ep_obs, ep_actions, ep_rewards, ep_dones)

            self.ep_obs.clear()
            self.ep_actions.clear()
            self.ep_rewards.clear()
            self.ep_dones.clear()

    @torch.no_grad()
    def act(self, obs, epsilon=None):
        """
        Single-step action for environment. 
        If discrete: do an epsilon-greedy w.r.t. the actor's logits
        If continuous: sample from Normal, optionally add exploration noise.
        """
        if epsilon is None:
            epsilon = self.epsilon
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Minimal encoding
        if hasattr(self.world_model, 'encoder'):
            obs_emb = self.world_model.encoder(obs_tensor)
            # For a single-step approach, no real state update, just use obs_emb
            # Or skip if you prefer direct obs as input
            feat = obs_emb
        else:
            feat = obs_tensor

        if self.discrete_actions:
            logits = self.actor(feat)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
            return action
        else:
            mu, log_std = self.actor(feat)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            return action.squeeze(0).cpu().numpy()

    def train(self):
        """
        Overall training step:
          1) Train world model
          2) Train actor-critic with imagination
          3) Update epsilon
        """
        if len(self.replay_buffer) < 5:
            return {}

        wm_logs = self.train_world_model()
        ac_logs = self.train_actor_critic()

        # Update epsilon (for discrete envs)
        if self.discrete_actions:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        logs = {**wm_logs, **ac_logs}
        return logs

    def train_world_model(self):
        """
        1) Sample sequences
        2) Compute sequence loss with self.world_model.sequence_loss
        3) Backprop + clip
        """
        obs_seq, act_seq, rew_seq, done_seq = self.replay_buffer.sample(self.batch_size)
        obs_seq   = obs_seq.to(self.device)  # [B, L, obs_dim]
        act_seq   = act_seq.to(self.device)  # [B, L]
        rew_seq   = rew_seq.to(self.device)
        done_seq  = done_seq.to(self.device)

        # Convert discrete actions -> one-hot
        if self.discrete_actions and act_seq.ndim == 2:
            act_seq_oh = F.one_hot(act_seq.long(), self.action_dim).float()
        else:
            act_seq_oh = act_seq

        wm_loss_dict = self.world_model.sequence_loss(obs_seq, act_seq_oh, rew_seq, done_seq, self.free_bits)

        self.world_opt.zero_grad()
        wm_loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_opt.step()

        # Return logs
        return {
            'wm_total': wm_loss_dict['total'].item(),
            'wm_kl': wm_loss_dict['kl'].item(),
            'wm_recon': wm_loss_dict['recon'].item(),
            'wm_reward': wm_loss_dict['reward'].item(),
            'wm_cont': wm_loss_dict['cont'].item() if 'cont' in wm_loss_dict else 0.0
        }

    def train_actor_critic(self):
        """
        1) Sample sequences (unroll real data in RSSM)
        2) Take final latent -> imagine ahead
        3) Compute returns with GAE-lambda or n-step
        4) Optimize actor & critic
        """
        obs_seq, act_seq, rew_seq, done_seq = self.replay_buffer.sample(self.batch_size)
        obs_seq   = obs_seq.to(self.device)
        act_seq   = act_seq.to(self.device)
        rew_seq   = rew_seq.to(self.device)
        done_seq  = done_seq.to(self.device)

        if self.discrete_actions and act_seq.ndim == 2:
            act_seq_oh = F.one_hot(act_seq.long(), self.action_dim).float()
        else:
            act_seq_oh = act_seq

        # 1) Get final latent from real data unroll
        with torch.no_grad():
            feat_seq = self.world_model.rollout_sequence(obs_seq, act_seq_oh)
            final_feat = feat_seq[-1]  # [B, feat_dim]

            final_state = {
                'deter': final_feat[:, :self.world_model.rssm.deter_dim],
                'stoch': final_feat[:, self.world_model.rssm.deter_dim:]
            }

        # 2) Imagination
        imagined_traj = self.world_model.imagine_ahead(final_state, self.actor, self.imagination_horizon, self.discrete_actions)
        # Each element: { feat, action, reward, cont }

        # 3) Critic values & GAE-lambda returns
        for step in imagined_traj:
            step['value'] = self.critic(step['feat'])

        returns, advantages = self.compute_gae(imagined_traj)

        # 4a) Actor update
        actor_loss = 0.0
        for t, step_data in enumerate(imagined_traj):
            feat = step_data['feat']
            action = step_data['action']
            advantage = advantages[t].detach()

            if self.discrete_actions:
                logits = self.actor(feat)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(action)
            else:
                mu, log_std = self.actor(feat)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                # action shape: [B, action_dim]
                logp = dist.log_prob(action).sum(dim=-1)

            actor_loss -= (logp * advantage).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        # 4b) Critic update
        critic_loss = 0.0
        for t, step_data in enumerate(imagined_traj):
            value_pred = step_data['value']
            target = returns[t].detach()
            critic_loss += F.mse_loss(value_pred, target)

        critic_loss /= len(imagined_traj)  # average across horizon

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def compute_gae(self, traj):
        """
        Compute λ-returns (GAE) for the imagined trajectory.
        traj is a list of dicts with keys: feat, action, reward, cont, value
        """
        horizon = len(traj)
        returns = [None] * horizon
        advantages = [None] * horizon

        next_value = traj[-1]['value'].detach()
        next_adv = torch.zeros_like(next_value)

        for t in reversed(range(horizon)):
            reward = traj[t]['reward']  # shape [B]
            cont   = traj[t]['cont'] * self.gamma
            value  = traj[t]['value']
            delta = reward + cont * next_value - value
            adv = delta + cont * self.lam * next_adv
            ret = value + adv

            returns[t] = ret
            advantages[t] = adv

            next_value = value.detach()
            next_adv = adv.detach()

        return returns, advantages
