# modules/world_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from .utils import symlog, symexp, twohot_encode


class Encoder(nn.Module):
    """Simple MLP encoder for vector observations."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )

    def forward(self, obs):
        # Optionally apply symlog to obs if it spans large ranges
        # obs = symlog(obs)
        return self.net(obs)


class Decoder(nn.Module):
    """Simple MLP decoder to reconstruct observations from latents."""
    def __init__(self, feat_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feat):
        # Output is a direct reconstruction, or symexp if you used symlog
        x = self.net(feat)
        return x  # or symexp(x)


class GRURSSM(nn.Module):
    """
    GRU-based RSSM with a deterministic (GRU) state and a stochastic state
    for each timestep.
    """
    def __init__(self, stoch_dim=32, deter_dim=256, hidden_dim=256, free_bits=1.0):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.free_bits = free_bits

        # GRU for deterministic path
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior MLP
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )

        # Posterior MLP
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )

    def forward(self, prev_state, obs_emb, act_emb):
        """
        Single-step RSSM update.
        Args:
          prev_state: dict with 'deter' [B, deter_dim], 'stoch' [B, stoch_dim]
          obs_emb: [B, hidden_dim] from encoder
          act_emb: [B, hidden_dim] from action
        Returns: next_state, kl
        """
        x = obs_emb + act_emb
        deter_state = self.gru(x, prev_state['deter'])

        # prior
        prior_params = self.prior_net(deter_state)
        prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 0.1
        prior_dist = Normal(prior_mean, prior_std)

        # posterior
        post_input = torch.cat([deter_state, obs_emb], dim=-1)
        post_params = self.post_net(post_input)
        post_mean, post_std = torch.chunk(post_params, 2, dim=-1)
        post_std = F.softplus(post_std) + 0.1
        post_dist = Normal(post_mean, post_std)

        stoch_state = post_dist.rsample()

        # KL with free bits
        kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        kl = torch.clamp(kl, min=self.free_bits).mean()

        next_state = {
            'deter': deter_state,
            'stoch': stoch_state
        }
        return next_state, kl


class RewardModel(nn.Module):
    """Predict reward (using two-hot if desired)."""
    def __init__(self, feat_dim, num_bins=255):
        super().__init__()
        self.num_bins = num_bins
        self.bins = nn.Parameter(torch.linspace(-20, 20, num_bins), requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Linear(256, num_bins)
        )

    def forward(self, feat):
        logits = self.net(feat)
        return logits


class ContinueModel(nn.Module):
    """Predict discount/continue (between 0 and 1)."""
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))


class WorldModel(nn.Module):
    """
    Full Dreamer-style world model:
      - Encoder
      - RSSM
      - Decoder (reconstruction)
      - RewardModel (twohot)
      - ContinueModel (if you want discount prediction)
    Provides:
      - sequence_loss(...) for multi-step training
      - rollout_sequence(...) to get latent states from real data
      - imagine_ahead(...) to produce imaginary trajectories
    """
    def __init__(self, obs_dim, action_dim, stoch_dim=32, deter_dim=256, hidden_dim=256, free_bits=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        self.encoder = Encoder(obs_dim, hidden_dim)
        self.decoder = Decoder(deter_dim + stoch_dim, obs_dim, hidden_dim)
        self.rssm = GRURSSM(stoch_dim, deter_dim, hidden_dim, free_bits)

        # For actions, we embed them to match the size used in GRU
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ELU()
        )

        # Reward & Continue
        self.reward_model = RewardModel(deter_dim + stoch_dim)
        self.continue_model = ContinueModel(deter_dim + stoch_dim)

    def get_initial_state(self, batch_size):
        return {
            'deter': torch.zeros(batch_size, self.rssm.deter_dim),
            'stoch': torch.zeros(batch_size, self.rssm.stoch_dim)
        }

    def sequence_loss(self, obs_seq, act_seq, rew_seq, done_seq, free_bits=1.0):
        """
        Unroll the RSSM for the entire sequence [B, L].
        Compute reconstruction, reward, continuation, KL losses.
        Return a dict of losses. 
        """
        B, L, obs_dim = obs_seq.shape
        state = self.get_initial_state(B).to(obs_seq.device)

        total_kl, total_recon, total_reward, total_cont = 0, 0, 0, 0

        for t in range(L):
            obs_t = obs_seq[:, t]
            act_t = act_seq[:, t]
            rew_t = rew_seq[:, t]
            done_t= done_seq[:, t]

            # Encode observation
            obs_emb = self.encoder(obs_t)
            # Action embed
            act_emb = self.action_embed(act_t)

            # RSSM update
            next_state, kl = self.rssm(state, obs_emb, act_emb)
            feat = torch.cat([next_state['deter'], next_state['stoch']], dim=-1)

            # Decoder recon
            recon = self.decoder(feat)
            recon_loss = F.mse_loss(recon, obs_t)

            # Reward (twohot)
            reward_logits = self.reward_model(feat)
            # Convert reward to symlog, then twohot
            rew_symlog = symlog(rew_t)
            twohot_target = twohot_encode(rew_symlog, self.reward_model.bins)
            reward_loss = -torch.sum(twohot_target * F.log_softmax(reward_logits, dim=-1), dim=-1).mean()

            # Continue
            cont_pred = self.continue_model(feat)
            cont_loss = F.binary_cross_entropy(cont_pred, 1 - done_t.unsqueeze(-1))

            total_kl += kl
            total_recon += recon_loss
            total_reward += reward_loss
            total_cont += cont_loss

            # Move forward
            state = next_state

        total_kl     /= L
        total_recon  /= L
        total_reward /= L
        total_cont   /= L

        total_loss = total_kl + total_recon + total_reward + total_cont
        return {
            'total': total_loss,
            'kl': total_kl,
            'recon': total_recon,
            'reward': total_reward,
            'cont': total_cont
        }

    @torch.no_grad()
    def rollout_sequence(self, obs_seq, act_seq):
        """
        Roll out RSSM over a real sequence (obs_seq, act_seq),
        return the list of final latent features at each time step.
        """
        B, L, _ = obs_seq.shape
        state = self.get_initial_state(B).to(obs_seq.device)
        feat_seq = []

        for t in range(L):
            obs_emb = self.encoder(obs_seq[:, t])
            act_emb = self.action_embed(act_seq[:, t])
            next_state, _ = self.rssm(state, obs_emb, act_emb)
            feat = torch.cat([next_state['deter'], next_state['stoch']], dim=-1)
            feat_seq.append(feat)
            state = next_state

        return feat_seq  # list of length L, each [B, feat_dim]

    @torch.no_grad()
    def imagine_ahead(self, start_state, actor, horizon, discrete_actions=True):
        """
        From a given latent state (deter+stoch),
        imagine a trajectory for 'horizon' steps by sampling actions from 'actor'.
        Returns a list of dicts: [{feat, action, reward, cont}, ...].
        """
        traj = []
        state = {
            'deter': start_state['deter'].clone(),
            'stoch': start_state['stoch'].clone()
        }
        B = state['deter'].shape[0]

        for t in range(horizon):
            feat = torch.cat([state['deter'], state['stoch']], dim=-1)

            # Sample action from actor
            if discrete_actions:
                logits = actor(feat)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                # One-hot for RSSM
                action_oh = F.one_hot(action, actor.logits.out_features).float()
                act_emb = self.action_embed(action_oh)
            else:
                mu, log_std = actor(feat)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                act_emb = self.action_embed(action)

            # RSSM forward (no real obs => we feed 0 or a learned 'phantom' observation embedding)
            obs_emb = torch.zeros_like(act_emb)  # or a learned phantom param
            next_state, _ = self.rssm(state, obs_emb, act_emb)
            next_feat = torch.cat([next_state['deter'], next_state['stoch']], dim=-1)

            # Predict reward
            reward_logits = self.reward_model(next_feat)
            # We can approximate reward by the expected value under two-hot
            # Or sample. We'll do 'expected reward' by weighting bins
            probs = F.softmax(reward_logits, dim=-1)
            bin_values = self.reward_model.bins
            reward_pred = torch.sum(probs * bin_values, dim=-1)  # shape [B]

            # Predict discount
            cont_pred = self.continue_model(next_feat).squeeze(-1)

            traj.append({
                'feat': next_feat,
                'action': action,   # shape [B]
                'reward': reward_pred,  # shape [B]
                'cont': cont_pred       # shape [B]
            })

            state = next_state

        return traj
