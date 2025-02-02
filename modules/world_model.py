# modules/world_model.py

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.distributions import Normal, Independent
from .utils import symlog, symexp, twohot_encode


class Encoder(nn.Module):
    """Multi-Layer Perceptron encoder for processing vector observations."""
    def __init__(self, input_dimension, hidden_dimension=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ELU()
        )

    def forward(self, observation):
        return self.network(observation)


class Decoder(nn.Module):
    """Multi-Layer Perceptron decoder for reconstructing observations from latent features."""
    def __init__(self, feature_dimension, output_dimension, hidden_dimension=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.ELU(),
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, feature):
        return self.network(feature)


class GRURecurrentStateSpaceModel(nn.Module):
    """
    Recurrent State Space Model with GRU-based deterministic path and stochastic latent variables.
    Combines deterministic and stochastic components for temporal state transitions.
    """
    def __init__(self, stochastic_dimension=32, deterministic_dimension=256, 
                 hidden_dimension=256, free_bits=1.0):
        super().__init__()
        self.stochastic_dimension = stochastic_dimension
        self.deterministic_dimension = deterministic_dimension
        self.free_bits = free_bits

        # Gated Recurrent Unit for deterministic state transitions
        self.gated_recurrent_unit = nn.GRUCell(hidden_dimension, deterministic_dimension)

        # Prior network components
        self.prior_network = nn.Sequential(
            nn.Linear(deterministic_dimension, hidden_dimension),
            nn.ELU(),
            nn.Linear(hidden_dimension, 2 * stochastic_dimension)
        )

        # Posterior network components
        self.posterior_network = nn.Sequential(
            nn.Linear(deterministic_dimension + hidden_dimension, hidden_dimension),
            nn.ELU(),
            nn.Linear(hidden_dimension, 2 * stochastic_dimension)
        )

    def forward(self, previous_state, encoded_observation, encoded_action):
        """
        Single-step state transition with KL divergence calculation.
        Args:
            previous_state: Dictionary containing:
                - 'deterministic': [batch_size, deterministic_dimension]
                - 'stochastic': [batch_size, stochastic_dimension]
            encoded_observation: [batch_size, hidden_dimension] from encoder
            encoded_action: [batch_size, hidden_dimension] from action embedding
        Returns: Updated state dictionary and KL divergence
        """
        combined_input = encoded_observation + encoded_action
        deterministic_state = self.gated_recurrent_unit(
            combined_input, previous_state['deterministic']
        )

        # Prior distribution calculations
        prior_parameters = self.prior_network(deterministic_state)
        prior_mean, prior_standard_deviation = torch.chunk(prior_parameters, 2, dim=-1)
        prior_standard_deviation = torch_functional.softplus(prior_standard_deviation) + 0.1
        prior_distribution = Normal(prior_mean, prior_standard_deviation)

        # Posterior distribution calculations
        posterior_input = torch.cat([deterministic_state, encoded_observation], dim=-1)
        posterior_parameters = self.posterior_network(posterior_input)
        posterior_mean, posterior_standard_deviation = torch.chunk(posterior_parameters, 2, dim=-1)
        posterior_standard_deviation = torch_functional.softplus(posterior_standard_deviation) + 0.1
        posterior_distribution = Normal(posterior_mean, posterior_standard_deviation)

        stochastic_state = posterior_distribution.rsample()

        # Kullback-Leibler divergence calculation with free bits constraint
        kullback_leibler_divergence = torch.distributions.kl.kl_divergence(
            posterior_distribution, prior_distribution
        )
        kullback_leibler_divergence = torch.clamp(
            kullback_leibler_divergence, min=self.free_bits
        ).mean()

        updated_state = {
            'deterministic': deterministic_state,
            'stochastic': stochastic_state
        }
        return updated_state, kullback_leibler_divergence


class RewardPredictor(nn.Module):
    """Predicts reward values using two-hot encoded targets."""
    def __init__(self, feature_dimension, number_of_bins=255):
        super().__init__()
        self.number_of_bins = number_of_bins
        self.reward_bins = nn.Parameter(
            torch.linspace(-20, 20, number_of_bins), requires_grad=False
        )
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, 256),
            nn.ELU(),
            nn.Linear(256, number_of_bins)
        )

    def forward(self, feature):
        return self.network(feature)


class ContinuationPredictor(nn.Module):
    """Predicts episode continuation probability (0-1 range)."""
    def __init__(self, feature_dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, feature):
        return torch.sigmoid(self.network(feature))


class WorldModel(nn.Module):
    """
    Comprehensive world model architecture integrating components for:
    - Observation encoding/decoding
    - Temporal state transitions
    - Reward prediction
    - Continuation prediction
    """
    def __init__(self, observation_dimension, action_dimension, 
                 stochastic_dimension=32, deterministic_dimension=256, 
                 hidden_dimension=256, free_bits=1.0):
        super().__init__()
        self.observation_dimension = observation_dimension
        self.action_dimension = action_dimension
        self.stochastic_dimension = stochastic_dimension
        self.deterministic_dimension = deterministic_dimension

        # Core components initialization
        self.encoder = Encoder(observation_dimension, hidden_dimension)
        self.decoder = Decoder(
            deterministic_dimension + stochastic_dimension, 
            observation_dimension, 
            hidden_dimension
        )
        self.recurrent_state_space_model = GRURecurrentStateSpaceModel(
            stochastic_dimension, 
            deterministic_dimension, 
            hidden_dimension, 
            free_bits
        )

        # Action processing components
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dimension, hidden_dimension),
            nn.ELU()
        )

        # Predictive components
        self.reward_predictor = RewardPredictor(deterministic_dimension + stochastic_dimension)
        self.continuation_predictor = ContinuationPredictor(deterministic_dimension + stochastic_dimension)

    def get_initial_state(self, batch_size):
        """Returns zero-initialized state tensors."""
        return {
            'deterministic': torch.zeros(batch_size, self.recurrent_state_space_model.deterministic_dimension),
            'stochastic': torch.zeros(batch_size, self.recurrent_state_space_model.stochastic_dimension)
        }

    def calculate_sequence_loss(self, observation_sequence, action_sequence, 
                               reward_sequence, termination_sequence):
        """
        Computes multi-step losses for world model training.
        Args:
            observation_sequence: [batch_size, sequence_length, observation_dimension]
            action_sequence: [batch_size, sequence_length, action_dimension]
            reward_sequence: [batch_size, sequence_length]
            termination_sequence: [batch_size, sequence_length]
        Returns: Dictionary of component losses and total loss
        """
        batch_size, sequence_length, observation_dimension = observation_sequence.shape
        current_state = self.get_initial_state(batch_size).to(observation_sequence.device)

        # Initialize loss accumulators
        total_divergence = 0.0
        total_reconstruction = 0.0
        total_reward_prediction = 0.0
        total_continuation_prediction = 0.0

        for step in range(sequence_length):
            current_observation = observation_sequence[:, step]
            current_action = action_sequence[:, step]
            current_reward = reward_sequence[:, step]
            current_termination = termination_sequence[:, step]

            # Process inputs
            encoded_observation = self.encoder(current_observation)
            encoded_action = self.action_embedder(current_action)

            # State transition
            next_state, step_divergence = self.recurrent_state_space_model(
                current_state, encoded_observation, encoded_action
            )
            latent_feature = torch.cat([
                next_state['deterministic'], 
                next_state['stochastic']
            ], dim=-1)

            # Reconstruction loss
            reconstructed_observation = self.decoder(latent_feature)
            reconstruction_loss = torch_functional.mse_loss(
                reconstructed_observation, current_observation
            )

            # Reward prediction loss
            reward_logits = self.reward_predictor(latent_feature)
            log_transformed_reward = symlog(current_reward)
            two_hot_targets = twohot_encode(
                log_transformed_reward, 
                self.reward_predictor.reward_bins
            )
            reward_loss = -torch.sum(
                two_hot_targets * torch_functional.log_softmax(reward_logits, dim=-1), 
                dim=-1
            ).mean()

            # Continuation prediction loss
            continuation_probability = self.continuation_predictor(latent_feature)
            continuation_loss = torch_functional.binary_cross_entropy(
                continuation_probability, 
                1 - current_termination.unsqueeze(-1)
            )

            # Update accumulators
            total_divergence += step_divergence
            total_reconstruction += reconstruction_loss
            total_reward_prediction += reward_loss
            total_continuation_prediction += continuation_loss

            # Advance state
            current_state = next_state

        # Average losses over sequence
        average_divergence = total_divergence / sequence_length
        average_reconstruction = total_reconstruction / sequence_length
        average_reward_prediction = total_reward_prediction / sequence_length
        average_continuation_prediction = total_continuation_prediction / sequence_length

        combined_loss = (average_divergence + average_reconstruction 
                        + average_reward_prediction + average_continuation_prediction)

        return {
            'total_loss': combined_loss,
            'state_divergence': average_divergence,
            'reconstruction_loss': average_reconstruction,
            'reward_prediction_loss': average_reward_prediction,
            'continuation_prediction_loss': average_continuation_prediction
        }

    @torch.no_grad()
    def rollout_latent_sequence(self, observation_sequence, action_sequence):
        """
        Generates latent state sequence from real observations and actions.
        Returns: List of latent features for each timestep
        """
        batch_size, sequence_length, _ = observation_sequence.shape
        current_state = self.get_initial_state(batch_size).to(observation_sequence.device)
        feature_sequence = []

        for step in range(sequence_length):
            encoded_observation = self.encoder(observation_sequence[:, step])
            encoded_action = self.action_embedder(action_sequence[:, step])
            next_state, _ = self.recurrent_state_space_model(
                current_state, encoded_observation, encoded_action
            )
            latent_feature = torch.cat([
                next_state['deterministic'], 
                next_state['stochastic']
            ], dim=-1)
            feature_sequence.append(latent_feature)
            current_state = next_state

        return feature_sequence

    @torch.no_grad()
    def imagine_trajectory(self, initial_state, behavior_policy, 
                          prediction_horizon, discrete_actions=True):
        """
        Generates imagined trajectory from initial state using specified policy.
        Returns: List of predicted states, actions, rewards, and continuations
        """
        trajectory = []
        current_state = {
            'deterministic': initial_state['deterministic'].clone(),
            'stochastic': initial_state['stochastic'].clone()
        }
        batch_size = current_state['deterministic'].shape[0]

        for step in range(prediction_horizon):
            latent_feature = torch.cat([
                current_state['deterministic'], 
                current_state['stochastic']
            ], dim=-1)

            # Action selection
            if discrete_actions:
                policy_logits = behavior_policy(latent_feature)
                action_distribution = torch.distributions.Categorical(logits=policy_logits)
                selected_action = action_distribution.sample()
                action_representation = torch_functional.one_hot(
                    selected_action, 
                    behavior_policy.logits.out_features
                ).float()
                encoded_action = self.action_embedder(action_representation)
            else:
                mean, log_scale = behavior_policy(latent_feature)
                scale = torch.exp(log_scale)
                action_distribution = torch.distributions.Normal(mean, scale)
                selected_action = action_distribution.sample()
                encoded_action = self.action_embedder(selected_action)

            # State prediction
            phantom_observation = torch.zeros_like(encoded_action)
            next_state, _ = self.recurrent_state_space_model(
                current_state, phantom_observation, encoded_action
            )
            next_latent_feature = torch.cat([
                next_state['deterministic'], 
                next_state['stochastic']
            ], dim=-1)

            # Reward prediction
            reward_logits = self.reward_predictor(next_latent_feature)
            reward_probabilities = torch_functional.softmax(reward_logits, dim=-1)
            predicted_reward = torch.sum(
                reward_probabilities * self.reward_predictor.reward_bins, 
                dim=-1
            )

            # Continuation prediction
            continuation_probability = self.continuation_predictor(next_latent_feature).squeeze(-1)

            trajectory.append({
                'latent_feature': next_latent_feature,
                'selected_action': selected_action,
                'predicted_reward': predicted_reward,
                'continuation_probability': continuation_probability
            })

            current_state = next_state

        return trajectory