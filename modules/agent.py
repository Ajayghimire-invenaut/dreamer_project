# modules/agent.py

import numpy as numpy
import torch
import torch.nn as neural_network
import torch.nn.functional as neural_functional
import torch.optim as optimization

from .utils import SequenceReplayBuffer, RunningMeanStd
from .world_model import WorldModel


class BehaviorPolicy(neural_network.Module):
    """
    Policy network for action selection.
    - For discrete actions: outputs categorical logits
    - For continuous actions: outputs mean and log standard deviation for normal distribution
    """
    def __init__(self, latent_dimension, action_dimension, discrete_actions=True):
        super().__init__()
        self.discrete_actions = discrete_actions
        hidden_units = 300
        self.network = neural_network.Sequential(
            neural_network.Linear(latent_dimension, hidden_units),
            neural_network.ELU(),
            neural_network.Linear(hidden_units, hidden_units),
            neural_network.ELU()
        )
        if self.discrete_actions:
            self.action_logits = neural_network.Linear(hidden_units, action_dimension)
        else:
            self.action_mean = neural_network.Linear(hidden_units, action_dimension)
            self.log_standard_deviation = neural_network.Parameter(torch.zeros(action_dimension))

    def forward(self, latent_feature):
        processed_feature = self.network(latent_feature)
        if self.discrete_actions:
            return self.action_logits(processed_feature)
        else:
            mean = self.action_mean(processed_feature)
            log_std = torch.clamp(self.log_standard_deviation, -5, 2)
            return mean, log_std


class StateValueEstimator(neural_network.Module):
    """Neural network for estimating state values."""
    def __init__(self, latent_dimension):
        super().__init__()
        hidden_units = 300
        self.network = neural_network.Sequential(
            neural_network.Linear(latent_dimension, hidden_units),
            neural_network.ELU(),
            neural_network.Linear(hidden_units, hidden_units),
            neural_network.ELU(),
            neural_network.Linear(hidden_units, 1)
        )

    def forward(self, latent_feature):
        return self.network(latent_feature).squeeze(-1)


class DreamerLearningAgent:
    """
    Implements the Dreamer algorithm components:
    - Experience replay with sequence storage
    - World model learning
    - Policy and value function optimization through imagined trajectories
    """
    def __init__(
        self,
        world_model: WorldModel,
        observation_dimension: int,
        action_dimension: int,
        discrete_actions: bool = True,
        sequence_length: int = 50,
        replay_capacity: int = 1000,
        training_batch_size: int = 16,
        imagination_depth: int = 15,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        free_information_bits: float = 1.0,
        world_model_learning_rate: float = 1e-4,
        policy_learning_rate: float = 4e-5,
        value_learning_rate: float = 4e-5,
        computation_device: str = 'cpu'
    ):
        self.observation_dimension = observation_dimension
        self.action_dimension = action_dimension
        self.discrete_actions = discrete_actions
        self.sequence_length = sequence_length
        self.training_batch_size = training_batch_size
        self.imagination_depth = imagination_depth
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.free_information_bits = free_information_bits
        self.computation_device = computation_device

        # Experience replay configuration
        self.experience_replay = SequenceReplayBuffer(
            capacity=replay_capacity,
            sequence_length=sequence_length
        )

        # World model initialization
        self.world_model = world_model.to(computation_device)

        # Policy and value networks
        latent_feature_size = world_model.recurrent_state_space_model.deterministic_dimension + \
                            world_model.recurrent_state_space_model.stochastic_dimension
        self.behavior_policy = BehaviorPolicy(
            latent_feature_size, 
            action_dimension, 
            discrete_actions
        ).to(computation_device)
        self.state_value_estimator = StateValueEstimator(latent_feature_size).to(computation_device)

        # Optimization setups
        self.world_model_optimizer = optimization.Adam(
            self.world_model.parameters(), 
            lr=world_model_learning_rate
        )
        self.policy_optimizer = optimization.Adam(
            self.behavior_policy.parameters(), 
            lr=policy_learning_rate
        )
        self.value_optimizer = optimization.Adam(
            self.state_value_estimator.parameters(), 
            lr=value_learning_rate
        )

        # Return normalization
        self.return_normalizer = RunningMeanStd()

        # Episode tracking
        self.current_episode_observations = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_dones = []

        # Exploration parameters
        self.exploration_rate = 0.10
        self.minimum_exploration_rate = 0.01
        self.exploration_decay_factor = 0.999

    def record_experience(self, observation, action, reward, termination):
        """Store transition data in temporary episode storage."""
        self.current_episode_observations.append(observation)
        self.current_episode_actions.append(action)
        self.current_episode_rewards.append(reward)
        self.current_episode_dones.append(termination)

        if termination:
            # Convert and store completed episode
            episode_data = {
                'observations': numpy.array(self.current_episode_observations, dtype=numpy.float32),
                'actions': numpy.array(
                    self.current_episode_actions, 
                    dtype=numpy.int32 if self.discrete_actions else numpy.float32
                ),
                'rewards': numpy.array(self.current_episode_rewards, dtype=numpy.float32),
                'terminations': numpy.array(self.current_episode_dones, dtype=numpy.float32)
            }
            self.experience_replay.store_episode(**episode_data)

            # Reset episode storage
            self.current_episode_observations.clear()
            self.current_episode_actions.clear()
            self.current_episode_rewards.clear()
            self.current_episode_dones.clear()

    @torch.no_grad()
    def select_action(self, observation, exploration_rate=None):
        """
        Selects an action using current policy with exploration.
        Returns:
            - Discrete action index or continuous action vector
        """
        current_exploration_rate = exploration_rate or self.exploration_rate
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.computation_device)

        # Feature extraction
        if hasattr(self.world_model, 'encoder'):
            encoded_observation = self.world_model.encoder(observation_tensor)
            policy_input = encoded_observation
        else:
            policy_input = observation_tensor

        # Action selection logic
        if self.discrete_actions:
            action_logits = self.behavior_policy(policy_input)
            if numpy.random.uniform() < current_exploration_rate:
                selected_action = numpy.random.randint(self.action_dimension)
            else:
                action_distribution = torch.distributions.Categorical(logits=action_logits)
                selected_action = action_distribution.sample().item()
            return selected_action
        else:
            action_mean, action_log_std = self.behavior_policy(policy_input)
            action_std = torch.exp(action_log_std)
            action_distribution = torch.distributions.Normal(action_mean, action_std)
            return action_distribution.sample().squeeze(0).cpu().numpy()

    def update_models(self):
        """
        Orchestrates the training process:
        1. Updates world model parameters
        2. Optimizes policy and value functions
        3. Adjusts exploration rate
        """
        if len(self.experience_replay) < 5:
            return {}

        world_model_metrics = self.update_world_model()
        policy_value_metrics = self.update_policy_and_value()

        # Gradually reduce exploration rate
        if self.discrete_actions:
            self.exploration_rate = max(
                self.minimum_exploration_rate,
                self.exploration_rate * self.exploration_decay_factor
            )

        return {**world_model_metrics, **policy_value_metrics}

    def update_world_model(self):
        """Executes world model training step using sampled sequences."""
        batch_data = self.experience_replay.sample(self.training_batch_size)
        observation_sequence = batch_data['observations'].to(self.computation_device)
        action_sequence = batch_data['actions'].to(self.computation_device)
        reward_sequence = batch_data['rewards'].to(self.computation_device)
        termination_sequence = batch_data['terminations'].to(self.computation_device)

        # Convert discrete actions to one-hot encoding
        if self.discrete_actions and action_sequence.ndim == 2:
            action_representation = neural_functional.one_hot(
                action_sequence.long(), 
                self.action_dimension
            ).float()
        else:
            action_representation = action_sequence

        # Calculate world model losses
        model_losses = self.world_model.calculate_sequence_loss(
            observation_sequence,
            action_representation,
            reward_sequence,
            termination_sequence,
            self.free_information_bits
        )

        # Optimization step
        self.world_model_optimizer.zero_grad()
        model_losses['total_loss'].backward()
        neural_network.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_model_optimizer.step()

        return {
            'world_model_total_loss': model_losses['total_loss'].item(),
            'state_divergence_loss': model_losses['state_divergence'].item(),
            'observation_reconstruction_loss': model_losses['reconstruction_loss'].item(),
            'reward_prediction_loss': model_losses['reward_prediction_loss'].item(),
            'continuation_prediction_loss': model_losses.get('continuation_prediction_loss', 0.0)
        }

    def update_policy_and_value(self):
        """Optimizes policy and value networks using imagined trajectories."""
        batch_data = self.experience_replay.sample(self.training_batch_size)
        observation_sequence = batch_data['observations'].to(self.computation_device)
        action_sequence = batch_data['actions'].to(self.computation_device)

        # Convert action representation if needed
        if self.discrete_actions and action_sequence.ndim == 2:
            action_representation = neural_functional.one_hot(
                action_sequence.long(), 
                self.action_dimension
            ).float()
        else:
            action_representation = action_sequence

        # Extract final latent state from real sequence
        with torch.no_grad():
            latent_sequence = self.world_model.rollout_latent_sequence(
                observation_sequence, 
                action_representation
            )
            final_latent_state = {
                'deterministic': latent_sequence[-1][:, :self.world_model.recurrent_state_space_model.deterministic_dimension],
                'stochastic': latent_sequence[-1][:, self.world_model.recurrent_state_space_model.deterministic_dimension:]
            }

        # Generate imagined trajectory
        imagined_rollout = self.world_model.imagine_trajectory(
            final_latent_state,
            self.behavior_policy,
            self.imagination_depth,
            self.discrete_actions
        )

        # Calculate value estimates and advantages
        for time_step in imagined_rollout:
            time_step['value_estimate'] = self.state_value_estimator(time_step['latent_feature'])
        
        imagined_returns, advantage_estimates = self.calculate_gae_returns(imagined_rollout)

        # Policy optimization
        policy_loss = torch.tensor(0.0, device=self.computation_device)
        for step_data, advantage in zip(imagined_rollout, advantage_estimates):
            latent_feature = step_data['latent_feature']
            selected_action = step_data['selected_action']
            
            if self.discrete_actions:
                action_logits = self.behavior_policy(latent_feature)
                action_distribution = torch.distributions.Categorical(logits=action_logits)
                log_probability = action_distribution.log_prob(selected_action)
            else:
                action_mean, action_log_std = self.behavior_policy(latent_feature)
                action_std = torch.exp(action_log_std)
                action_distribution = torch.distributions.Normal(action_mean, action_std)
                log_probability = action_distribution.log_prob(selected_action).sum(dim=-1)

            policy_loss -= (log_probability * advantage.detach()).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        neural_network.utils.clip_grad_norm_(self.behavior_policy.parameters(), 100.0)
        self.policy_optimizer.step()

        # Value function optimization
        value_loss = torch.tensor(0.0, device=self.computation_device)
        for step_data, target_return in zip(imagined_rollout, imagined_returns):
            predicted_value = step_data['value_estimate']
            value_loss += neural_functional.mse_loss(predicted_value, target_return.detach())

        value_loss /= len(imagined_rollout)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        neural_network.utils.clip_grad_norm_(self.state_value_estimator.parameters(), 100.0)
        self.value_optimizer.step()

        return {
            'policy_improvement_loss': policy_loss.item(),
            'value_estimation_loss': value_loss.item()
        }

    def calculate_gae_returns(self, trajectory):
        """
        Computes Generalized Advantage Estimation (GAE) returns.
        Args:
            trajectory: List of dictionaries containing:
                - reward: Immediate reward
                - continuation_probability: Episode continuation probability
                - value_estimate: Predicted state value
        Returns:
            Tuple of (returns, advantages) for each timestep
        """
        trajectory_length = len(trajectory)
        calculated_returns = [None] * trajectory_length
        calculated_advantages = [None] * trajectory_length

        next_value = trajectory[-1]['value_estimate'].detach()
        next_advantage = torch.zeros_like(next_value)

        # Reverse temporal processing
        for time_step in reversed(range(trajectory_length)):
            current_reward = trajectory[time_step]['reward']
            continuation = trajectory[time_step]['continuation_probability'] * self.discount_factor
            current_value = trajectory[time_step]['value_estimate']
            
            temporal_difference = current_reward + continuation * next_value - current_value
            advantage_estimate = temporal_difference + continuation * self.gae_lambda * next_advantage
            return_estimate = current_value + advantage_estimate

            calculated_returns[time_step] = return_estimate
            calculated_advantages[time_step] = advantage_estimate

            # Update carryover values
            next_value = current_value.detach()
            next_advantage = advantage_estimate.detach()

        return calculated_returns, calculated_advantages