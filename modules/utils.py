# modules/utils.py

import numpy as numpy
import torch
from collections import deque


# --------------------------------------------------------------------------------
#  Experience Replay Buffer for Sequence-Based Learning
# --------------------------------------------------------------------------------
class ExperienceSequenceBuffer:
    """
    Stores complete episodes and samples fixed-length sequences for training recurrent models.
    
    Attributes:
        capacity (int): Maximum number of episodes to store
        sequence_length (int): Length of subsequences to sample
        episodes (list): Circular buffer of stored episodes
        current_position (int): Pointer to next storage position
    """
    
    def __init__(self, capacity=1000, sequence_length=50):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.episodes = []
        self.current_position = 0

    def store_episode(self, observations, actions, rewards, dones):
        """
        Stores a complete episode in the buffer.
        
        Args:
            observations: Sequence of environment observations
            actions: Sequence of agent actions
            rewards: Sequence of environment rewards
            dones: Sequence of episode termination flags
        """
        episode_record = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }

        if len(self.episodes) < self.capacity:
            self.episodes.append(episode_record)
        else:
            self.episodes[self.current_position] = episode_record
        self.current_position = (self.current_position + 1) % self.capacity

    def sample_sequences(self, batch_size):
        """
        Samples batch of fixed-length sequences from stored episodes.
        
        Returns:
            observations: Tensor [batch_size, seq_len, obs_dim]
            actions: Tensor [batch_size, seq_len, ...]
            rewards: Tensor [batch_size, seq_len]
            dones: Tensor [batch_size, seq_len]
        """
        valid_episodes = [ep for ep in self.episodes if ep is not None]
        selected_episodes = numpy.random.choice(valid_episodes, size=batch_size)

        observation_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []

        for episode in selected_episodes:
            episode_length = len(episode['observations'])
            
            # Handle episodes shorter than required sequence length
            if episode_length <= self.sequence_length:
                start_index = 0
            else:
                start_index = numpy.random.randint(0, episode_length - self.sequence_length)

            # Extract sequence chunk
            end_index = start_index + self.sequence_length
            observation_sequences.append(episode['observations'][start_index:end_index])
            action_sequences.append(episode['actions'][start_index:end_index])
            reward_sequences.append(episode['rewards'][start_index:end_index])
            done_sequences.append(episode['dones'][start_index:end_index])

        # Convert to tensors with explicit type casting
        return (
            torch.as_tensor(numpy.array(observation_sequences), dtype=torch.float32),
            torch.as_tensor(numpy.array(action_sequences), dtype=torch.float32),
            torch.as_tensor(numpy.array(reward_sequences), dtype=torch.float32),
            torch.as_tensor(numpy.array(done_sequences), dtype=torch.float32)
        )

    def __len__(self):
        """Returns current number of stored episodes."""
        return len([ep for ep in self.episodes if ep is not None])


# --------------------------------------------------------------------------------
#  Adaptive Normalization Statistics
# --------------------------------------------------------------------------------
class RunningStatisticsTracker:
    """
    Maintains and updates running statistics for data normalization.
    
    Attributes:
        mean (float): Running mean
        variance (float): Running variance
        sample_count (int): Total number of processed samples
        epsilon (float): Numerical stability constant
    """
    
    def __init__(self, epsilon=1e-5):
        self.mean = 0.0
        self.variance = 1.0
        self.sample_count = 0
        self.epsilon = epsilon

    def update(self, data_batch: torch.Tensor):
        """Updates statistics with new batch of data using Welford's algorithm."""
        batch_mean = data_batch.mean().item()
        batch_variance = data_batch.var().item()
        batch_size = data_batch.numel()

        total_samples = self.sample_count + batch_size
        mean_delta = batch_mean - self.mean
        
        # Update mean using combined average
        combined_mean = self.mean + mean_delta * batch_size / total_samples
        
        # Update variance using combined sum of squares
        combined_variance = (
            (self.variance * self.sample_count) + 
            (batch_variance * batch_size) + 
            (mean_delta**2 * self.sample_count * batch_size) / total_samples
        ) / total_samples

        self.mean = combined_mean
        self.variance = combined_variance
        self.sample_count = total_samples

    def normalize(self, data: torch.Tensor):
        """Applies z-score normalization using current statistics."""
        return (data - self.mean) / torch.sqrt(torch.tensor(self.variance) + self.epsilon)

    def denormalize(self, normalized_data: torch.Tensor):
        """Reverses normalization transformation."""
        return normalized_data * torch.sqrt(torch.tensor(self.variance) + self.epsilon) + self.mean


# --------------------------------------------------------------------------------
#  Value Transformation Utilities
# --------------------------------------------------------------------------------
def symmetric_logarithm_transform(values: torch.Tensor) -> torch.Tensor:
    """Applies signed logarithmic transformation to preserve sign and compresses magnitude."""
    return torch.sign(values) * torch.log1p(torch.abs(values))

def symmetric_exponential_transform(values: torch.Tensor) -> torch.Tensor:
    """Inverts symmetric logarithm transformation."""
    return torch.sign(values) * (torch.exp(torch.abs(values)) - 1.0)

def create_two_hot_distribution(values: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """
    Creates soft two-hot distribution over predefined bins.
    
    Args:
        values: Tensor of values to encode [batch_size]
        bin_centers: Tensor of bin center positions [num_bins]
        
    Returns:
        Tensor of probabilities over bins [batch_size, num_bins]
    """
    value_bin_distances = torch.abs(values.unsqueeze(-1) - bin_centers)
    sharpness_factor = 10.0  # Controls distribution concentration
    return torch.softmax(-value_bin_distances * sharpness_factor, dim=-1)