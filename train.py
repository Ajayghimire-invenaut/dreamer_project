# main.py

import yaml
import gymnasium as gym
import torch
import numpy as numpy
from typing import Dict, Any, Optional
from modules.world_model import WorldModel
from modules.agent import DreamerLearningAgent


def execute_training_workflow(configuration_path: str = "configs/default.yaml") -> None:
    """
    Orchestrates the complete training process for the Dreamer agent.
    
    Args:
        configuration_path: Path to YAML configuration file
    """
    # Load training configuration
    configuration = _load_training_configuration(configuration_path)
    
    # Initialize training environment
    environment, observation_dimension, action_dimension = _setup_training_environment(
        configuration["environment_name"],
        configuration.get("discrete_actions", True)
    )
    
    # Initialize world model components
    world_model = _create_world_model(
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        configuration=configuration
    )
    
    # Create learning agent
    learning_agent = _initialize_learning_agent(
        world_model=world_model,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        configuration=configuration
    )
    
    # Execute main training sequence
    _run_training_loop(
        environment=environment,
        learning_agent=learning_agent,
        max_training_steps=configuration.get("max_training_steps", 100000),
        training_start_delay=configuration.get("training_start_delay", 1000),
        training_interval=configuration.get("training_interval", 50)
    )


def _load_training_configuration(config_path: str) -> Dict[str, Any]:
    """Load and return training parameters from YAML file."""
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing configuration file: {e}")


def _setup_training_environment(
    environment_name: str,
    discrete_actions: bool
) -> tuple[gym.Env, int, int]:
    """Initialize and configure the training environment."""
    try:
        environment = gym.make(environment_name)
    except gym.error.Error as e:
        raise RuntimeError(f"Failed to create environment {environment_name}: {e}")
    
    # Validate action space configuration
    if discrete_actions and not isinstance(environment.action_space, gym.spaces.Discrete):
        raise ValueError("Discrete action configuration mismatch with environment's action space")
    if not discrete_actions and not isinstance(environment.action_space, gym.spaces.Box):
        raise ValueError("Continuous action configuration mismatch with environment's action space")
    
    observation_dimension = environment.observation_space.shape[0]
    action_dimension = (
        environment.action_space.n if discrete_actions 
        else environment.action_space.shape[0]
    )
    
    return environment, observation_dimension, action_dimension


def _create_world_model(
    observation_dimension: int,
    action_dimension: int,
    configuration: Dict[str, Any]
) -> WorldModel:
    """Instantiate and return the world model component."""
    return WorldModel(
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        stochastic_dimension=configuration.get("stochastic_dimension", 32),
        deterministic_dimension=configuration.get("deterministic_dimension", 256),
        hidden_dimension=configuration.get("hidden_dimension", 256),
        free_information_bits=configuration.get("free_information_bits", 1.0)
    )


def _initialize_learning_agent(
    world_model: WorldModel,
    observation_dimension: int,
    action_dimension: int,
    configuration: Dict[str, Any]
) -> DreamerLearningAgent:
    """Configure and return the Dreamer learning agent."""
    computation_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return DreamerLearningAgent(
        world_model=world_model,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        discrete_actions=configuration.get("discrete_actions", True),
        sequence_length=configuration.get("sequence_length", 50),
        replay_capacity=configuration.get("replay_capacity", 1000),
        training_batch_size=configuration.get("training_batch_size", 16),
        imagination_depth=configuration.get("imagination_depth", 15),
        discount_factor=configuration.get("discount_factor", 0.99),
        gae_lambda=configuration.get("gae_lambda", 0.95),
        free_information_bits=configuration.get("free_information_bits", 1.0),
        world_model_learning_rate=configuration.get("world_model_learning_rate", 1e-4),
        policy_learning_rate=configuration.get("policy_learning_rate", 4e-5),
        value_learning_rate=configuration.get("value_learning_rate", 4e-5),
        computation_device=computation_device
    )


def _run_training_loop(
    environment: gym.Env,
    learning_agent: DreamerLearningAgent,
    max_training_steps: int,
    training_start_delay: int,
    training_interval: int
) -> None:
    """Execute the main training sequence with periodic updates."""
    current_observation, _ = environment.reset()
    episode_cumulative_reward = 0.0

    for training_step in range(1, max_training_steps + 1):
        # Agent interaction with environment
        selected_action = learning_agent.select_action(current_observation)
        next_observation, reward, terminated, truncated, _ = environment.step(selected_action)
        
        # Store experience and update state
        learning_agent.record_experience(current_observation, selected_action, reward, terminated or truncated)
        current_observation = next_observation
        episode_cumulative_reward += reward

        # Handle episode completion
        if terminated or truncated:
            print(f"Episode completed at step {training_step} | Total Reward: {episode_cumulative_reward:.2f}")
            episode_cumulative_reward = 0.0
            current_observation, _ = environment.reset()

        # Perform training updates
        if training_step > training_start_delay and training_step % training_interval == 0:
            training_metrics = learning_agent.update_models()
            if training_metrics:
                print(f"Training Step: {training_step}")
                for metric, value in training_metrics.items():
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")


if __name__ == "__main__":
    try:
        execute_training_workflow()
    except KeyboardInterrupt:
        print("\nTraining process interrupted by user.")
    except Exception as error:
        print(f"Unexpected error occurred: {error}")
        raise