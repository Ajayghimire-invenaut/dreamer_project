# main.py

import yaml
import gym
import torch
import numpy as np
from modules.world_model import WorldModel
from modules.agent import DreamerAgent

def run_training(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_name = config["env_name"]
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    
    # If discrete, e.g., CartPole, action_dim = env.action_space.n
    # If continuous, e.g., MuJoCo, action_dim = env.action_space.shape[0]
    if config.get("discrete_actions", True):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    # Create WorldModel
    # You can set stoch_dim/deter_dim/hidden_dim from config if you want
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        stoch_dim=32,
        deter_dim=256,
        hidden_dim=256,
        free_bits=config.get("free_bits", 1.0)
    )

    # DreamerAgent
    agent = DreamerAgent(
        world_model=world_model,
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete_actions=config.get("discrete_actions", True),
        seq_length=config.get("seq_length", 50),
        buffer_capacity=config.get("buffer_capacity", 1000),
        batch_size=config.get("batch_size", 16),
        imagination_horizon=config.get("imagination_horizon", 15),
        gamma=config.get("gamma", 0.99),
        lam=config.get("lambda", 0.95),
        free_bits=config.get("free_bits", 1.0),
        wm_lr=config.get("wm_lr", 1e-4),
        actor_lr=config.get("actor_lr", 4e-5),
        critic_lr=config.get("critic_lr", 4e-5),
        device="cpu"  # or 'cuda' if available
    )

    obs = env.reset()
    max_steps = config.get("max_steps", 100000)
    episode_reward = 0

    for step in range(max_steps):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.store_transition(obs, action, reward, done)
        obs = next_obs
        episode_reward += reward

        if done:
            print(f"Episode finished. Total reward: {episode_reward}")
            episode_reward = 0
            obs = env.reset()

        # Periodically train
        if step > 1000 and step % 50 == 0:
            logs = agent.train()
            if logs:
                print(f"Step={step}, Logs={logs}")


if __name__ == "__main__":
    run_training()
