import torch
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


# Custom CNN to process the grid observations
class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the grid observations
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume the observation is of shape (grid_size, grid_size)
        n_input_channels = 1  # Because observation is (grid_size, grid_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(
                1,
                n_input_channels,
                observation_space.shape[0],
                observation_space.shape[1],
            )
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        # observations shape: (batch_size, grid_size, grid_size)
        observations = observations.float()
        observations = observations.unsqueeze(1)  # Add channel dimension
        features = self.cnn(observations)
        return self.linear(features)


# Wrapper to flatten the MultiDiscrete action space
class FlattenActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)
        # Original action space is MultiDiscrete([9,9])
        self.orig_action_space = env.action_space
        # Flattened action space is Discrete(81)
        self.action_space = gym.spaces.Discrete(81)
        self.observation_space = env.observation_space

    def step(self, action):
        # Convert action back to MultiDiscrete
        action0 = action // 9
        action1 = action % 9
        multi_action = [action0, action1]
        obs, reward, done, info = self.env.step(multi_action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


# Agent class using PPO
class TetheredBoatsAgent:
    def __init__(self, env):
        self.env = FlattenActionWrapper(env)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.model = PPO("CnnPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def get_action(self, obs, training=False):
        action, _ = self.model.predict(obs, deterministic=not training)
        return action

    def load_model(self, path):
        self.model = PPO.load(path)

    def save_model(self, path):
        self.model.save(path)
