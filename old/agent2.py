import torch
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from env import TetheredBoatsEnv
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        # Log or print actions and rewards
        print(f"Actions: {actions}, Rewards: {rewards}")
        return True


# Custom CNN to process the grid observations
class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN to process the grid observations
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
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
        assert 0 <= action < 81, "Invalid action"
        # Convert action back to MultiDiscrete
        action0 = action // 9
        action1 = action % 9
        multi_action = [action0, action1]
        obs, reward, done, info = self.env.step(multi_action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)


# Agent class using PPO
class TetheredBoatsAgent:
    def __init__(self, env):
        self.env = env  # Already wrapped environment
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        self.model = PPO(
            "CnnPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Increase entropy coefficient to encourage exploration
            verbose=1,
        )

    def train(self, total_timesteps):
        # In the train method
        agent.model.learn(
            total_timesteps=500000,
        )
        # agent.model.learn(total_timesteps=500000, callback=ActionLoggingCallback())

    def get_action(self, obs, training=False):
        # Add batch dimension if necessary
        if len(obs.shape) == 2:
            obs = obs[np.newaxis, ...]
        action, _ = self.model.predict(obs, deterministic=not training)
        return action

    def load_model(self, path):
        self.model = PPO.load(path, env=self.env)

    def save_model(self, path):
        self.model.save(path)


# if __name__ == "__main__":
# Create environment
env = TetheredBoatsEnv()
# Wrap the environment
wrapped_env = FlattenActionWrapper(env)
agent = TetheredBoatsAgent(wrapped_env)

# Optionally train the agent
agent.train(total_timesteps=500000)
agent.save_model("tethered_boats_model.pth")

# Load the model
agent.load_model("tethered_boats_model.pth")

# Evaluate the agent
for episode in range(env.num_episode):
    env.current_episode = episode + 1
    obs = wrapped_env.reset()
    obs = wrapped_env.reset()
    print(f"Initial observation shape: {obs.shape}")

    done = False
    while not done:
        wrapped_env.render()
        action = agent.get_action(obs)
        print(f"Action taken: {action}")
        obs, reward, done, info = wrapped_env.step(action)
        if done:
            break
wrapped_env.close()
