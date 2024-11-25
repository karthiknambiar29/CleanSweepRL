import torch
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from env import TetheredBoatsEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb


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
        # Ensure action is an integer scalar
        if isinstance(action, np.ndarray):
            action = action.item()
        elif isinstance(action, list):
            action = action[0]
        # Convert action back to MultiDiscrete
        action0 = int(action // 9)
        action1 = int(action % 9)
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
        wandb.init(project="tethered-boats")
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
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,  # Increase entropy coefficient to encourage exploration
            verbose=1,
        )

    def train(self, total_timesteps):
        # Initialize logging variables
        total_reward_episode = 0
        total_loss_episode = 0
        total_trash_collected = 0
        previous_trash_count = self.env.n_trash

        # ...existing training loop...
        for timestep in range(total_timesteps):
            # ...existing training steps...

            # Example: Interact with the environment
            obs, reward, done, info = self.env.step(action)

            # Calculate trash collected in this step
            current_trash_count = len(self.env.trash_positions)
            trash_collected_step = previous_trash_count - current_trash_count
            previous_trash_count = current_trash_count
            total_trash_collected += trash_collected_step

            # Accumulate rewards
            reward_step = reward
            total_reward_episode += reward_step

            # Placeholder for loss (requires access to model's loss)
            loss_step = 0  # Replace with actual loss retrieval if possible
            total_loss_episode += loss_step

            # After each step
            wandb.log(
                {
                    "trash_collected_step": trash_collected_step,
                    "reward_step": reward_step,
                }
            )

            if done:
                # After each episode
                wandb.log(
                    {
                        "trash_collected_episode": total_trash_collected,
                        "reward_episode": total_reward_episode,
                        "loss_episode": total_loss_episode,
                    }
                )
                # Reset episode variables
                total_reward_episode = 0
                total_trash_collected = 0
                total_loss_episode = 0
                previous_trash_count = self.env.n_trash

        # ...existing training steps...

    def get_action(self, obs, training=False):
        # Add batch dimension if necessary
        if len(obs.shape) == 2:
            obs = obs[np.newaxis, ...]
        action, _ = self.model.predict(obs, deterministic=not training)
        if isinstance(action, np.ndarray):
            action = action.item()
        return action

    def load_model(self, path):
        self.model = PPO.load(path, env=self.env)

    def save_model(self, path):
        self.model.save(path)


# if __name__ == "__main__":
# Create environment
env = TetheredBoatsEnv(num_episode=100, n_trash=1)
# Wrap the environment
wrapped_env = FlattenActionWrapper(env)
agent = TetheredBoatsAgent(wrapped_env)

# Optionally train the agent
agent.train(total_timesteps=500000)
agent.save_model("tethered_boats_model.pth")

# Load the model
agent.load_model("tethered_boats_model.pth")
trash_left = []
# Evaluate the agent
for episode in range(env.num_episode):
    env.current_episode = episode + 1
    obs = wrapped_env.reset()
    obs = wrapped_env.reset()
    # print(f"Initial observation shape: {obs.shape}")
    steps = 0
    done = False
    while not done:
        # wrapped_env.render()
        action = agent.get_action(obs, training=True)
        # take random action
        # action = env.action_space.sample()
        # # print(action)
        # # convert to multi discrete action
        # # covnert to wrapped action
        # action = action[0] * 9 + action[1]

        # print(f"Action taken: {action}")
        obs, reward, done, info = wrapped_env.step(action)
        if done:
            break
        steps += 1
        if steps > 200:
            done = True
    # print(f"{len(env.trash_positions)} trash left")
    trash_left.append(len(env.trash_positions))

print(f"Average trash left: {np.mean(trash_left)}")
wrapped_env.close()
