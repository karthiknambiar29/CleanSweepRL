import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import TetheredBoatsEnv
from agent import TetheredBoatsAgent


# Define the environment
class CustomTetheredBoatsEnv(TetheredBoatsEnv):
    def __init__(self, **kwargs):
        super(CustomTetheredBoatsEnv, self).__init__(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs.flatten(), reward, done, info  # Flatten grid for PPO

    def reset(self):
        return super().reset().flatten()  # Flatten grid for PPO


# if __name__ == "__main2__":
# Create and wrap the environment
env = CustomTetheredBoatsEnv()
vec_env = make_vec_env(lambda: env, n_envs=4)  # Parallel environments for efficiency

# Define the PPO model
model = PPO(
    "MlpPolicy",  # Multi-layer Perceptron policy
    vec_env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    seed=42,
)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_tethered_boats")

# Evaluate the agent
env = CustomTetheredBoatsEnv()
obs = env.reset()
for _ in range(env.step_per_episode):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
