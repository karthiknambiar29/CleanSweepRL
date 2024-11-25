from env import TetheredBoatsEnv
from agent import TetheredBoatsAgent

if __name__ == "__main__":
    # Create environment
    env = TetheredBoatsEnv()
    agent = TetheredBoatsAgent(env)

    # Train the agent
    agent.train(total_timesteps=100000)

    # Save the model
    agent.save_model("tethered_boats_model.pth")

    # Load the model
    agent.load_model("tethered_boats_model.pth")

    # Evaluate the agent
    for episode in range(env.num_episode):
        env.current_episode = episode + 1
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.get_action(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            if done:
                break
    env.close()
