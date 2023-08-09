import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create the environment
env = gym.make('ALE/Blackjack-v5', render_mode='human')

# Vectorize the environment to enable multiple environments
env = DummyVecEnv([lambda: env])

# Instantiate the agent
model = PPO('CnnPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("ppo_blackjack")

# Load the trained agent
model = PPO.load("ppo_blackjack")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")