import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from minigrid import wrappers

# Create the environment using gymnasium
def make_env():
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = wrappers.RGBImgPartialObsWrapper(env)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env for _ in range(4)])

# Set up tensorboard logging
log_dir = "./ppo_minigrid_logs/"
new_logger = configure(log_dir, ["tensorboard"])

# Set PPO parameters
ppo_params = {
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'learning_rate': 3e-4,
    'clip_range': 0.2,
    'vf_coef': 0.5  # Starting value for tuning
}

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **ppo_params)
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000)

# Save the final model
model.save("ppo_minigrid")
env.close()
