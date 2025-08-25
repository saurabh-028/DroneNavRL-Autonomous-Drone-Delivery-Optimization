
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def train_ppo(env_fn, timesteps=200_000, save_path="checkpoints/ppo_best"):
    env = DummyVecEnv([env_fn])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return save_path

def load_ppo(path, env_fn):
    env = DummyVecEnv([env_fn])
    model = PPO.load(path, env=env)
    return model, env
