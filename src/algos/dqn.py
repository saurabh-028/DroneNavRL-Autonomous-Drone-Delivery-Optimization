
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def train_dqn(env_fn, timesteps=200_000, save_path="checkpoints/dqn_best"):
    env = DummyVecEnv([env_fn])
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        target_update_interval=1000,
        gamma=0.99,
        exploration_fraction=0.2,
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return save_path

def load_dqn(path, env_fn):
    env = DummyVecEnv([env_fn])
    model = DQN.load(path, env=env)
    return model, env
