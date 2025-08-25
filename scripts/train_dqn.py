
import argparse
from src.envs.grid_world import DroneDeliveryEnv
from src.algos.dqn import train_dqn

def make_env(args):
    return DroneDeliveryEnv(grid_size=args.grid_size, num_birds=args.num_birds, max_steps=args.max_steps)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_birds", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--save_path", type=str, default="checkpoints/dqn_best")
    args = p.parse_args()

    env_fn = lambda: make_env(args)
    path = train_dqn(env_fn, timesteps=args.timesteps, save_path=args.save_path)
    print(f"Saved DQN model to {path}.zip")

if __name__ == "__main__":
    main()
