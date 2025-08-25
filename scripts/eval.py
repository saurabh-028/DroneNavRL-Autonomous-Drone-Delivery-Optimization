
import argparse, imageio
from stable_baselines3 import DQN, PPO
from src.envs.grid_world import DroneDeliveryEnv

def rollout(model, env, episodes=3):
    frames = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))
            frames.append(env.render())
    return frames

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_birds", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--gif_path", type=str, default="media/rollout.gif")
    args = p.parse_args()

    env = DroneDeliveryEnv(grid_size=args.grid_size, num_birds=args.num_birds, max_steps=args.max_steps)
    model = PPO.load(args.model_path) if args.algo == "ppo" else DQN.load(args.model_path)

    frames = rollout(model, env, episodes=args.episodes)
    imageio.mimsave(args.gif_path, frames, fps=8)
    print(f"Saved GIF to {args.gif_path}")

if __name__ == "__main__":
    main()
