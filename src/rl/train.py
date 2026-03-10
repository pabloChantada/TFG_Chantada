"""
Simple RL training script for the Minecraft wood-chopping task.

Prerequisites:
    1. Start Minecraft server with LAN open
    2. Start the JS bot server:  node src/rl/server.js --mc_port <port>
    3. Run this script:          python src/rl/train.py

Supports:
    - Random baseline
    - PPO (via stable-baselines3)
    - Imitation learning from recorded demos (behavioral cloning)
"""

import sys
import os
import argparse
import json
import time
import numpy as np

# Add src/rl to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MinecraftWoodEnv, MinecraftWoodSimpleEnv


def run_random_baseline(env, episodes=10):
    """Run random agent as a baseline."""
    print("=" * 50)
    print("RANDOM BASELINE")
    print("=" * 50)

    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        results.append({
            "episode": ep + 1,
            "steps": steps,
            "reward": round(total_reward, 2),
            "logs": info.get("logs_collected", 0),
            "success": terminated and not truncated,
        })
        print(f"  Ep {ep+1:3d} | Steps: {steps:4d} | Reward: {total_reward:7.2f} | "
              f"Logs: {info.get('logs_collected', '?')}")

    avg_r = np.mean([r["reward"] for r in results])
    successes = sum(r["success"] for r in results)
    print(f"\nAvg reward: {avg_r:.2f} | Successes: {successes}/{episodes}")
    return results


def train_ppo(env, total_timesteps=50_000, save_path="models/ppo_minecraft"):
    """Train PPO agent using stable-baselines3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return None

    print("=" * 50)
    print(f"PPO TRAINING — {total_timesteps} timesteps")
    print("=" * 50)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./logs/ppo_minecraft/",
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    return model


def evaluate_model(env, model, episodes=10):
    """Evaluate a trained model."""
    print("=" * 50)
    print("EVALUATION")
    print("=" * 50)

    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        results.append({"episode": ep + 1, "steps": steps, "reward": round(total_reward, 2)})
        print(f"  Ep {ep+1:3d} | Steps: {steps:4d} | Reward: {total_reward:7.2f}")

    avg_r = np.mean([r["reward"] for r in results])
    print(f"\nAvg reward: {avg_r:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Minecraft RL Training")
    parser.add_argument("--mode", choices=["random", "ppo", "eval"], default="random",
                        help="Training mode")
    parser.add_argument("--server", default="http://localhost:3001",
                        help="JS bot server URL")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes (for random/eval)")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total timesteps for PPO training")
    parser.add_argument("--simple", action="store_true",
                        help="Use simplified wood-only action space (7 dims)")
    parser.add_argument("--model_path", default="models/ppo_minecraft",
                        help="Path to save/load model")
    args = parser.parse_args()

    # Create environment
    EnvClass = MinecraftWoodSimpleEnv if args.simple else MinecraftWoodEnv
    env = EnvClass(server_url=args.server)

    try:
        # Check server connectivity
        info = env.get_server_info()
        print(f"Connected to server: {args.server}")
        print(f"Action space: MultiDiscrete({list(info['action_space'])})")
        print(f"Observation dim: {info['observation_dim']}")
        print()

        if args.mode == "random":
            run_random_baseline(env, args.episodes)

        elif args.mode == "ppo":
            model = train_ppo(env, args.timesteps, args.model_path)
            if model:
                evaluate_model(env, model, args.episodes)

        elif args.mode == "eval":
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)
            evaluate_model(env, model, args.episodes)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
