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
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image
import gymnasium as gym
from gymnasium import spaces

# Add src/rl to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MinecraftWoodEnv, MinecraftWoodSimpleEnv


ACTION_DIMS_FULL = [3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5]
ACTION_DIMS_SIMPLE = [3, 2, 3, 3, 5, 5, 2]


class MultiDiscreteToDiscreteActionWrapper(gym.ActionWrapper):
    """Map MultiDiscrete action space to a single Discrete index (for DQN)."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError("MultiDiscreteToDiscreteActionWrapper requires MultiDiscrete action space")

        self.nvec = np.array(env.action_space.nvec, dtype=np.int64)
        self.action_space = spaces.Discrete(int(np.prod(self.nvec)))

    def action(self, act):
        act = int(act)
        return np.array(np.unravel_index(act, tuple(self.nvec)), dtype=np.int64)


def _prepare_env_for_algo(env, algo):
    """Return environment adapted to the requested algorithm."""
    algo = algo.lower()

    if algo != "dqn":
        return env

    if isinstance(env.action_space, spaces.Discrete):
        return env

    if isinstance(env.action_space, spaces.MultiDiscrete):
        wrapped = MultiDiscreteToDiscreteActionWrapper(env)
        max_actions_for_dqn = 20_000
        if wrapped.action_space.n > max_actions_for_dqn:
            raise ValueError(
                f"DQN with flattened MultiDiscrete has {wrapped.action_space.n} actions, "
                f"too large for practical training. Use --simple or choose PPO/A2C."
            )
        return wrapped

    raise ValueError(f"DQN only supports Discrete actions, got: {type(env.action_space)}")


def _get_eval_env_for_model(model, fallback_env):
    return getattr(model, "_rl_eval_env", fallback_env)


def _state_to_obs13(state_dict):
    """Map dataset state into env observation layout (13 dims).
    
    Supports two formats:
      - New: state_dict has 'vector' key with full 13-dim list from extractState()
      - Legacy: state_dict has x,y,z,yaw,pitch only (fills rest with defaults)
    """
    # New format: full state vector already normalized
    if "vector" in state_dict:
        vec = state_dict["vector"]
        if isinstance(vec, list) and len(vec) == 13:
            return np.array(vec, dtype=np.float32)

    # Legacy format: only position + camera
    obs = np.zeros(13, dtype=np.float32)
    obs[0] = float(state_dict.get("x", 0.0)) / 100.0
    obs[1] = float(state_dict.get("y", 0.0)) / 100.0
    obs[2] = float(state_dict.get("z", 0.0)) / 100.0
    obs[3] = float(state_dict.get("yaw", 0.0)) / np.pi
    obs[4] = float(state_dict.get("pitch", 0.0)) / (np.pi / 2)

    # Unknown features in offline dataset -> neutral defaults
    obs[5] = 0.0      # log_count
    obs[6] = 0.0      # looking_at_log
    obs[7] = 0.0      # hardness
    obs[8] = 1.0      # on_ground
    obs[9] = 1.0      # health
    obs[10] = 1.0     # food
    obs[11] = 0.0     # cursor distance
    obs[12] = 0.0     # time of day
    return obs


def _resolve_image_path(raw_path, jsonl_path):
    """Resolve image path from dataset row into an absolute path."""
    if not raw_path:
        return None
    path = os.path.normpath(raw_path)
    if os.path.isabs(path):
        return path

    # 1) Relative to current working directory (project root in normal usage)
    cwd_path = os.path.normpath(os.path.join(os.getcwd(), path))
    if os.path.exists(cwd_path):
        return cwd_path

    # 2) Relative to JSONL location
    jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
    return os.path.normpath(os.path.join(jsonl_dir, path))


def _load_image_tensor(image_path, image_size):
    """Load image as float tensor in CHW format, normalized to [0,1]."""
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def load_imitation_dataset(jsonl_path, simple=False, use_images=False, image_size=84):
    """Load data/train.jsonl for behavioral cloning (state-only or state+image)."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    obs_list = []
    img_list = []
    act_list = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            state = row.get("state", {})
            action = row.get("action", None)
            if not isinstance(action, list) or len(action) < 11:
                continue

            obs = _state_to_obs13(state)
            action = np.array(action[:7] if simple else action[:11], dtype=np.int64)

            if use_images:
                image_path = _resolve_image_path(row.get("image"), jsonl_path)
                if not image_path or not os.path.exists(image_path):
                    continue
                try:
                    image = _load_image_tensor(image_path, image_size)
                except Exception:
                    continue
                img_list.append(image)

            obs_list.append(obs)
            act_list.append(action)

    if not obs_list:
        raise RuntimeError("No valid rows found in imitation dataset")

    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(act_list), dtype=torch.long)
    if use_images:
        I = torch.tensor(np.array(img_list), dtype=torch.float32)
        return X, I, Y
    return X, Y


class MultiHeadBC(nn.Module):
    """Behavioral cloning policy for MultiDiscrete actions."""

    def __init__(self, obs_dim, action_dims):
        super().__init__()
        self.action_dims = list(action_dims)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, d) for d in self.action_dims])

    def forward(self, x):
        h = self.backbone(x)
        return [head(h) for head in self.heads]


class VisualStateMultiHeadBC(nn.Module):
    """Behavioral cloning policy from image + low-dimensional state."""

    def __init__(self, obs_dim, action_dims):
        super().__init__()
        self.action_dims = list(action_dims)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.state_backbone = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, d) for d in self.action_dims])

    def forward(self, obs, image):
        h_img = self.cnn(image).flatten(1)
        h_obs = self.state_backbone(obs)
        h = self.fusion(torch.cat([h_obs, h_img], dim=1))
        return [head(h) for head in self.heads]


class BCPredictor:
    """Small adapter to reuse evaluate_model() API."""

    def __init__(self, model, action_dims, device, use_images=False, image_size=84):
        self.model = model
        self.action_dims = action_dims
        self.device = device
        self.use_images = use_images
        self.image_size = image_size

    def predict(self, obs, deterministic=True):
        _ = deterministic
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.use_images:
                # During online env eval we currently don't have RGB frames in env obs,
                # so we pass a neutral image. Training still uses real screenshots.
                neutral = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32, device=self.device)
                logits = self.model(x, neutral)
            else:
                logits = self.model(x)
            action = [int(torch.argmax(h, dim=1).item()) for h in logits]
        return np.array(action, dtype=np.int64), None


def train_bc(jsonl_path, save_path="models/bc_minecraft.pt", simple=False,
             epochs=20, batch_size=256, lr=1e-3,
             use_images=False, image_size=84):
    """Train behavioral cloning policy from recorded dataset."""
    print("=" * 50)
    print("BEHAVIORAL CLONING" + (" (CNN + STATE)" if use_images else " (STATE ONLY)"))
    print("=" * 50)

    action_dims = ACTION_DIMS_SIMPLE if simple else ACTION_DIMS_FULL
    if use_images:
        X, I, Y = load_imitation_dataset(jsonl_path, simple=simple, use_images=True, image_size=image_size)
        dataset = TensorDataset(X, I, Y)
    else:
        X, Y = load_imitation_dataset(jsonl_path, simple=simple, use_images=False, image_size=image_size)
        dataset = TensorDataset(X, Y)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(dataset, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_images:
        model = VisualStateMultiHeadBC(obs_dim=X.shape[1], action_dims=action_dims).to(device)
    else:
        model = MultiHeadBC(obs_dim=X.shape[1], action_dims=action_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Samples: {n_total} (train={n_train}, val={n_val})")
    print(f"Obs dim: {X.shape[1]}, Action dims: {action_dims}")
    if use_images:
        print(f"Image tensor: {tuple(I.shape[1:])}")
    print(f"Device: {device}")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            if use_images:
                xb, ib, yb = batch
                xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
                logits = model(xb, ib)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
            loss = sum(nn.functional.cross_entropy(logits[i], yb[:, i]) for i in range(len(action_dims)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct = np.zeros(len(action_dims), dtype=np.float64)
        total = 0
        with torch.no_grad():
            for batch in dl_val:
                if use_images:
                    xb, ib, yb = batch
                    xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
                    logits = model(xb, ib)
                else:
                    xb, yb = batch
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                loss = sum(nn.functional.cross_entropy(logits[i], yb[:, i]) for i in range(len(action_dims)))
                val_loss += loss.item() * xb.size(0)

                preds = [torch.argmax(logits[i], dim=1) for i in range(len(action_dims))]
                for i in range(len(action_dims)):
                    correct[i] += (preds[i] == yb[:, i]).sum().item()
                total += xb.size(0)

        val_loss /= n_val
        per_dim_acc = correct / max(total, 1)
        mean_acc = float(np.mean(per_dim_acc))
        print(f"Epoch {ep:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | mean_dim_acc={mean_acc:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "action_dims": action_dims,
        "obs_dim": int(X.shape[1]),
        "use_images": bool(use_images),
        "image_size": int(image_size),
    }, save_path)
    print(f"\nBC model saved to {save_path}")

    return BCPredictor(model, action_dims, device, use_images=use_images, image_size=image_size)


def load_bc_model(path):
    """Load previously trained BC model."""
    ckpt = torch.load(path, map_location="cpu")
    action_dims = ckpt["action_dims"]
    obs_dim = ckpt.get("obs_dim", 13)
    use_images = bool(ckpt.get("use_images", False))
    image_size = int(ckpt.get("image_size", 84))
    model = VisualStateMultiHeadBC(obs_dim=obs_dim, action_dims=action_dims) if use_images else MultiHeadBC(obs_dim=obs_dim, action_dims=action_dims)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return BCPredictor(model, action_dims, device, use_images=use_images, image_size=image_size)


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

def train_sb3(env, algo="ppo", total_timesteps=50_000, save_path="models/ppo_minecraft", device="cpu"):
    """Train an SB3 algorithm with a compact unified entrypoint."""
    try:
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return None

    algo = algo.lower()
    print("=" * 50)
    print(f"{algo.upper()} TRAINING — {total_timesteps} timesteps")
    print("=" * 50)

    model_env = _prepare_env_for_algo(env, algo)

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            model_env,
            device=device,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            tensorboard_log="./logs/ppo_minecraft/",
        )
    elif algo == "dqn":
        model = DQN(
            "MlpPolicy",
            model_env,
            device=device,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            tensorboard_log="./logs/dqn_minecraft/",
        )
    elif algo == "a2c":
        model = A2C(
            "MlpPolicy",
            model_env,
            device=device,
            verbose=1,
            learning_rate=7e-4,
            n_steps=64,
            gamma=0.99,
            tensorboard_log="./logs/a2c_minecraft/",
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model.learn(total_timesteps=total_timesteps)
    model._rl_eval_env = model_env

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    return model

def benchmark_algorithms(env, algorithms, timesteps, episodes, device="cpu", model_dir="models"):
    """Train + evaluate several algorithms in one run."""
    print("=" * 50)
    print("RL ALGORITHM BENCHMARK")
    print("=" * 50)

    summary = {}

    # Baseline first
    random_results = run_random_baseline(env, episodes=episodes)
    summary["random"] = float(np.mean([r["reward"] for r in random_results]))

    for algo in algorithms:
        algo = algo.lower().strip()
        if not algo:
            continue
        save_path = os.path.join(model_dir, f"{algo}_minecraft")
        model = train_sb3(env, algo=algo, total_timesteps=timesteps, save_path=save_path, device=device)
        if model is None:
            summary[algo] = None
            continue
        results = evaluate_model(_get_eval_env_for_model(model, env), model, episodes=episodes)
        summary[algo] = float(np.mean([r["reward"] for r in results]))

    print("\n" + "-" * 50)
    print("Benchmark summary (avg reward):")
    for name, value in summary.items():
        label = "N/A" if value is None else f"{value:.2f}"
        print(f"  {name:>8s}: {label}")
    print("-" * 50)

    return summary


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
    parser.add_argument("--mode", choices=["random", "ppo", "dqn", "a2c", "eval", "bc", "bc_eval", "bench"], default="random",
                        help="Training mode")
    parser.add_argument("--server", default="http://localhost:3001",
                        help="JS bot server URL")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes (for random/eval)")
    parser.add_argument("--timesteps", type=int, default=500,
                        help="Total timesteps for RL training")
    parser.add_argument("--simple", action="store_true",
                        help="Use simplified wood-only action space (7 dims)")
    parser.add_argument("--model_path", default="models/dqn_minecraft",
                        help="Path to save/load model")
    parser.add_argument("--algo", default="dqn", choices=["ppo", "dqn", "a2c"],
                        help="Algorithm to use in eval mode")
    parser.add_argument("--algorithms", default="ppo,dqn,a2c",
                        help="Comma-separated algorithms for --mode bench")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Device for PPO training/eval (default: cpu)")
    parser.add_argument("--jsonl", default="data/train_balanced.jsonl",
                        help="Imitation dataset path (JSONL)")
    parser.add_argument("--bc_epochs", type=int, default=20,
                        help="Behavioral cloning epochs")
    parser.add_argument("--bc_batch_size", type=int, default=256,
                        help="Behavioral cloning batch size")
    parser.add_argument("--bc_lr", type=float, default=1e-3,
                        help="Behavioral cloning learning rate")
    parser.add_argument("--use_images", action="store_true", default=True,
                        help="Use dataset screenshots with a CNN during BC")
    parser.add_argument("--image_size", type=int, default=84,
                        help="Image size for CNN BC (square)")
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

        elif args.mode in {"ppo", "dqn", "a2c"}:
            model = train_sb3(env, algo=args.mode, total_timesteps=args.timesteps, save_path=args.model_path, device=args.device)
            if model:
                evaluate_model(_get_eval_env_for_model(model, env), model, args.episodes)

        elif args.mode == "eval":
            from stable_baselines3 import PPO, DQN, A2C
            if args.algo == "ppo":
                model = PPO.load(args.model_path, device=args.device)
            elif args.algo == "dqn":
                model = DQN.load(args.model_path, device=args.device)
                model._rl_eval_env = _prepare_env_for_algo(env, "dqn")
            else:
                model = A2C.load(args.model_path, device=args.device)
            evaluate_model(_get_eval_env_for_model(model, env), model, args.episodes)

        elif args.mode == "bc":
            bc_model = train_bc(
                jsonl_path=args.jsonl,
                save_path=args.model_path,
                simple=args.simple,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
                use_images=args.use_images,
                image_size=args.image_size,
            )
            evaluate_model(env, bc_model, args.episodes)

        elif args.mode == "bc_eval":
            bc_model = load_bc_model(args.model_path)
            evaluate_model(env, bc_model, args.episodes)

        elif args.mode == "bench":
            algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
            benchmark_algorithms(
                env,
                algorithms=algos,
                timesteps=args.timesteps,
                episodes=args.episodes,
                device=args.device,
                model_dir=os.path.dirname(args.model_path) or "models",
            )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
