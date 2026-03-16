"""
Simple imitation-learning script for the Minecraft wood-chopping task.

First iteration scope:
    - Behavioral cloning only (no RL training algorithms)
    - CNN + state policy for action prediction
"""

import sys
import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Add src/rl to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import MinecraftWoodEnv, MinecraftWoodSimpleEnv


ACTION_DIMS_FULL = [3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5]
ACTION_DIMS_SIMPLE = [3, 2, 3, 3, 5, 5, 2]


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


class MinecraftBCDataset(Dataset):
    """
    Lazy-loading dataset for behavioral cloning.
    Images are loaded from disk on demand — not all at once — so startup is fast
    even with thousands of samples.
    """

    def __init__(self, jsonl_path, simple=False, image_size=84):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

        self.image_size = image_size
        self.obs_list = []
        self.img_paths = []
        self.act_list = []

        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                action = row.get("action", None)
                if not isinstance(action, list) or len(action) < 11:
                    skipped += 1
                    continue

                image_path = _resolve_image_path(row.get("image"), jsonl_path)
                if not image_path or not os.path.exists(image_path):
                    skipped += 1
                    continue

                obs = _state_to_obs13(row.get("state", {}))
                act = np.array(action[:7] if simple else action[:11], dtype=np.int64)

                self.obs_list.append(obs)
                self.img_paths.append(image_path)
                self.act_list.append(act)

        if not self.obs_list:
            raise RuntimeError(
                "No valid rows found in imitation dataset. "
                f"Make sure images exist and actions have >= 11 dims. "
                f"Skipped {skipped} rows."
            )

        print(f"  Dataset: {len(self.obs_list)} samples loaded "
              f"({skipped} skipped, images loaded lazily)")

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs = torch.tensor(self.obs_list[idx], dtype=torch.float32)
        act = torch.tensor(self.act_list[idx], dtype=torch.long)
        image = torch.tensor(
            _load_image_tensor(self.img_paths[idx], self.image_size),
            dtype=torch.float32
        )
        return obs, image, act


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

    def __init__(self, model, action_dims, device, image_size=84, obs_dim=13):
        self.model = model
        self.action_dims = action_dims
        self.device = device
        self.image_size = image_size
        self.obs_dim = int(obs_dim)

    def _adapt_obs_dim(self, obs):
        """Pad or truncate observation to match model input dimension."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] == self.obs_dim:
            return obs
        if obs.shape[0] > self.obs_dim:
            return obs[: self.obs_dim]

        out = np.zeros(self.obs_dim, dtype=np.float32)
        out[: obs.shape[0]] = obs
        return out

    def predict(self, obs, deterministic=True):
        _ = deterministic
        obs = self._adapt_obs_dim(obs)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # During online env eval we currently don't have RGB frames in env obs,
            # so we pass a neutral image. Training still uses real screenshots.
            neutral = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32, device=self.device)
            logits = self.model(x, neutral)
            action = [int(torch.argmax(h, dim=1).item()) for h in logits]
        return np.array(action, dtype=np.int64), None


def train_bc(jsonl_path, save_path="models/bc_minecraft.pt", simple=False,
             epochs=20, batch_size=256, lr=1e-3,
             image_size=84):
    """Train behavioral cloning policy from recorded dataset (CNN + state)."""
    print("=" * 50)
    print("BEHAVIORAL CLONING (CNN + STATE)")
    print("=" * 50)

    action_dims = ACTION_DIMS_SIMPLE if simple else ACTION_DIMS_FULL
    dataset = MinecraftBCDataset(jsonl_path, simple=simple, image_size=image_size)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(dataset, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = len(dataset.obs_list[0])
    model = VisualStateMultiHeadBC(obs_dim=obs_dim, action_dims=action_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Samples: {n_total} (train={n_train}, val={n_val})")
    print(f"Obs dim: {obs_dim}, Action dims: {action_dims}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Device: {device}")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            xb, ib, yb = batch
            xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
            logits = model(xb, ib)
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
                xb, ib, yb = batch
                xb, ib, yb = xb.to(device), ib.to(device), yb.to(device)
                logits = model(xb, ib)
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
        "obs_dim": int(obs_dim),
        "use_images": True,
        "image_size": int(image_size),
    }, save_path)
    print(f"\nBC model saved to {save_path}")

    return BCPredictor(model, action_dims, device, image_size=image_size, obs_dim=obs_dim)


def load_bc_model(path):
    """Load previously trained BC model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    action_dims = ckpt["action_dims"]
    obs_dim = ckpt.get("obs_dim", 13)
    image_size = int(ckpt.get("image_size", 84))
    model = VisualStateMultiHeadBC(obs_dim=obs_dim, action_dims=action_dims)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return BCPredictor(model, action_dims, device, image_size=image_size, obs_dim=obs_dim)


def evaluate_model(env, model, episodes=10, max_steps=300, progress_every=50):
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
            timeout_reached = steps >= max_steps
            done = terminated or truncated or timeout_reached

            if progress_every > 0 and steps % progress_every == 0 and not done:
                print(f"    Ep {ep+1:3d} | step {steps:4d} | running_reward={total_reward:7.2f}")

        if steps >= max_steps:
            print(f"    Ep {ep+1:3d} reached max_steps={max_steps}, truncating eval episode.")

        results.append({"episode": ep + 1, "steps": steps, "reward": round(total_reward, 2)})
        print(f"  Ep {ep+1:3d} | Steps: {steps:4d} | Reward: {total_reward:7.2f}")

    avg_r = np.mean([r["reward"] for r in results])
    print(f"\nAvg reward: {avg_r:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Minecraft Imitation Learning (CNN + state)")
    parser.add_argument("--mode", choices=["bc", "bc_eval"], default="bc",
                        help="Training mode")
    parser.add_argument("--server", default="http://localhost:3001",
                        help="JS bot server URL")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes (for bc_eval)")
    parser.add_argument("--eval_max_steps", type=int, default=300,
                        help="Max steps per evaluation episode (bc_eval)")
    parser.add_argument("--simple", action="store_true",
                        help="Use simplified wood-only action space (7 dims)")
    parser.add_argument("--model_path", default="models/bc_minecraft.pt",
                        help="Path to save/load model")
    parser.add_argument("--jsonl", default="data/train_balanced.jsonl",
                        help="Imitation dataset path (JSONL)")
    parser.add_argument("--bc_epochs", type=int, default=20,
                        help="Behavioral cloning epochs")
    parser.add_argument("--bc_batch_size", type=int, default=256,
                        help="Behavioral cloning batch size")
    parser.add_argument("--bc_lr", type=float, default=1e-3,
                        help="Behavioral cloning learning rate")
    parser.add_argument("--image_size", type=int, default=84,
                        help="Image size for CNN BC (square)")
    args = parser.parse_args()

    try:
        if args.mode == "bc":
            train_bc(
                jsonl_path=args.jsonl,
                save_path=args.model_path,
                simple=args.simple,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
                image_size=args.image_size,
            )

        elif args.mode == "bc_eval":
            EnvClass = MinecraftWoodSimpleEnv if args.simple else MinecraftWoodEnv
            env = EnvClass(server_url=args.server)

            info = env.get_server_info()
            print(f"Connected to server: {args.server}")
            print(f"Action space: MultiDiscrete({list(info['action_space'])})")
            print(f"Observation dim: {info['observation_dim']}")
            print()

            bc_model = load_bc_model(args.model_path)
            evaluate_model(env, bc_model, args.episodes, max_steps=args.eval_max_steps)
            env.close()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        pass


if __name__ == "__main__":
    main()
