"""
BC pretrain del HybridActor sobre el dataset MineRL Treechop-v0.

Objetivo: arrancar el actor de Hybrid SAC con un sesgo razonable derivado de
demostraciones humanas (~453k transiciones), antes de fine-tune online en
Mineflayer.

Pérdidas:
    BCEWithLogits sobre los 4 flags (flag_logits vs flags humanos)
    MSE sobre los 2 canales de cámara (cam_mu vs dyaw, dpitch en radianes)
    El log_std no se entrena (la política BC es determinista; el muestreo SAC
    arranca explorando con el log_std inicial del actor — eso lo regula α
    durante fine-tune).

Estado: durante BC se pasan ceros — MineRL no expone tree_visible/distance/etc.
El `state_proj` del trunk aprende implícitamente a no contribuir; cuando SAC
empiece online con state real, el SAC lo recalibra.

Uso:
    python src/rl/minerl/bc_pretrain.py [--epochs 3] [--batch-size 64]
        [--lr 3e-4] [--limit-eps N] [--out checkpoints/bc_hybrid.pth]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(_RL_DIR / "visual" / "models"))
sys.path.insert(0, str(_RL_DIR / "visual" / "algorithms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (HYBRID_FLAGS, N_HYBRID_FLAGS, IMG_SIZE,
                       IMAGENET_MEAN, IMAGENET_STD, STATE_DIM)
from model_hybrid import HybridActor, CAMERA_DIM
from map_actions import list_episodes, load_episode_actions

DATASET_ROOT  = "data/MineRLTreechop-v0/MineRLTreechop-v0"
DEFAULT_OUT   = "src/rl/visual/runs/bc_hybrid.pth"
FRAME_STACK   = 4
NORM_MEAN     = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
NORM_STD      = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default=DATASET_ROOT)
    p.add_argument("--out",         default=DEFAULT_OUT)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--feat-dim",    type=int,   default=256)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--limit-eps",   type=int,   default=None,
                   help="Limita nº de episodios (debug).")
    p.add_argument("--cam-loss-weight",  type=float, default=1.0)
    p.add_argument("--flag-loss-weight", type=float, default=1.0)
    p.add_argument("--log-every",   type=int,   default=200)
    return p.parse_args()


def _preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) → CHW float32 normalizado a (3, IMG_SIZE, IMG_SIZE)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0           # (H, W, 3)
    arr = arr.transpose(2, 0, 1)                   # (3, H, W)
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr


class MineRLBCDataset(IterableDataset):
    """
    Stream de samples (img_stacked, state_zero, action_vec) sobre el dataset.

    Cada episodio:
      - lee frames del MP4 secuencialmente
      - mantiene un deque de `frame_stack` frames previos
      - emite un sample por step alineado con `action_vec[t]`
    """

    def __init__(self, dataset_root: str, frame_stack: int = FRAME_STACK,
                 limit_eps: int | None = None, shuffle_episodes: bool = True):
        self.episodes    = list_episodes(dataset_root)
        if limit_eps:
            self.episodes = self.episodes[:limit_eps]
        self.frame_stack = frame_stack
        self.shuffle     = shuffle_episodes

    def __iter__(self):
        order = np.arange(len(self.episodes))
        if self.shuffle:
            np.random.shuffle(order)
        for ep_idx in order:
            ep_dir = self.episodes[ep_idx]
            yield from self._iter_episode(ep_dir)

    def _iter_episode(self, ep_dir: Path):
        actions = load_episode_actions(ep_dir / "rendered.npz")  # (T, 6)
        T = actions.shape[0]

        cap = cv2.VideoCapture(str(ep_dir / "recording.mp4"))
        try:
            stack: list[np.ndarray] = []   # FIFO de frames procesados (CHW)
            for t in range(T):
                ok, frame = cap.read()
                if not ok:
                    break
                proc = _preprocess_frame(frame)              # (3, H, W)
                if not stack:
                    stack = [proc] * self.frame_stack        # bootstrap
                else:
                    stack.append(proc)
                    if len(stack) > self.frame_stack:
                        stack.pop(0)

                stacked = np.concatenate(stack, axis=0)      # (3*FS, H, W)
                yield (
                    torch.from_numpy(stacked),
                    torch.zeros(STATE_DIM, dtype=torch.float32),
                    torch.from_numpy(actions[t]),
                )
        finally:
            cap.release()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Flags: {HYBRID_FLAGS}  |  cam_dim: {CAMERA_DIM}")

    actor = HybridActor(
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        n_flags=N_HYBRID_FLAGS,
        use_state=True,
        in_channels=3 * FRAME_STACK,
    ).to(device)

    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    ds = MineRLBCDataset(args.dataset, frame_stack=FRAME_STACK,
                         limit_eps=args.limit_eps, shuffle_episodes=True)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # Estimación de pasos por epoch (sin tocar disco)
    total_eps = args.limit_eps or len(list_episodes(args.dataset))
    print(f"Episodios: {total_eps}  |  epochs: {args.epochs}  "
          f"|  batch: {args.batch_size}  |  lr: {args.lr}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        run_loss_flag = run_loss_cam = 0.0
        run_count = 0
        ep_t0 = time.time()
        for batch in loader:
            imgs, states, actions = (b.to(device, non_blocking=True) for b in batch)
            target_flags = actions[:, :N_HYBRID_FLAGS]
            target_cam   = actions[:, N_HYBRID_FLAGS:]

            flag_logits, cam_mu, _ = actor(imgs, states)
            loss_flag = F.binary_cross_entropy_with_logits(flag_logits, target_flags)
            loss_cam  = F.mse_loss(cam_mu, target_cam)
            loss = args.flag_loss_weight * loss_flag + args.cam_loss_weight * loss_cam

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()

            run_loss_flag += loss_flag.item()
            run_loss_cam  += loss_cam.item()
            run_count     += 1
            global_step   += 1

            if global_step % args.log_every == 0:
                avg_f = run_loss_flag / run_count
                avg_c = run_loss_cam  / run_count
                elapsed = time.time() - t0
                print(f"  ep {ep}  step {global_step:>6d}  "
                      f"BCE_flag={avg_f:.4f}  MSE_cam={avg_c:.5f}  "
                      f"elapsed={elapsed/60:.1f}min")
                run_loss_flag = run_loss_cam = 0.0
                run_count = 0

        ep_time = time.time() - ep_t0
        print(f"-- Epoch {ep}/{args.epochs} done en {ep_time/60:.1f}min --")

        # Checkpoint por epoch
        ckpt_path = out_path.with_name(out_path.stem + f"_ep{ep}.pth")
        torch.save({
            "actor":        actor.state_dict(),
            "kind":         "hybrid_bc",
            "frame_stack":  FRAME_STACK,
            "n_flags":      N_HYBRID_FLAGS,
            "feat_dim":     args.feat_dim,
            "hidden":       args.hidden,
        }, ckpt_path)
        print(f"  -> checkpoint: {ckpt_path}")

    # Checkpoint final con el nombre solicitado
    torch.save({
        "actor":        actor.state_dict(),
        "kind":         "hybrid_bc",
        "frame_stack":  FRAME_STACK,
        "n_flags":      N_HYBRID_FLAGS,
        "feat_dim":     args.feat_dim,
        "hidden":       args.hidden,
    }, out_path)
    print(f"\n[OK] BC pretrain final: {out_path}")


if __name__ == "__main__":
    train(parse_args())
