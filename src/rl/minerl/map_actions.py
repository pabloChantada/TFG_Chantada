"""
Mapping del dataset MineRL Treechop-v0 al espacio de acciones de Hybrid SAC.

Espacio destino:
    flags  : 4 binarios = [forward, jump, sprint, attack]   (HYBRID_FLAGS)
    camera : (dyaw, dpitch) en RADIANES, clipeado a [-CAMERA_MAX_RAD, +CAMERA_MAX_RAD]

Notas:
- MineRL guarda la cámara como (dpitch, dyaw) en GRADOS — los convertimos.
- back/left/right/sneak (<2.5% del dataset) se descartan: el flag se pone a 0.
  No filtramos esos steps porque la imagen + cámara siguen siendo señal útil.
- Las muestras se devuelven como vectores numpy listos para el ReplayBuffer
  o el dataloader BC.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

import sys
_RL_SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(_RL_SHARED))

from constants import HYBRID_FLAGS, N_HYBRID_FLAGS, CAMERA_MAX_RAD  # noqa: E402

DEG2RAD = math.pi / 180.0


def list_episodes(dataset_root: str | Path) -> list[Path]:
    """Lista los subdirectorios con `rendered.npz` bajo `dataset_root`."""
    root = Path(dataset_root)
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "rendered.npz").exists())


def load_episode_actions(npz_path: str | Path) -> np.ndarray:
    """
    Devuelve un array (T, N_HYBRID_FLAGS + 2) con la trayectoria de acciones
    híbridas de un episodio. NO carga los frames de imagen (se hace aparte).
    """
    d = np.load(npz_path)

    T = int(d["reward"].shape[0])
    out = np.zeros((T, N_HYBRID_FLAGS + 2), dtype=np.float32)

    # Flags binarios — solo los que existen en HYBRID_FLAGS
    for i, flag in enumerate(HYBRID_FLAGS):
        key = f"action${flag}"
        if key in d.files:
            out[:, i] = d[key].astype(np.float32)

    # Cámara: dataset en grados, queremos radianes y clipeado al bound del actor
    cam_deg = d["action$camera"].astype(np.float32)        # (T, 2): pitch, yaw
    cam_rad = cam_deg * DEG2RAD
    cam_rad = np.clip(cam_rad, -CAMERA_MAX_RAD, CAMERA_MAX_RAD)

    # Convención del actor: action_vec[..., -2:] = (dyaw, dpitch)
    out[:, N_HYBRID_FLAGS + 0] = cam_rad[:, 1]   # dyaw
    out[:, N_HYBRID_FLAGS + 1] = cam_rad[:, 0]   # dpitch

    return out


def load_episode_meta(npz_path: str | Path) -> dict:
    """Devuelve rewards y dones para un episodio (para llenar el replay buffer)."""
    d = np.load(npz_path)
    rewards = d["reward"].astype(np.float32)
    dones   = np.zeros_like(rewards, dtype=np.float32)
    dones[-1] = 1.0
    return {"reward": rewards, "done": dones}


def iter_episodes(dataset_root: str | Path) -> Iterable[tuple[Path, np.ndarray, dict]]:
    """Itera (path, actions (T, 6), meta) sobre todos los episodios."""
    for ep in list_episodes(dataset_root):
        npz = ep / "rendered.npz"
        actions = load_episode_actions(npz)
        meta    = load_episode_meta(npz)
        yield ep, actions, meta


def summary(dataset_root: str | Path) -> dict:
    """Resumen rápido para verificación: count, distribución de flags, magnitudes cámara."""
    eps = list_episodes(dataset_root)
    n_steps  = 0
    flag_sum = np.zeros(N_HYBRID_FLAGS, dtype=np.int64)
    cam_abs_sum  = np.zeros(2, dtype=np.float64)
    cam_active   = 0
    for ep in eps:
        a = load_episode_actions(ep / "rendered.npz")
        n_steps += a.shape[0]
        flag_sum += a[:, :N_HYBRID_FLAGS].astype(np.int64).sum(axis=0)
        cam = a[:, N_HYBRID_FLAGS:]
        cam_abs_sum += np.abs(cam).sum(axis=0)
        cam_active  += int((np.abs(cam).max(axis=1) > 1e-4).sum())
    return {
        "n_episodes":   len(eps),
        "n_steps":      int(n_steps),
        "flag_pct":     {HYBRID_FLAGS[i]: 100.0 * flag_sum[i] / max(1, n_steps)
                         for i in range(N_HYBRID_FLAGS)},
        "cam_active_pct": 100.0 * cam_active / max(1, n_steps),
        "cam_abs_mean_rad": (cam_abs_sum / max(1, n_steps)).tolist(),
    }


if __name__ == "__main__":
    import json
    s = summary("data/MineRLTreechop-v0/MineRLTreechop-v0")
    print(json.dumps(s, indent=2))
