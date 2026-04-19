"""
Arquitectura visual para el agente DQN.

VisualQNetwork
  Entrada : imagen (C, H, W)  +  estado (STATE_DIM,)
  Salida  : Q-values (N_ACTIONS,)

Pipeline:
  imagen → CNN (3 bloques conv) → flatten → proj lineal → img_feat
  img_feat + state_proj(estado)  → fusión por suma
  fusión → MLP → Q-values
"""

import sys
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))

import torch
import torch.nn as nn

from constants import STATE_DIM, ACTIONS, IMG_SIZE

N_ACTIONS = len(ACTIONS)


class CNNExtractor(nn.Module):
    """
    CNN ligera para extraer features de un frame RGB.

    Entrada : (N, 3, H, W)
    Salida  : (N, feat_dim)

    3 bloques Conv→BN→ReLU→MaxPool reducen 128×128 → 16×16.
    """

    def __init__(self, img_size: int = IMG_SIZE, feat_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim

        self.cnn = nn.Sequential(
            # 128 → 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # 64 → 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # 32 → 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        reduced = img_size // 8          # 128//8 = 16
        flat_dim = 128 * reduced * reduced  # 128 * 16 * 16 = 32768
        self.proj = nn.Linear(flat_dim, feat_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (N, 3, H, W) → (N, feat_dim)"""
        x = self.cnn(imgs)
        x = x.flatten(1)
        return self.proj(x)


class VisualQNetwork(nn.Module):
    """
    Q-network visual con fusión opcional de estado.

    use_state=True  → img_feat + state_proj(state) → MLP → Q-values
    use_state=False → img_feat                     → MLP → Q-values
    """

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True):
        super().__init__()
        self.use_state  = use_state
        self.extractor  = CNNExtractor(img_size=img_size, feat_dim=feat_dim)
        self.state_proj = nn.Linear(state_dim, feat_dim) if use_state else None

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, imgs: torch.Tensor,
                states: torch.Tensor | None = None) -> torch.Tensor:
        """
        imgs   : (N, 3, H, W)
        states : (N, STATE_DIM)  — ignorado si use_state=False
        → Q    : (N, N_ACTIONS)
        """
        img_feat = self.extractor(imgs)
        if self.use_state and states is not None:
            img_feat = img_feat + self.state_proj(states)
        return self.mlp(img_feat)
