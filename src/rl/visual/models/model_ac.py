"""
Arquitectura actor-crítico visual compartida por PPO y SAC (variante discreta).

Reutiliza el mismo `CNNExtractor` que el DQN (model.py) para mantener una
representación visual consistente entre algoritmos.

ActorCritic (PPO)
    Entrada : imagen (C, H, W)  +  estado (STATE_DIM,)
    Salida  : logits (N_ACTIONS,)  +  V(s)  (escalar)

DiscreteActor + DiscreteTwinQ (SAC discreto)
    Actor  : logits sobre acciones discretas
    Twin Q : dos Q-networks Q1(s,·) y Q2(s,·), cada una devuelve un vector
             (N_ACTIONS,) con el Q-value de cada acción.
"""

import sys
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RL_DIR / "shared"))
for _sub in ("visual/algorithms", "visual/models", "visual/train", "visual/eval"):
    sys.path.insert(0, str(_RL_DIR / _sub))

import torch
import torch.nn as nn

from constants import STATE_DIM, ACTIONS, IMG_SIZE
from model import CNNExtractor

N_ACTIONS = len(ACTIONS)


class _SharedTrunk(nn.Module):
    """CNN + fusión opcional con vector de estado → tensor de features compartido."""

    def __init__(self, feat_dim: int, state_dim: int, img_size: int,
                 use_state: bool, in_channels: int):
        super().__init__()
        self.use_state  = use_state
        self.extractor  = CNNExtractor(img_size=img_size, feat_dim=feat_dim,
                                       in_channels=in_channels)
        self.state_proj = nn.Linear(state_dim, feat_dim) if use_state else None

    def forward(self, imgs, states=None):
        feat = self.extractor(imgs)
        if self.use_state and states is not None:
            feat = feat + self.state_proj(states)
        return feat


class ActorCritic(nn.Module):
    """
    Cabezas separadas (actor + crítico) sobre un trunk visual compartido.

    `forward()` devuelve (logits, value). `act()` muestrea una acción y devuelve
    también log_prob y value, lo necesario para el rollout de PPO.
    """

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True,
                 in_channels: int = 3):
        super().__init__()
        self.trunk = _SharedTrunk(feat_dim, state_dim, img_size, use_state, in_channels)

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, imgs, states=None):
        feat   = self.trunk(imgs, states)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, imgs, states=None, deterministic: bool = False):
        logits, value = self.forward(imgs, states)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, logits

    def evaluate(self, imgs, actions, states=None):
        """Usado durante la actualización PPO sobre el batch del rollout."""
        logits, value = self.forward(imgs, states)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value, entropy


class DiscreteActor(nn.Module):
    """Política categórica para SAC discreto."""

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True,
                 in_channels: int = 3):
        super().__init__()
        self.trunk = _SharedTrunk(feat_dim, state_dim, img_size, use_state, in_channels)
        self.head  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, imgs, states=None):
        feat   = self.trunk(imgs, states)
        logits = self.head(feat)
        return logits

    def sample(self, imgs, states=None):
        """Devuelve (action, log_pi(·|s) sobre TODAS las acciones, probs)."""
        logits = self.forward(imgs, states)
        probs  = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs=probs).sample()
        return action, log_probs, probs


class DiscreteQNet(nn.Module):
    """Q-network que devuelve un vector (N, n_actions) para SAC discreto."""

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_actions: int = N_ACTIONS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True,
                 in_channels: int = 3):
        super().__init__()
        self.trunk = _SharedTrunk(feat_dim, state_dim, img_size, use_state, in_channels)
        self.head  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, imgs, states=None):
        return self.head(self.trunk(imgs, states))
