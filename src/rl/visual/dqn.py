"""
DQN visual para Minecraft woodcutting.

Observación: Dict {"state": (STATE_DIM,), "image": (3, H, W)}
Acciones: discretas (ver ACTIONS en shared/constants.py)
"""

import sys
import random
from collections import deque
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from constants import STATE_DIM, ACTIONS, IMG_SIZE
from model import VisualQNetwork

N_ACTIONS = len(ACTIONS)


class VisualReplayBuffer:
    """Buffer circular que almacena transiciones (img, state, action, reward, next_img, next_state, done)."""

    def __init__(self, capacity: int = 20_000):
        self._buf = deque(maxlen=capacity)

    def push(self, img, state, action: int, reward: float,
             next_img, next_state, done: bool):
        self._buf.append((
            np.array(img,        dtype=np.float32),
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_img,   dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        imgs, states, actions, rewards, next_imgs, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(imgs),        dtype=torch.float32),
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_imgs),   dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


class VisualDQNAgent:
    """
    DQN visual con target network y epsilon-greedy.

    use_state=True  → observación imagen + vector de estado
    use_state=False → solo imagen (estado se ignora en la red, pero sigue
                      guardándose en el buffer por compatibilidad)
    """

    def __init__(
        self,
        feat_dim:      int   = 256,
        state_dim:     int   = STATE_DIM,
        n_actions:     int   = N_ACTIONS,
        hidden:        int   = 256,
        img_size:      int   = IMG_SIZE,
        use_state:     bool  = True,
        lr:            float = 1e-4,
        gamma:         float = 0.99,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 10_000,
        target_update: int   = 200,
        batch_size:    int   = 32,
        buffer_size:   int   = 20_000,
    ):
        self.n_actions     = n_actions
        self.use_state     = use_state
        self.gamma         = gamma
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.target_update = target_update
        self.batch_size    = batch_size
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = VisualQNetwork(feat_dim, state_dim, n_actions, hidden, img_size, use_state).to(self.device)
        self.target_net = VisualQNetwork(feat_dim, state_dim, n_actions, hidden, img_size, use_state).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = VisualReplayBuffer(buffer_size)
        self._step     = 0

    @property
    def epsilon(self) -> float:
        t = min(self._step / self.eps_decay, 1.0)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - t)

    def _obs_to_tensors(self, obs: dict):
        """Convierte observación del env (dict) a tensors normalizados en device."""
        img   = torch.tensor(obs["image"],  dtype=torch.float32, device=self.device)
        state = torch.tensor(obs["state"],  dtype=torch.float32, device=self.device)
        # Normalizar imagen de uint8 [0,255] a float [0,1]
        if img.max() > 1.0:
            img = img / 255.0
        return img, state

    def select_action(self, obs: dict) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        img, state = self._obs_to_tensors(obs)
        with torch.no_grad():
            q = self.q_net(img.unsqueeze(0),
                           state.unsqueeze(0) if self.use_state else None)
        return int(q.argmax(dim=1).item())

    def step(self, obs: dict, action: int, reward: float,
             next_obs: dict, done: bool) -> float | None:
        img,      state      = obs["image"],      obs["state"]
        next_img, next_state = next_obs["image"], next_obs["state"]

        # Normalizar antes de guardar
        img      = np.array(img,      dtype=np.float32) / 255.0 if img.max() > 1.0 else np.array(img,      dtype=np.float32)
        next_img = np.array(next_img, dtype=np.float32) / 255.0 if next_img.max() > 1.0 else np.array(next_img, dtype=np.float32)

        self.buffer.push(img, state, action, reward, next_img, next_state, done)
        self._step += 1

        if self._step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if len(self.buffer) < self.batch_size:
            return None

        return self._train_batch()

    def _train_batch(self) -> float:
        imgs, states, actions, rewards, next_imgs, next_states, dones = \
            self.buffer.sample(self.batch_size)

        imgs       = imgs.to(self.device)
        states     = states.to(self.device)
        actions    = actions.to(self.device)
        rewards    = rewards.to(self.device)
        next_imgs  = next_imgs.to(self.device)
        next_states= next_states.to(self.device)
        dones      = dones.to(self.device)

        s      = states      if self.use_state else None
        next_s = next_states if self.use_state else None

        q_values = self.q_net(imgs, s).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q   = self.target_net(next_imgs, next_s).max(dim=1).values
            q_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_net":   self.q_net.state_dict(),
            "step":    self._step,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self._step = ckpt.get("step", 0)
