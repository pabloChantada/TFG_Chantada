"""
DQN para Minecraft woodcutting.

Estado: 6D  [yaw, pitch, dx, dz, tree_visible, tree_distance]  (todos en [-1, 1])
Acciones: 5 discretas  (ver ACTIONS en constants.py)

Componentes:
  - QNetwork      : MLP  state → Q(s, a)
  - ReplayBuffer  : buffer circular con muestreo uniforme
  - DQNAgent      : epsilon-greedy, target network, paso de entrenamiento
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import STATE_DIM, ACTIONS

N_ACTIONS = len(ACTIONS)


# ── Red Q ─────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """MLP  state (STATE_DIM,) → Q-values (N_ACTIONS,)."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self._buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


# ── Agente DQN ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN con target network y epsilon-greedy.

    Parámetros
    ----------
    lr            : learning rate
    gamma         : factor de descuento
    eps_start     : epsilon inicial (exploración)
    eps_end       : epsilon mínimo
    eps_decay     : nº de steps hasta llegar a eps_end (decay lineal)
    target_update : cada cuántos steps se sincroniza la target network
    batch_size    : tamaño de batch para el paso de optimización
    buffer_size   : capacidad del replay buffer
    """

    def __init__(
        self,
        state_dim:     int   = STATE_DIM,
        n_actions:     int   = N_ACTIONS,
        hidden:        int   = 128,
        lr:            float = 1e-3,
        gamma:         float = 0.99,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 5_000,
        target_update: int   = 200,
        batch_size:    int   = 64,
        buffer_size:   int   = 50_000,
        device:        str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.target_update = target_update
        self.batch_size    = batch_size
        self.device        = torch.device(device)

        self.q_net      = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self._step = 0   # contador global de steps (para epsilon y target update)

    # ── Epsilon actual ────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        t = min(self._step / self.eps_decay, 1.0)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - t)

    # ── Selección de acción ───────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    # ── Paso de entrenamiento ─────────────────────────────────────────────────

    def step(self, state, action, reward, next_state, done) -> float | None:
        """Guarda la transición y, si hay suficientes muestras, entrena un batch.

        Devuelve la loss del batch o None si aún no hay suficientes muestras.
        """
        self.buffer.push(state, action, reward, next_state, done)
        self._step += 1

        # Sincronizar target network
        if self._step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if len(self.buffer) < self.batch_size:
            return None

        return self._train_batch()

    def _train_batch(self) -> float:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q(s, a) actual
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a')  (0 si done)
        with torch.no_grad():
            next_q   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    # ── Guardar / cargar ──────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net":    self.q_net.state_dict(),
            "step":     self._step,
            "epsilon":  self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self._step = ckpt.get("step", 0)
