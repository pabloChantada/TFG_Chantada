"""
Agente SAC (Soft Actor-Critic) — variante DISCRETA.

Adaptación del paper original (Haarnoja 2018, continuo) al dominio discreto
siguiendo Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings".

Diferencias clave frente a SAC continuo:
- El actor produce logits sobre N acciones; las probabilidades π(a|s) se
  obtienen vía softmax y se calcula la esperanza analíticamente (no muestreo).
- Las Q-networks devuelven un vector (N, n_actions) — el Q-value de cada
  acción dado el estado, igual que en DQN.
- La entropía se calcula como Σ π(a|s) · log π(a|s) (forma cerrada).
- Target de Q:  r + γ · Σ π(a'|s') · (Q_target_min(s',a') - α · log π(a'|s'))

Hereda de DQN la idea de replay buffer + target networks, pero:
- Twin Q (Q1 y Q2) — mitiga sobreestimación tomando el mínimo.
- Soft update de los targets (Polyak τ ≈ 0.005).
- Temperatura α auto-ajustable (entropy target = factor · log(n_actions)).
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
from model_ac import DiscreteActor, DiscreteQNet

N_ACTIONS = len(ACTIONS)


class ReplayBuffer:
    """Buffer circular para SAC. Mismo formato que el de DQN visual."""

    def __init__(self, capacity: int = 100_000):
        self._buf = deque(maxlen=capacity)

    def push(self, img, state, action, reward, next_img, next_state, done):
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


class DiscreteSACAgent:
    """
    SAC discreto con twin Q, target soft-update, y temperatura α auto-ajustable.

    Hiperparámetros (Christodoulou 2019 + ajustes para entornos lentos):
        tau                = 0.005   # soft update
        target_entropy_pct = 0.98    # frac. de log(n_actions) — exploración alta
        lr_actor           = 3e-4
        lr_critic          = 3e-4
        lr_alpha           = 3e-4
    """

    def __init__(
        self,
        feat_dim:         int   = 256,
        hidden:           int   = 256,
        use_state:        bool  = True,
        img_channels:     int   = 12,
        lr_actor:         float = 3e-4,
        lr_critic:        float = 3e-4,
        lr_alpha:         float = 3e-4,
        gamma:            float = 0.99,
        tau:              float = 0.005,
        batch_size:       int   = 64,
        buffer_size:      int   = 100_000,
        warmup_steps:     int   = 1_000,
        target_entropy_pct:float = 0.98,
        reward_clip:      float | None = 1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor    = DiscreteActor(feat_dim=feat_dim, hidden=hidden,
                                      use_state=use_state, in_channels=img_channels).to(self.device)
        self.q1       = DiscreteQNet(feat_dim=feat_dim, hidden=hidden,
                                     use_state=use_state, in_channels=img_channels).to(self.device)
        self.q2       = DiscreteQNet(feat_dim=feat_dim, hidden=hidden,
                                     use_state=use_state, in_channels=img_channels).to(self.device)
        self.q1_targ  = DiscreteQNet(feat_dim=feat_dim, hidden=hidden,
                                     use_state=use_state, in_channels=img_channels).to(self.device)
        self.q2_targ  = DiscreteQNet(feat_dim=feat_dim, hidden=hidden,
                                     use_state=use_state, in_channels=img_channels).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        for p in self.q1_targ.parameters(): p.requires_grad_(False)
        for p in self.q2_targ.parameters(): p.requires_grad_(False)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),     lr=lr_actor)
        self.q1_opt     = torch.optim.Adam(self.q1.parameters(),        lr=lr_critic)
        self.q2_opt     = torch.optim.Adam(self.q2.parameters(),        lr=lr_critic)

        # Temperatura α auto-ajustable. Optimizamos log_alpha para garantizar α>0.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(np.log(1.0 / N_ACTIONS) * target_entropy_pct)

        self.gamma          = gamma
        self.tau            = tau
        self.batch_size     = batch_size
        self.warmup_steps   = warmup_steps
        self.use_state      = use_state
        self.img_channels   = img_channels
        self.reward_clip    = reward_clip

        self.buffer = ReplayBuffer(buffer_size)
        self._step  = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> int:
        img   = self._normalize_img(obs["image"])
        img_t   = torch.as_tensor(img,           dtype=torch.float32, device=self.device).unsqueeze(0)
        state_t = (torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device).unsqueeze(0)
                   if self.use_state else None)
        logits = self.actor(img_t, state_t)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    @staticmethod
    def _normalize_img(img):
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def step(self, obs: dict, action: int, reward: float,
             next_obs: dict, done: bool) -> dict | None:
        img      = self._normalize_img(obs["image"])
        next_img = self._normalize_img(next_obs["image"])

        train_reward = reward
        if self.reward_clip is not None:
            train_reward = max(-self.reward_clip, min(self.reward_clip, reward))

        self.buffer.push(img, obs["state"], action, train_reward,
                         next_img, next_obs["state"], done)
        self._step += 1

        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None
        return self._train_batch()

    def _train_batch(self) -> dict:
        imgs, states, actions, rewards, next_imgs, next_states, dones = \
            self.buffer.sample(self.batch_size)

        imgs        = imgs.to(self.device)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_imgs   = next_imgs.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        s      = states      if self.use_state else None
        next_s = next_states if self.use_state else None
        alpha  = self.log_alpha.exp().detach()

        # ── Target Q (sin gradiente) ──────────────────────────────────────────
        with torch.no_grad():
            next_logits   = self.actor(next_imgs, next_s)
            next_probs    = torch.softmax(next_logits, dim=-1)
            next_log_probs= torch.log_softmax(next_logits, dim=-1)
            q1_next = self.q1_targ(next_imgs, next_s)
            q2_next = self.q2_targ(next_imgs, next_s)
            q_min   = torch.min(q1_next, q2_next)
            # Esperanza sobre acciones: V(s') = Σ π · (Q - α·log π)
            v_next  = (next_probs * (q_min - alpha * next_log_probs)).sum(dim=-1)
            q_target = rewards + self.gamma * v_next * (1.0 - dones)

        # ── Critics: MSE contra el target en la acción ejecutada ──────────────
        q1_all = self.q1(imgs, s)
        q2_all = self.q2(imgs, s)
        q1_a   = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a   = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_q1 = F.mse_loss(q1_a, q_target)
        loss_q2 = F.mse_loss(q2_a, q_target)

        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        # ── Actor: maximizar Σ π · (α·log π - Q_min) (gradient ascent del retorno) ─
        logits     = self.actor(imgs, s)
        probs      = torch.softmax(logits, dim=-1)
        log_probs  = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q_min_a = torch.min(self.q1(imgs, s), self.q2(imgs, s))
        loss_actor = (probs * (alpha * log_probs - q_min_a)).sum(dim=-1).mean()

        self.actor_opt.zero_grad(); loss_actor.backward(); self.actor_opt.step()

        # ── Temperatura α: ajustar para alcanzar la entropía objetivo ─────────
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss_alpha = -(self.log_alpha * (self.target_entropy - entropy).detach())
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()

        # ── Soft update de los target networks ────────────────────────────────
        self._polyak(self.q1, self.q1_targ)
        self._polyak(self.q2, self.q2_targ)

        return {
            "loss_q1":     loss_q1.item(),
            "loss_q2":     loss_q2.item(),
            "loss_actor":  loss_actor.item(),
            "loss_alpha":  loss_alpha.item(),
            "alpha":       self.alpha,
            "entropy":     entropy.item(),
        }

    def _polyak(self, online: torch.nn.Module, target: torch.nn.Module):
        with torch.no_grad():
            for p, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    @torch.no_grad()
    def weight_max_abs(self) -> dict:
        out = {}
        for name, p in self.actor.named_parameters():
            if name.startswith("head.") and name.endswith(".weight"):
                out[f"actor.{name}"] = float(p.abs().max().item())
        for name, p in self.q1.named_parameters():
            if name.startswith("head.") and name.endswith(".weight"):
                out[f"q1.{name}"] = float(p.abs().max().item())
        return out

    def save(self, path: str):
        torch.save({
            "actor":        self.actor.state_dict(),
            "q1":           self.q1.state_dict(),
            "q2":           self.q2.state_dict(),
            "q1_targ":      self.q1_targ.state_dict(),
            "q2_targ":      self.q2_targ.state_dict(),
            "log_alpha":    self.log_alpha.detach().cpu(),
            "step":         self._step,
            "img_channels": self.img_channels,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_targ.load_state_dict(ckpt["q1_targ"])
        self.q2_targ.load_state_dict(ckpt["q2_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        self._step = ckpt.get("step", 0)
