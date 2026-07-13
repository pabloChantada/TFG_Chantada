"""
Hybrid SAC — variante para acción híbrida (Bernoulli factorizada × Gaussiana 2D).

Esquema:
  - Actor reparametrizado (cámara) + Bernoulli no-reparam (flags). Las gradientes
    fluyen por la cámara; los flags se entrenan vía la corriente directa de los
    log-probs en el objetivo del actor (variante "score function" para los flags).
  - Twin Q (Q1, Q2) con target soft-update τ.
  - Temperatura α auto-ajustable hacia un objetivo de entropía.
    target_entropy = - target_pct · ( log(2)·N_FLAGS + DIM_CAM·log(2π·e·σ₀²)/2 )
    Aproximación: usamos un objetivo simple proporcional al máximo teórico de
    entropía Bernoulli + Gaussiana de referencia (ver `_default_target_entropy`).

Diseñado para soportar BC pretrain del actor sin modificar el agente: cargas
solo la parte `actor` del checkpoint y entrenas Q desde cero (warm-up).
"""

import sys
import random
from collections import deque
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RL_DIR / "shared"))
for _sub in ("visual/algorithms", "visual/models", "visual/train", "visual/eval"):
    sys.path.insert(0, str(_RL_DIR / _sub))

import math
import numpy as np
import torch
import torch.nn.functional as F

from constants import STATE_DIM, IMG_SIZE, N_HYBRID_FLAGS, CAMERA_MAX_RAD
from model_hybrid import HybridActor, HybridQNet, CAMERA_DIM, ACTION_VEC_DIM


def _default_target_entropy(n_flags: int, sigma_ref: float = 0.3) -> float:
    """
    Objetivo de entropía razonable para una política híbrida.

    - Bernoulli: H_max por flag = log(2) ≈ 0.693
    - Gaussiana 2D (referencia σ=sigma_ref·CAMERA_MAX_RAD):
        H = 0.5·log(2π·e·σ²) por dimensión
    Retorna ~70% del máximo Bernoulli + entropía Gaussiana de referencia.
    """
    h_bern = 0.7 * math.log(2.0) * n_flags
    sigma  = sigma_ref * CAMERA_MAX_RAD
    h_gauss = CAMERA_DIM * 0.5 * math.log(2.0 * math.pi * math.e * sigma**2)
    return h_bern + h_gauss


class HybridReplayBuffer:
    """Buffer con action_vec híbrido (flags 0/1 concatenados con cámara continua)."""

    def __init__(self, capacity: int = 100_000):
        self._buf = deque(maxlen=capacity)

    def push(self, img, state, action_vec, reward, next_img, next_state, done):
        self._buf.append((
            np.asarray(img,        dtype=np.float32),
            np.asarray(state,      dtype=np.float32),
            np.asarray(action_vec, dtype=np.float32),
            float(reward),
            np.asarray(next_img,   dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        imgs, states, actions, rewards, next_imgs, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(imgs),        dtype=torch.float32),
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.float32),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_imgs),   dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


class HybridSACAgent:
    """
    SAC híbrido (Bernoulli×Gaussiana) con twin Q + soft target + α auto-ajustable.
    """

    def __init__(
        self,
        feat_dim:         int   = 256,
        hidden:           int   = 256,
        use_state:        bool  = True,
        img_channels:     int   = 12,
        n_flags:          int   = N_HYBRID_FLAGS,
        camera_max_rad:   float = CAMERA_MAX_RAD,
        lr_actor:         float = 3e-4,
        lr_critic:        float = 3e-4,
        lr_alpha:         float = 3e-4,
        gamma:            float = 0.99,
        tau:              float = 0.005,
        batch_size:       int   = 64,
        buffer_size:      int   = 200_000,
        warmup_steps:     int   = 1_000,
        target_entropy:   float | None = None,
        reward_clip:      float | None = 1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor   = HybridActor(feat_dim=feat_dim, hidden=hidden, n_flags=n_flags,
                                   use_state=use_state, in_channels=img_channels,
                                   camera_max_rad=camera_max_rad).to(self.device)
        self.q1      = HybridQNet(feat_dim=feat_dim, hidden=hidden, n_flags=n_flags,
                                  use_state=use_state, in_channels=img_channels).to(self.device)
        self.q2      = HybridQNet(feat_dim=feat_dim, hidden=hidden, n_flags=n_flags,
                                  use_state=use_state, in_channels=img_channels).to(self.device)
        self.q1_targ = HybridQNet(feat_dim=feat_dim, hidden=hidden, n_flags=n_flags,
                                  use_state=use_state, in_channels=img_channels).to(self.device)
        self.q2_targ = HybridQNet(feat_dim=feat_dim, hidden=hidden, n_flags=n_flags,
                                  use_state=use_state, in_channels=img_channels).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        for p in self.q1_targ.parameters(): p.requires_grad_(False)
        for p in self.q2_targ.parameters(): p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q1_opt    = torch.optim.Adam(self.q1.parameters(),    lr=lr_critic)
        self.q2_opt    = torch.optim.Adam(self.q2.parameters(),    lr=lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = target_entropy if target_entropy is not None \
                              else _default_target_entropy(n_flags)

        self.gamma         = gamma
        self.tau           = tau
        self.batch_size    = batch_size
        self.warmup_steps  = warmup_steps
        self.use_state     = use_state
        self.img_channels  = img_channels
        self.n_flags       = n_flags
        self.camera_max_rad = camera_max_rad
        self.reward_clip   = reward_clip

        self.buffer = HybridReplayBuffer(buffer_size)
        self._step  = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """Devuelve action_vec en numpy (n_flags + 2)."""
        img   = self._normalize_img(obs["image"])
        img_t = torch.as_tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0)
        st_t  = (torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device).unsqueeze(0)
                 if self.use_state else None)
        action_vec, _, _ = self.actor.act(img_t, st_t, deterministic=deterministic)
        return action_vec.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _normalize_img(img):
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def step(self, obs: dict, action_vec: np.ndarray, reward: float,
             next_obs: dict, done: bool) -> dict | None:
        img      = self._normalize_img(obs["image"])
        next_img = self._normalize_img(next_obs["image"])

        train_reward = reward
        if self.reward_clip is not None:
            train_reward = max(-self.reward_clip, min(self.reward_clip, reward))

        self.buffer.push(img, obs["state"], action_vec, train_reward,
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
            next_action, next_log_pi, _ = self.actor.sample(next_imgs, next_s)
            q1_next = self.q1_targ(next_imgs, next_action, next_s)
            q2_next = self.q2_targ(next_imgs, next_action, next_s)
            q_min   = torch.min(q1_next, q2_next)
            q_target = rewards + self.gamma * (q_min - alpha * next_log_pi) * (1.0 - dones)

        # ── Critic loss ──────────────────────────────────────────────────────
        q1_pred = self.q1(imgs, actions, s)
        q2_pred = self.q2(imgs, actions, s)
        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        # ── Actor loss ───────────────────────────────────────────────────────
        # Reparametrización para la cámara; los flags se muestrean (no-reparam)
        # pero su log_prob entra en el objetivo y el gradiente fluye por flag_logits
        # vía el término α·log_pi (rama "score function").
        action, log_pi, _ = self.actor.sample(imgs, s)
        q1_pi  = self.q1(imgs, action, s)
        q2_pi  = self.q2(imgs, action, s)
        q_min_pi = torch.min(q1_pi, q2_pi)
        loss_actor = (alpha * log_pi - q_min_pi).mean()

        self.actor_opt.zero_grad(); loss_actor.backward(); self.actor_opt.step()

        # ── α auto-ajuste ────────────────────────────────────────────────────
        with torch.no_grad():
            entropy = -log_pi.mean()
        loss_alpha = -(self.log_alpha * (self.target_entropy - entropy).detach())
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()

        # ── Soft update ──────────────────────────────────────────────────────
        self._polyak(self.q1, self.q1_targ)
        self._polyak(self.q2, self.q2_targ)

        return {
            "loss_q1":    loss_q1.item(),
            "loss_q2":    loss_q2.item(),
            "loss_actor": loss_actor.item(),
            "loss_alpha": loss_alpha.item(),
            "alpha":      self.alpha,
            "entropy":    entropy.item(),
        }

    def _polyak(self, online, target):
        with torch.no_grad():
            for p, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    @torch.no_grad()
    def weight_max_abs(self) -> dict:
        out = {}
        for name, p in self.actor.named_parameters():
            if name.endswith(".weight") and ("flag_head" in name or "cam_" in name):
                out[f"actor.{name}"] = float(p.abs().max().item())
        for name, p in self.q1.named_parameters():
            if name.startswith("head.") and name.endswith(".weight"):
                out[f"q1.{name}"] = float(p.abs().max().item())
        return out

    def save(self, path: str):
        torch.save({
            "actor":         self.actor.state_dict(),
            "q1":            self.q1.state_dict(),
            "q2":            self.q2.state_dict(),
            "q1_targ":       self.q1_targ.state_dict(),
            "q2_targ":       self.q2_targ.state_dict(),
            "log_alpha":     self.log_alpha.detach().cpu(),
            "step":          self._step,
            "img_channels":  self.img_channels,
            "n_flags":       self.n_flags,
            "camera_max":    self.camera_max_rad,
            "kind":          "hybrid_sac",
        }, path)

    def load(self, path: str, only_actor: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        if only_actor:
            return
        if "q1" in ckpt:
            self.q1.load_state_dict(ckpt["q1"])
            self.q2.load_state_dict(ckpt["q2"])
            self.q1_targ.load_state_dict(ckpt["q1_targ"])
            self.q2_targ.load_state_dict(ckpt["q2_targ"])
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
            self._step = ckpt.get("step", 0)
