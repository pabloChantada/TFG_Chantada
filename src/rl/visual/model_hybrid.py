"""
Arquitectura Hybrid SAC — espacio de acciones tipo MineRL.

Acción híbrida = (flags binarios concurrentes  +  cámara continua 2D):
    flags  : Bernoulli factorizada sobre HYBRID_FLAGS (e.g. [forward, jump, sprint, attack])
    camera : (dyaw, dpitch) en radianes, bounded a [-CAMERA_MAX_RAD, +CAMERA_MAX_RAD]
             vía tanh squashing.

HybridActor
    Salidas:  (flag_logits, cam_mu, cam_log_std)
              flag_logits  : (N, N_FLAGS)
              cam_mu       : (N, 2)
              cam_log_std  : (N, 2)  (clipeado a [-5, 2])
    sample(): muestrea acción reparametrizada y devuelve también log_prob conjunto.

HybridQNet
    Q(s, a) escalar.  Toma (img, state, action_vec) y proyecta la acción al
    espacio del trunk para fusionarla por suma con las features de imagen+estado.
"""

import sys
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import STATE_DIM, IMG_SIZE, HYBRID_FLAGS, N_HYBRID_FLAGS, CAMERA_MAX_RAD
from model_ac import _SharedTrunk

CAMERA_DIM         = 2
ACTION_VEC_DIM     = N_HYBRID_FLAGS + CAMERA_DIM   # flags binarios + (dyaw, dpitch)
LOG_STD_MIN        = -5.0
LOG_STD_MAX        =  2.0


class HybridActor(nn.Module):
    """
    Política factorizada Bernoulli×Gaussiana sobre el trunk visual+estado.

    Bernoulli factorizada para los flags (asumimos independencia condicional
    dado el estado — práctica estándar en hybrid SAC).
    Gaussiana diagonal squashed con tanh para la cámara.
    """

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_flags: int = N_HYBRID_FLAGS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True,
                 in_channels: int = 3,
                 camera_max_rad: float = CAMERA_MAX_RAD):
        super().__init__()
        self.n_flags        = n_flags
        self.camera_max_rad = camera_max_rad

        self.trunk = _SharedTrunk(feat_dim, state_dim, img_size, use_state, in_channels)

        # Tronco MLP compartido entre cabezas
        self.shared = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
        )

        # Cabezas
        self.flag_head    = nn.Linear(hidden, n_flags)
        self.cam_mu_head  = nn.Linear(hidden, CAMERA_DIM)
        self.cam_lstd_head = nn.Linear(hidden, CAMERA_DIM)

    def _features(self, imgs, states):
        return self.shared(self.trunk(imgs, states))

    def forward(self, imgs, states=None):
        """
        Returns:
            flag_logits : (N, n_flags)
            cam_mu      : (N, 2)
            cam_log_std : (N, 2)  clipeado
        """
        h = self._features(imgs, states)
        flag_logits = self.flag_head(h)
        cam_mu      = self.cam_mu_head(h)
        cam_log_std = self.cam_lstd_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return flag_logits, cam_mu, cam_log_std

    def sample(self, imgs, states=None, deterministic: bool = False):
        """
        Muestreo reparametrizado.

        Returns:
            action_vec : (N, ACTION_VEC_DIM) — flags en {0,1} (rama no diferenciable)
                         + cámara en [-camera_max, +camera_max] (rama diferenciable)
            log_prob   : (N,) log π(a|s) sumado sobre flags + cámara
            extra      : dict con (flag_probs, flags_sampled, cam_pre_tanh) — útil
                         para BC (soft labels) y diagnóstico.
        """
        flag_logits, cam_mu, cam_log_std = self.forward(imgs, states)

        # ── Flags: Bernoulli factorizada ─────────────────────────────────────
        flag_probs = torch.sigmoid(flag_logits)
        if deterministic:
            flags = (flag_probs > 0.5).float()
        else:
            flags = torch.bernoulli(flag_probs)
        # log_prob bajo Bernoulli: stable via BCE pero con signo positivo de log p(a)
        # log p(a=1)=log σ(z), log p(a=0)=log(1-σ(z))=−z+log σ(z)
        log_prob_flags = -F.binary_cross_entropy_with_logits(
            flag_logits, flags, reduction="none"
        ).sum(dim=-1)  # (N,)

        # ── Cámara: Gaussiana squashed ───────────────────────────────────────
        cam_std  = cam_log_std.exp()
        if deterministic:
            cam_pre = cam_mu
        else:
            eps     = torch.randn_like(cam_mu)
            cam_pre = cam_mu + cam_std * eps           # reparametrizado
        cam_tanh = torch.tanh(cam_pre)
        camera   = cam_tanh * self.camera_max_rad      # bounded a [-max, +max]

        # log π_gauss(pre) − log|d tanh / d pre|, ajustado por la escala (ver SAC original)
        # log|d(camera)/d(pre)| = log(camera_max) + log(1 - tanh²(pre))
        log_prob_pre = (-0.5 * ((cam_pre - cam_mu) / (cam_std + 1e-8)).pow(2)
                        - cam_log_std
                        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi)))
        # corrección de cambio de variable
        log_prob_cam = log_prob_pre - torch.log(1 - cam_tanh.pow(2) + 1e-6) \
                       - torch.log(torch.tensor(self.camera_max_rad))
        log_prob_cam = log_prob_cam.sum(dim=-1)        # (N,)

        log_prob   = log_prob_flags + log_prob_cam
        action_vec = torch.cat([flags, camera], dim=-1)

        extra = {
            "flag_probs":   flag_probs,
            "flags":        flags,
            "camera":       camera,
            "cam_mu":       cam_mu,
            "cam_log_std":  cam_log_std,
        }
        return action_vec, log_prob, extra

    @torch.no_grad()
    def act(self, imgs, states=None, deterministic: bool = False):
        """Inferencia: muestra acción y devuelve también flag_probs / cámara para logging."""
        action_vec, log_prob, extra = self.sample(imgs, states, deterministic=deterministic)
        return action_vec, log_prob, extra


class HybridQNet(nn.Module):
    """
    Q(s, a) → escalar.

    Fusión: trunk(s) + action_proj(a)  → MLP → 1.
    Esquema "feature addition" alineado con cómo el trunk fusiona estado e imagen.
    """

    def __init__(self, feat_dim: int = 256, state_dim: int = STATE_DIM,
                 n_flags: int = N_HYBRID_FLAGS, hidden: int = 256,
                 img_size: int = IMG_SIZE, use_state: bool = True,
                 in_channels: int = 3):
        super().__init__()
        self.trunk        = _SharedTrunk(feat_dim, state_dim, img_size, use_state, in_channels)
        self.action_proj  = nn.Linear(n_flags + CAMERA_DIM, feat_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, imgs, action_vec, states=None):
        """
        imgs       : (N, C, H, W)
        action_vec : (N, n_flags + 2)  — flags ∈ {0,1}, cámara en radianes
        states     : (N, state_dim) o None
        → Q        : (N,)
        """
        feat = self.trunk(imgs, states) + self.action_proj(action_vec)
        return self.head(feat).squeeze(-1)
