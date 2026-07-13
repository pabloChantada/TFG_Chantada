"""
Agente PPO (Proximal Policy Optimization) — variante discreta con clipping.

Reutiliza el `MinecraftRLEnv` y la representación visual de DQN. Diferencias
clave frente a DQN:

- On-policy: cada actualización usa solo trayectorias recientes (rollout buffer
  que se vacía tras cada update). No hay replay buffer ni target network.
- Política estocástica entrenada por gradient ascent sobre el ratio recortado.
- Critic V(s) entrenado por MSE contra los retornos calculados con GAE.

Uso típico (rollout largo, varias épocas de SGD):
    rollout_steps = 2048
    epochs        = 4
    minibatch     = 64
"""

import sys
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RL_DIR / "shared"))
for _sub in ("visual/algorithms", "visual/models", "visual/train", "visual/eval"):
    sys.path.insert(0, str(_RL_DIR / _sub))

import numpy as np
import torch
import torch.nn.functional as F

from constants import STATE_DIM, ACTIONS, IMG_SIZE
from model_ac import ActorCritic

N_ACTIONS = len(ACTIONS)


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


class RolloutBuffer:
    """
    Buffer on-policy para PPO. Almacena una trayectoria contigua de longitud T
    (puede abarcar varios episodios concatenados) y al cerrar calcula
    advantages con GAE-λ y retornos.

    No es circular: se rellena, se computan ventajas, se itera sobre minibatches
    aleatorios para los updates, y se vacía con `clear()`.
    """

    def __init__(self, capacity: int, img_shape: tuple, state_dim: int,
                 device: torch.device, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.capacity   = capacity
        self.device     = device
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

        self.imgs      = np.zeros((capacity,) + img_shape, dtype=np.float32)
        self.states    = np.zeros((capacity, state_dim),   dtype=np.float32)
        self.actions   = np.zeros(capacity,                dtype=np.int64)
        self.log_probs = np.zeros(capacity,                dtype=np.float32)
        self.values    = np.zeros(capacity,                dtype=np.float32)
        self.rewards   = np.zeros(capacity,                dtype=np.float32)
        self.dones     = np.zeros(capacity,                dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns    = np.zeros(capacity, dtype=np.float32)
        self.idx = 0

    def add(self, img, state, action, log_prob, value, reward, done):
        i = self.idx
        self.imgs[i]      = img
        self.states[i]    = state
        self.actions[i]   = int(action)
        self.log_probs[i] = float(log_prob)
        self.values[i]    = float(value)
        self.rewards[i]   = float(reward)
        self.dones[i]     = float(done)
        self.idx += 1

    def full(self) -> bool:
        return self.idx >= self.capacity

    def clear(self):
        self.idx = 0

    def compute_gae(self, last_value: float):
        """GAE-λ. Cierra el rollout con un bootstrap del crítico en el último estado."""
        T = self.idx
        adv = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_v * next_nonterminal - self.values[t]
            adv   = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[:T] = self.advantages[:T] + self.values[:T]

    def iterate_minibatches(self, batch_size: int):
        T = self.idx
        idx = np.random.permutation(T)
        for start in range(0, T, batch_size):
            sl = idx[start:start + batch_size]
            yield {
                "imgs":      _to_tensor(self.imgs[sl],      self.device),
                "states":    _to_tensor(self.states[sl],    self.device),
                "actions":   _to_tensor(self.actions[sl],   self.device, dtype=torch.long),
                "log_probs": _to_tensor(self.log_probs[sl], self.device),
                "advantages":_to_tensor(self.advantages[sl],self.device),
                "returns":   _to_tensor(self.returns[sl],   self.device),
                "values":    _to_tensor(self.values[sl],    self.device),
            }


class PPOAgent:
    """
    PPO con clip-objective + GAE + entropy bonus + value-function clipping.

    Hiperparámetros estándar (Schulman et al. 2017, "PPO baselines"):
        clip_eps    = 0.2
        epochs      = 4
        gae_lambda  = 0.95
        ent_coef    = 0.01
        vf_coef     = 0.5
        max_grad    = 0.5
    """

    def __init__(
        self,
        feat_dim:    int   = 256,
        hidden:      int   = 256,
        use_state:   bool  = True,
        img_channels:int   = 12,
        lr:          float = 3e-4,
        gamma:       float = 0.99,
        gae_lambda:  float = 0.95,
        clip_eps:    float = 0.2,
        ent_coef:    float = 0.01,
        vf_coef:     float = 0.5,
        max_grad:    float = 0.5,
        epochs:      int   = 4,
        minibatch:   int   = 64,
        rollout_steps:int  = 2048,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(
            feat_dim=feat_dim, hidden=hidden, use_state=use_state,
            in_channels=img_channels,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        self.use_state    = use_state
        self.img_channels = img_channels
        self.clip_eps     = clip_eps
        self.ent_coef     = ent_coef
        self.vf_coef      = vf_coef
        self.max_grad     = max_grad
        self.epochs       = epochs
        self.minibatch    = minibatch
        self.rollout_steps= rollout_steps

        self.buffer = RolloutBuffer(
            capacity=rollout_steps,
            img_shape=(img_channels, IMG_SIZE, IMG_SIZE),
            state_dim=STATE_DIM,
            device=self.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self._step = 0

    # ── Interacción con el entorno ────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False):
        img   = self._normalize_img(obs["image"])
        state = obs["state"]
        img_t   = _to_tensor(img,   self.device).unsqueeze(0)
        state_t = _to_tensor(state, self.device).unsqueeze(0) if self.use_state else None
        action, log_prob, value, _ = self.net.act(img_t, state_t, deterministic=deterministic)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs: dict) -> float:
        img   = self._normalize_img(obs["image"])
        img_t   = _to_tensor(img, self.device).unsqueeze(0)
        state_t = _to_tensor(obs["state"], self.device).unsqueeze(0) if self.use_state else None
        _, value = self.net.forward(img_t, state_t)
        return float(value.item())

    def store(self, obs, action, log_prob, value, reward, done):
        img = self._normalize_img(obs["image"])
        self.buffer.add(img, obs["state"], action, log_prob, value, reward, done)

    @staticmethod
    def _normalize_img(img):
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, last_obs: dict, last_done: bool) -> dict:
        """Calcula GAE y ejecuta `epochs` épocas sobre minibatches del rollout."""
        last_value = 0.0 if last_done else self.value(last_obs)
        self.buffer.compute_gae(last_value)

        # Normalizar advantages (estándar en PPO).
        T = self.buffer.idx
        adv = self.buffer.advantages[:T]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages[:T] = adv

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl":   0.0, "clip_frac":  0.0, "n_updates": 0}

        for _ in range(self.epochs):
            for batch in self.buffer.iterate_minibatches(self.minibatch):
                states = batch["states"] if self.use_state else None
                new_log_probs, new_values, entropy = self.net.evaluate(
                    batch["imgs"], batch["actions"], states
                )

                ratio = torch.exp(new_log_probs - batch["log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value-function clipping (PPO2): limita el cambio del crítico.
                v_clipped  = batch["values"] + torch.clamp(
                    new_values - batch["values"], -self.clip_eps, self.clip_eps
                )
                vf1 = (new_values - batch["returns"]).pow(2)
                vf2 = (v_clipped  - batch["returns"]).pow(2)
                value_loss = 0.5 * torch.max(vf1, vf2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - batch["log_probs"]
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += entropy.mean().item()
                stats["approx_kl"]   += approx_kl
                stats["clip_frac"]   += clip_frac
                stats["n_updates"]   += 1

        n = max(1, stats["n_updates"])
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[k] /= n

        self.buffer.clear()
        self._step += T
        return stats

    @torch.no_grad()
    def weight_max_abs(self) -> dict:
        out = {}
        for name, p in self.net.named_parameters():
            if (".actor." in name or ".critic." in name) and name.endswith(".weight"):
                out[name] = float(p.abs().max().item())
        return out

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "net":          self.net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "step":         self._step,
            "img_channels": self.img_channels,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
