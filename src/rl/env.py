"""
Entorno Gymnasium para el agente RL de Minecraft (woodcutting).

Arquitectura de comunicación:
    Python (Gymnasium) <──HTTP──> Node.js rl_agent.js <──Mineflayer──> Minecraft

Espacio de observaciones (use_visual=False, por defecto):
    Box(-1, 1, (STATE_DIM,))     ← vector de estado normalizado (8 componentes)

Espacio de observaciones (use_visual=True):
    Dict {
        "state": Box(-1, 1, (STATE_DIM,))
        "image": Box(0, 255, (3, IMG_SIZE, IMG_SIZE))
    }

Espacio de acciones:
    Discrete(7)   ← índice de: attack | move_forward_sprint | move_forward_jump
                               | camera_right | camera_left | camera_up | camera_down
"""

import io
import sys
import os
import base64
import time

sys.path.insert(0, os.path.dirname(__file__))  # src/rl/ primero → pilla rl/constants.py

import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from constants import (
    ACTIONS,
    STATE_DIM, STATE_BOUNDS,
    IMG_SIZE,
    RL_BRIDGE_PORT,
    REWARD_BREAK_LOG, REWARD_HIT_TREE, REWARD_LOOK_AT_LOG,
    REWARD_COLLECT_LOG,
    REWARD_APPROACH, REWARD_STEP, REWARD_DONE_PENALTY,
    MAX_STEPS, CUMULATIVE_REWARD_THRESHOLD,
)

LOG_TYPES = frozenset([
    'oak_log', 'birch_log', 'spruce_log',
    'dark_oak_log', 'jungle_log', 'acacia_log',
])


def _norm(val: float, key: str) -> float:
    """Normaliza un valor escalar a [-1, 1] usando los bounds de STATE_BOUNDS."""
    lo, hi = STATE_BOUNDS[key]
    return float(2.0 * (val - lo) / (hi - lo) - 1.0)


class MinecraftRLEnv(gym.Env):
    """
    Entorno Gymnasium que se comunica con rl_agent.js vía HTTP.

    El agente Node.js expone:
        POST /step  {action: str}
                    → {state: {...}, events: {...}}
        POST /reset {}
                    → {state: {...}}

    Parámetros
    ----------
    bridge_port : int
        Puerto del servidor HTTP de rl_agent.js (por defecto RL_BRIDGE_PORT).
    use_visual : bool
        Si True, la observación incluye un frame RGB además del vector de estado.
    render_mode : str | None
        Solo soportado: "rgb_array".
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        bridge_port: int = RL_BRIDGE_PORT,
        use_visual:  bool = False,
        frame_stack: int  = 1,
        render_mode: str | None = None,
        max_steps:   int  = MAX_STEPS,
    ):
        super().__init__()
        self.bridge_url  = f"http://localhost:{bridge_port}"
        self.use_visual  = use_visual
        self.frame_stack = max(1, frame_stack)
        self.render_mode = render_mode
        self._max_steps  = max_steps

        # ── Espacio de observaciones ──────────────────────────────────────────
        obs_dim     = STATE_DIM * self.frame_stack
        state_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        if use_visual:
            self.observation_space = spaces.Dict({
                "state": state_space,
                "image": spaces.Box(0, 255, shape=(3, IMG_SIZE, IMG_SIZE), dtype=np.uint8),
            })
        else:
            self.observation_space = state_space

        # ── Espacio de acciones: discreto puro ───────────────────────────────
        self.action_space = spaces.Discrete(len(ACTIONS))

        # ── Estado interno del episodio ───────────────────────────────────────
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_log_count: int = 0
        self._last_render_frame: np.ndarray | None = None
        self._frame_buffer: list[np.ndarray] = []
        # Shaping: is_looking_at_log
        self._last_is_looking_at_log: float = 0.0
        # Shaping: distancia al árbol (para reward de acercamiento)
        self._prev_tree_distance: float = 0.0
        self._last_tree_distance: float = 0.0
        self._prev_tree_visible:  float = 0.0
        self._last_tree_visible:  float = 0.0
        # Recompensa por recoger troncos: delta de inventario entre steps
        self._log_count_delta: int = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._prev_pos          = None
        self._prev_log_count    = 0
        self._frame_buffer      = []
        self._last_is_looking_at_log = 0.0
        self._prev_tree_distance = 0.0
        self._last_tree_distance = 0.0
        self._prev_tree_visible  = 0.0
        self._last_tree_visible  = 0.0
        self._log_count_delta    = 0

        resp = self._post("/reset", {})
        obs  = self._parse_obs(resp)
        return obs, {}

    def step(self, action: int):
        action_name = ACTIONS[int(action)]

        resp   = self._post("/step", {"action": action_name})
        obs    = self._parse_obs(resp)
        events = resp.get("events", {})
        reward = self._compute_reward(events)

        self._step_count        += 1
        self._cumulative_reward += reward

        # ── Condiciones de terminación ────────────────────────────────────────
        terminated = False
        truncated  = False

        if events.get("is_dead", False):
            terminated = True
            reward    += REWARD_DONE_PENALTY

        elif self._cumulative_reward < CUMULATIVE_REWARD_THRESHOLD:
            terminated = True
            reward    += REWARD_DONE_PENALTY

        if self._step_count >= self._max_steps:
            truncated = True

        info = {
            "step":              self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "action_name":       action_name,
            "blocks_broken":     events.get("blocks_broken", []),
            "is_attacking_tree": events.get("is_attacking_tree", False),
            "attacked_block":    events.get("attacked_block", None),
            "log_broken":         self._prev_log_count,
            "logs_collected":    self._log_count_delta,   # troncos recogidos este step
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self._last_render_frame is not None:
            return self._last_render_frame
        return None

    def close(self):
        pass

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, events: dict) -> float:
        reward = REWARD_STEP  # penalty base por step

        for block in events.get("blocks_broken", []):
            if block in LOG_TYPES:
                reward += REWARD_BREAK_LOG

        if events.get("is_attacking_tree", False):
            reward += REWARD_HIT_TREE

        if self._log_count_delta > 0:
            reward += self._log_count_delta * REWARD_COLLECT_LOG

        # Shaping: cursor apuntando a un log
        if self._last_is_looking_at_log > 0.5:
            reward += REWARD_LOOK_AT_LOG

        # Shaping: acercarse al árbol visible
        # Requiere árbol visible en ambos steps Y limita el delta a 2 bloques/step
        # para evitar señal falsa cuando cambia el árbol detectado por el FOV cone
        if self._prev_tree_visible > 0.5 and self._last_tree_visible > 0.5:
            delta = self._prev_tree_distance - self._last_tree_distance
            delta = max(-2.0, min(2.0, delta))  # clamp: máximo 2 bloques por step
            reward += delta * REWARD_APPROACH

        return reward

    # ── Observación ───────────────────────────────────────────────────────────

    def _parse_obs(self, resp: dict):
        raw = resp.get("state", {})

        x    = float(raw.get("x",                  0.0))
        z    = float(raw.get("z",                  0.0))
        yaw  = float(raw.get("yaw",                0.0))
        pitch= float(raw.get("pitch",              0.0))
        tv   = float(raw.get("tree_visible",       0.0))
        _td_raw = raw.get("tree_distance")
        td   = float(_td_raw) if (_td_raw is not None and tv > 0.5) else STATE_BOUNDS["tree_distance"][1]
        lc   = float(raw.get("log_count",          0.0))
        ill  = float(raw.get("is_looking_at_log",  0.0))

        # dx/dz entre steps consecutivos (x/z absolutos solo para el delta)
        pos = np.array([x, z], dtype=np.float64)
        if self._prev_pos is None:
            dx, dz = 0.0, 0.0
        else:
            delta = pos - self._prev_pos
            lo, hi = STATE_BOUNDS["dx"]
            dx = float(np.clip(delta[0], lo, hi))
            dz = float(np.clip(delta[1], lo, hi))
        self._prev_pos = pos

        # delta de troncos en inventario → usado en _compute_reward
        self._log_count_delta = max(0, int(lc) - self._prev_log_count)
        self._prev_log_count  = int(lc)

        # Shaping: cursor sobre log
        self._last_is_looking_at_log = ill

        # Shaping: distancia al árbol (rotamos prev → last para _compute_reward)
        self._prev_tree_distance = self._last_tree_distance
        self._last_tree_distance = td
        self._prev_tree_visible  = self._last_tree_visible
        self._last_tree_visible  = tv

        state_vec = np.array([
            _norm(yaw,   "yaw"),
            _norm(pitch, "pitch"),
            _norm(dx,    "dx"),
            _norm(dz,    "dz"),
            _norm(tv,    "tree_visible"),
            _norm(td,    "tree_distance"),
            _norm(lc,    "log_count"),
            _norm(ill,   "is_looking_at_log"),
        ], dtype=np.float32)

        # Frame stacking: mantener buffer de los últimos k frames
        self._frame_buffer.append(state_vec)
        if len(self._frame_buffer) > self.frame_stack:
            self._frame_buffer.pop(0)
        # Padding al inicio del episodio: repetir el primer frame
        pad = self.frame_stack - len(self._frame_buffer)
        stacked = np.concatenate(
            [self._frame_buffer[0]] * pad + self._frame_buffer
        )

        if not self.use_visual:
            return stacked

        img_b64 = resp.get("screenshot", "")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
            img_arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)  # HWC→CHW
        else:
            img_arr = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        self._last_render_frame = img_arr.transpose(1, 2, 0)
        return {"state": state_vec, "image": img_arr}

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        url = self.bridge_url + path
        for attempt in range(retries):
            try:
                r = requests.post(url, json=payload, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Bridge request a {url} falló tras {retries} intentos: {exc}"
                    ) from exc
                time.sleep(1.0)
