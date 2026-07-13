"""
Entorno Gymnasium para el agente RL de Minecraft (woodcutting).

Arquitectura de comunicación:
    Python (Gymnasium) <──HTTP──> Node.js rl_agent.js <──Mineflayer──> Minecraft

Espacio de observaciones (use_visual=False, por defecto):
    Box(-1, 1, (STATE_DIM,))     ← vector de estado normalizado (8 componentes)

Espacio de observaciones (use_visual=True):
    Dict {
        "state": Box(-1, 1, (STATE_DIM,))
        "image": Box(0, 255, (3 * img_frame_stack, IMG_SIZE, IMG_SIZE))
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
    REWARD_WRONG_BLOCK,
    REWARD_SUCCESS, LOGS_TO_SUCCESS,
    MAX_STEPS, CUMULATIVE_REWARD_THRESHOLD,
    HYBRID_FLAGS, N_HYBRID_FLAGS, CAMERA_MAX_RAD,
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
        bridge_port:     int = RL_BRIDGE_PORT,
        use_visual:      bool = False,
        frame_stack:     int  = 1,
        img_frame_stack: int  = 1,
        render_mode:     str | None = None,
        max_steps:       int  = MAX_STEPS,
        logs_to_success: int  = LOGS_TO_SUCCESS,
        hybrid:          bool = False,
        cumulative_reward_threshold: float | None = CUMULATIVE_REWARD_THRESHOLD,
    ):
        super().__init__()
        self.bridge_url      = f"http://localhost:{bridge_port}"
        self.use_visual      = use_visual
        self.frame_stack     = max(1, frame_stack)
        self.img_frame_stack = max(1, img_frame_stack)
        self.render_mode     = render_mode
        self._max_steps      = max_steps
        self._logs_to_success = max(1, int(logs_to_success))
        self.hybrid          = hybrid
        self._cumulative_reward_threshold = cumulative_reward_threshold

        # ── Espacio de observaciones ──────────────────────────────────────────
        obs_dim     = STATE_DIM * self.frame_stack
        state_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        if use_visual:
            img_channels = 3 * self.img_frame_stack
            self.observation_space = spaces.Dict({
                "state": state_space,
                "image": spaces.Box(0, 255, shape=(img_channels, IMG_SIZE, IMG_SIZE), dtype=np.uint8),
            })
        else:
            self.observation_space = state_space

        # ── Espacio de acciones ──────────────────────────────────────────────
        if self.hybrid:
            # Hybrid SAC: vector concatenado [flags binarios (0/1) | dyaw | dpitch (rad)].
            # Bounds asimétricos: [0,1] para flags + [-CAMERA_MAX_RAD, +CAMERA_MAX_RAD]
            # para cámara. Usamos un solo Box con bounds vectorizados.
            low  = np.array([0.0]*N_HYBRID_FLAGS + [-CAMERA_MAX_RAD, -CAMERA_MAX_RAD],
                            dtype=np.float32)
            high = np.array([1.0]*N_HYBRID_FLAGS + [+CAMERA_MAX_RAD, +CAMERA_MAX_RAD],
                            dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(ACTIONS))

        # ── Estado interno del episodio ───────────────────────────────────────
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_log_count: int = 0
        self._last_render_frame: np.ndarray | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._img_frame_buffer: list[np.ndarray] = []
        # Shaping: is_looking_at_log
        self._last_is_looking_at_log: float = 0.0
        # Shaping: distancia al árbol (para reward de acercamiento)
        self._prev_tree_distance: float = 0.0
        self._last_tree_distance: float = 0.0
        self._prev_tree_visible:  float = 0.0
        self._last_tree_visible:  float = 0.0
        # Recompensa por recoger troncos: delta de inventario entre steps
        self._log_count_delta: int = 0
        # Troncos recogidos acumulados en el episodio (condición de éxito)
        self._logs_collected_total: int = 0
        # Troncos rotos por el agente en el episodio (gate del éxito)
        self._logs_broken_total: int = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._prev_pos          = None
        self._prev_log_count    = 0
        self._frame_buffer      = []
        self._img_frame_buffer  = []
        self._last_is_looking_at_log = 0.0
        self._prev_tree_distance = 0.0
        self._last_tree_distance = 0.0
        self._prev_tree_visible  = 0.0
        self._last_tree_visible  = 0.0
        self._log_count_delta    = 0
        self._logs_collected_total = 0
        self._logs_broken_total  = 0

        resp = self._post("/reset", {})
        obs  = self._parse_obs(resp)
        return obs, {}

    def step(self, action):
        if self.hybrid:
            payload, action_name = self._build_hybrid_payload(action)
        else:
            action_name = ACTIONS[int(action)]
            payload     = {"action": action_name}

        resp   = self._post("/step", payload)
        obs    = self._parse_obs(resp)
        events = resp.get("events", {})
        reward = self._compute_reward(events)

        self._step_count           += 1
        self._logs_collected_total += self._log_count_delta
        # Troncos rotos por el agente en este step (gate para success real)
        logs_broken_this_step = sum(
            1 for b in events.get("blocks_broken", []) if b in LOG_TYPES
        )
        self._logs_broken_total += logs_broken_this_step
        success                     = False

        # ── Condiciones de terminación ────────────────────────────────────────
        terminated = False
        truncated  = False

        if events.get("is_dead", False):
            terminated = True
            reward    += REWARD_DONE_PENALTY

        # Éxito real: recoger ≥ LOGS_TO_SUCCESS *Y* haber roto al menos un tronco
        # en el episodio. El gate evita el patrón degenerado en el que el agente
        # spawnea sobre drops residuales del reset previo y "cosecha" sin atacar
        # — eso inflaba la success rate y disparaba Q-values de estados triviales.
        elif (
            self._logs_collected_total >= self._logs_to_success
            and self._logs_broken_total >= 1
        ):
            terminated = True
            reward    += REWARD_SUCCESS
            success    = True

        elif (self._cumulative_reward_threshold is not None
              and self._cumulative_reward + reward < self._cumulative_reward_threshold):
            terminated = True
            reward    += REWARD_DONE_PENALTY

        if self._step_count >= self._max_steps:
            truncated = True

        self._cumulative_reward += reward

        info = {
            "step":              self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "action_name":       action_name,
            "blocks_broken":     events.get("blocks_broken", []),
            "is_attacking_tree": events.get("is_attacking_tree", False),
            "attacked_block":    events.get("attacked_block", None),
            "log_broken":         self._prev_log_count,
            "logs_collected":    self._log_count_delta,   # troncos recogidos este step
            "logs_collected_total": self._logs_collected_total,
            "logs_broken_total":  self._logs_broken_total,
            "success":           success,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self._last_render_frame is not None:
            return self._last_render_frame
        return None

    def close(self):
        pass

    # ── Hybrid action helpers ─────────────────────────────────────────────────

    def _build_hybrid_payload(self, action) -> tuple[dict, str]:
        """
        Convierte un action_vec numpy (n_flags + 2,) en el payload HTTP del bridge.

        El último par son (dyaw, dpitch) en radianes — el agente Mineflayer los
        suma a la orientación actual del bot tal cual. Devuelve también un
        nombre legible para info["action_name"] (logging/metrics).
        """
        a = np.asarray(action, dtype=np.float32).flatten()
        flags_arr = a[:N_HYBRID_FLAGS]
        dyaw      = float(a[N_HYBRID_FLAGS + 0])
        dpitch    = float(a[N_HYBRID_FLAGS + 1])

        flags = {name: bool(flags_arr[i] > 0.5) for i, name in enumerate(HYBRID_FLAGS)}
        payload = {
            "flags":  flags,
            "camera": {"dyaw": dyaw, "dpitch": dpitch},
        }
        active = [n for n, v in flags.items() if v]
        cam_str = f"yaw={dyaw:+.2f} pitch={dpitch:+.2f}"
        action_name = ("+".join(active) or "noop") + f"|{cam_str}"
        return payload, action_name

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, events: dict) -> float:
        reward = REWARD_STEP  # penalty base por step

        for block in events.get("blocks_broken", []):
            if block in LOG_TYPES:
                reward += REWARD_BREAK_LOG

        if events.get("is_attacking_tree", False):
            reward += REWARD_HIT_TREE
        else:
            # Atacar algo que no sea tronco: penalización del mismo orden
            # que REWARD_STEP para desincentivar spam de `attack` sin colapsar
            # la exploración.
            attacked = events.get("attacked_block")
            if attacked and attacked not in LOG_TYPES:
                reward += REWARD_WRONG_BLOCK

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

        # Frame stacking visual: concatenamos los últimos k frames en el eje de canales.
        # El DQN vanilla no puede inferir movimiento desde un único frame; stack de k=4
        # es el truco estándar de Atari (Mnih 2015) para expresar velocidades.
        self._img_frame_buffer.append(img_arr)
        if len(self._img_frame_buffer) > self.img_frame_stack:
            self._img_frame_buffer.pop(0)
        pad = self.img_frame_stack - len(self._img_frame_buffer)
        stacked_img = np.concatenate(
            [self._img_frame_buffer[0]] * pad + self._img_frame_buffer,
            axis=0,
        )

        return {"state": state_vec, "image": stacked_img}

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        url = self.bridge_url + path
        for attempt in range(retries):
            try:
                # 30s de margen: un step puede acumular dig (hasta 8s) + captura de
                # screenshot (hasta 8s) + esperas de física; 15s se quedaba corto.
                r = requests.post(url, json=payload, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Bridge request a {url} falló tras {retries} intentos: {exc}"
                    ) from exc
                time.sleep(1.0)
