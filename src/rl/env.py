"""
Gymnasium environment for Minecraft RL via the JS bot server.

Connects to the Node.js server (src/rl/server.js) which wraps a Mineflayer bot.
Uses the same MultiDiscrete action space as the metrics tracker.

Usage:
    import gymnasium as gym
    from env import MinecraftWoodEnv

    env = MinecraftWoodEnv(server_url="http://localhost:3001")
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

import json
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action space definition — matches rl_action_tracker.js exactly
# MultiDiscrete([3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5])
ACTION_DIMS = np.array([3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5], dtype=np.int64)

ACTION_LABELS = [
    "move_forward",   # 0=still, 1=walk, 2=sprint
    "move_backward",  # 0=still, 1=walk
    "move_lateral",   # 0=still, 1=left, 2=right
    "move_vertical",  # 0=still, 1=jump, 2=sneak
    "camera_yaw",     # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    "camera_pitch",   # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    "attack",         # 0=no, 1=yes
    "craft",          # 0=none..6=iron_pickaxe
    "smelt",          # 0=none, 1=iron_ingot
    "place",          # 0=none..3=torch
    "equip",          # 0=none..4=axe
]

# Observation: 13-dim vector from state.js
OBS_DIM = 13


class MinecraftWoodEnv(gym.Env):
    """
    Gymnasium wrapper for the Minecraft wood-chopping RL task.

    The environment communicates via HTTP with a Node.js server
    that runs a Mineflayer bot inside Minecraft.
    """

    metadata = {"render_modes": []}

    def __init__(self, server_url="http://localhost:3001", timeout=30):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.timeout_penalty = -1.0

        # Spaces
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._step_count = 0
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        data = self._post("/reset")
        obs = np.array(data["observation"], dtype=np.float32)
        info = data.get("info", {})
        self._step_count = 0
        self._last_obs = obs
        return obs, info

    def step(self, action):
        action_list = [int(a) for a in action]
        try:
            data = self._post("/step", {"action": action_list})
        except requests.exceptions.RequestException as exc:
            # Prevent training crash on transient server stalls/timeouts.
            # We truncate current episode and keep training moving.
            info = {
                "error": str(exc),
                "step_timeout": True,
                "step": self._step_count,
            }

            try:
                obs, state_info = self.get_state()
                info.update({"recovered_state": True, "state_info": state_info})
            except requests.exceptions.RequestException:
                obs = self._last_obs.copy()
                info["recovered_state"] = False

            self._step_count += 1
            self._last_obs = obs
            return obs, float(self.timeout_penalty), False, True, info

        obs = np.array(data["observation"], dtype=np.float32)
        reward = float(data["reward"])
        terminated = bool(data["terminated"])
        truncated = bool(data["truncated"])
        info = data.get("info", {})

        self._step_count += 1
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._post("/close")
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_state(self):
        """Get current observation without stepping."""
        data = self._get("/state")
        return np.array(data["observation"], dtype=np.float32), data.get("info", {})

    def get_server_info(self):
        """Get server metadata (action/obs space, config)."""
        return self._get("/info")

    def _post(self, path, body=None):
        url = f"{self.server_url}{path}"
        resp = requests.post(url, json=body or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path):
        url = f"{self.server_url}{path}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Simplified sub-env for wood-only task (fewer action dims)
# ──────────────────────────────────────────────────────────────────────────────

# For wood chopping we only need: movement(4 dims) + camera(2 dims) + attack(1 dim) = 7 dims
WOOD_ACTION_DIMS = np.array([3, 2, 3, 3, 5, 5, 2], dtype=np.int64)


class MinecraftWoodSimpleEnv(MinecraftWoodEnv):
    """
    Simplified env for wood-only task.
    Only exposes movement + camera + attack (7 dimensions).
    Craft/smelt/place/equip are always 0.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.MultiDiscrete(WOOD_ACTION_DIMS)

    def step(self, action):
        # Pad the 7-dim action to full 11-dim with zeros for craft/smelt/place/equip
        full_action = np.zeros(len(ACTION_DIMS), dtype=np.int64)
        full_action[: len(action)] = action
        return super().step(full_action)


# ──────────────────────────────────────────────────────────────────────────────
# Register with Gymnasium
# ──────────────────────────────────────────────────────────────────────────────

gym.register(
    id="MinecraftWood-v0",
    entry_point="env:MinecraftWoodEnv",
)

gym.register(
    id="MinecraftWoodSimple-v0",
    entry_point="env:MinecraftWoodSimpleEnv",
)
