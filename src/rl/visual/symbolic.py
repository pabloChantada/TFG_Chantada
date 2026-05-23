"""
Política simbólica reactiva — Minecraft woodcutting.

Heurística basada en reglas sobre el vector de estado del entorno
(`STATE_KEYS = [yaw, pitch, dx, dz, tree_visible, tree_distance, log_count,
is_looking_at_log]`). Sin red neuronal ni aprendizaje: cada step se mira el
estado y se aplica la primera regla que matchea.

Ideado como "tutor" para guiar la exploración de SAC en el modo DAgger
(`train_sac_dagger.py`). También usable como baseline aislado.

Reglas (en orden de prioridad):
    1. is_looking_at_log == 1                       → attack
    2. tree_visible == 1  ∧  tree_distance > 3      → avanzar (sprint / jump)
    3. tree_visible == 1  ∧  tree_distance ≤ 3      → barrer cámara (cerca,
                                                        falta solo alinear cursor)
    4. tree_visible == 0  ∧  |pitch| > 0.25 rad     → centrar mirada al horizonte
    5. tree_visible == 0                            → rotar yaw + avanzar a ratos
"""

import sys
import math
import random
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))

import numpy as np

from constants import ACTIONS, STATE_KEYS, STATE_BOUNDS

# Índices de ACTIONS, calculados una vez al importar
_AIDX = {a: i for i, a in enumerate(ACTIONS)}

# Índices en el vector de estado normalizado
_SIDX = {k: i for i, k in enumerate(STATE_KEYS)}

# Umbrales de la heurística (en unidades del mundo, ya denormalizadas)
CLOSE_DISTANCE  = 3.0    # bloques — debajo: cámara fina, encima: avanzar
LEVEL_PITCH_RAD = 0.25   # rad ~14° — más allá: recentrar al horizonte

# Pesos de la rotación cuando no hay árbol: el bot tiende a moverse para
# evitar quedarse mirando estático. Suma 1.0.
SCAN_WEIGHTS = {
    "camera_left":         0.30,
    "camera_right":        0.30,
    "move_forward_sprint": 0.30,
    "move_forward_jump":   0.10,
}


def _denorm(val_norm: float, key: str) -> float:
    """Invierte la normalización a [-1, 1] aplicada por env.MinecraftRLEnv."""
    lo, hi = STATE_BOUNDS[key]
    return float((val_norm + 1.0) / 2.0 * (hi - lo) + lo)


def _is_true(val_norm: float) -> bool:
    """
    Bool a partir de un valor normalizado de un campo {0, 1}.

    STATE_BOUNDS para tree_visible / is_looking_at_log es (0, 1) → el valor
    normalizado es exactamente -1 o +1. Tomamos el centro como umbral seguro.
    """
    return val_norm > 0.0


class SymbolicPolicy:
    """
    Política reactiva sobre el último frame del vector de estado normalizado.

    Solo lee `obs["state"]` (las últimas STATE_DIM componentes si hay
    frame-stacking). El resto del observation dict se ignora.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, obs) -> tuple[int, str]:
        """
        Devuelve (idx_acción, nombre_regla) dado un obs dict del MinecraftRLEnv.

        El segundo valor identifica qué regla disparó la decisión — útil para
        debug y para anotar el dataset DAgger.
        """
        state = obs["state"] if isinstance(obs, dict) else obs
        state = np.asarray(state, dtype=np.float32).flatten()

        # Si hay frame stacking, las STATE_DIM últimas son el frame actual
        from constants import STATE_DIM
        cur = state[-STATE_DIM:]

        tree_vis     = _is_true(cur[_SIDX["tree_visible"]])
        looking_log  = _is_true(cur[_SIDX["is_looking_at_log"]])
        tree_dist    = _denorm(float(cur[_SIDX["tree_distance"]]), "tree_distance")
        pitch_rad    = _denorm(float(cur[_SIDX["pitch"]]),         "pitch")

        # 1. Cursor sobre un log → atacar
        if looking_log:
            return _AIDX["attack"], "attack_log"

        # 2. Árbol visible y lejos → avanzar
        if tree_vis and tree_dist > CLOSE_DISTANCE:
            action = ("move_forward_jump"
                      if self._rng.random() < 0.2 else "move_forward_sprint")
            return _AIDX[action], "approach_tree"

        # 3. Árbol visible y cerca → barrer cámara para alinear cursor
        if tree_vis:
            # Pequeño bias hacia arriba: los troncos suelen estar a altura ojo
            # y los muestreos uniformes sobre 4 direcciones desperdician
            # mitad de los steps mirando al suelo / al cielo.
            action = self._rng.choices(
                ["camera_left", "camera_right", "camera_up", "camera_down"],
                weights=[0.30, 0.30, 0.25, 0.15],
                k=1,
            )[0]
            return _AIDX[action], "align_cursor"

        # 4. Sin árbol y mirando demasiado arriba/abajo → recentrar
        if pitch_rad > LEVEL_PITCH_RAD:
            return _AIDX["camera_up"],   "recenter_pitch"
        if pitch_rad < -LEVEL_PITCH_RAD:
            return _AIDX["camera_down"], "recenter_pitch"

        # 5. Sin árbol y mirada nivelada → escanear (rotar / avanzar)
        names   = list(SCAN_WEIGHTS.keys())
        weights = [SCAN_WEIGHTS[n] for n in names]
        action  = self._rng.choices(names, weights=weights, k=1)[0]
        return _AIDX[action], "scan"
