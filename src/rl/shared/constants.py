"""
Constantes del pipeline de Reinforcement Learning.

Reutiliza definiciones de IL (ACTIONS, STATE_KEYS, bounds, etc.) para mantener
coherencia entre ambos pipelines.
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'il'))

# Posición absoluta (x, y, z) excluida: no generaliza entre spawns.
# dx/dz incluidos: permiten detectar si el bot está atascado.
STATE_KEYS = ["yaw", "pitch", "dx", "dz", "tree_visible", "tree_distance", "log_count", "is_looking_at_log"]

STATE_DIM = len(STATE_KEYS)  # 6

# Bounds fijos para normalización a [-1, 1].
# Preferimos bounds conocidos del entorno (no data-driven) para que la
# normalización sea estable entre rebuilds del dataset.
STATE_BOUNDS = {
    "x":              (-200,    200),
    "y":              (0,       256),
    "z":              (-200,    200),
    "yaw":            (-math.pi, math.pi),      # mineflayer → radianes
    "pitch":          (-math.pi/2, math.pi/2),   # mineflayer → radianes
    "dx":             (-1,       1),
    "dz":             (-1,       1),
    "tree_visible":      (0,        1),
    "tree_distance":     (0,       40),   # bioma forestal: siempre < 30 bloques
    "log_count":         (0,       64),   # máximo 1 stack en inventario
    "is_looking_at_log": (0,        1),
}

# Resolución de entrada al modelo visual
IMG_SIZE = 128

# Normalización estándar (media y std de ImageNet, válida como baseline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Acciones discretas del agente RL.
# La cámara es discreta (no hay cabeza de regresión continua como en IL).
ACTIONS = [
    "attack",
    "move_forward_sprint",
    "move_forward_jump",
    "camera_right",
    "camera_left",
    "camera_up",      
    "camera_down",    
]

CAMERA_TURN_RAD  = 0.15   # ~8.6° — más fino que 11.5°
CAMERA_PITCH_RAD = 0.10   # vertical más pequeño

REWARD_BREAK_LOG   =  20.0   # romper un tronco: objetivo principal
REWARD_COLLECT_LOG =  10.0   # recoger el tronco del suelo (señal principal)
REWARD_HIT_TREE    =   0.5   # golpear un tronco (señal auxiliar, valor bajo)
REWARD_LOOK_AT_LOG =   0.0   # cursor apuntando a un log (señal de alineación)
REWARD_APPROACH    =   0.0  # acercarse al árbol visible
REWARD_STEP        =  -0.01  # aumentado: -0.001 era demasiado pequeño para guiar
REWARD_WRONG_BLOCK =  -0.01  # penalización por atacar un bloque que no es tronco
REWARD_DONE_PENALTY = -5.0   # penalización al terminar el episodio anticipadamente
REWARD_SUCCESS     =  30.0   # bono al cumplir la condición de éxito (1 tronco recogido)
LOGS_TO_SUCCESS    =   1     # nº de troncos recogidos que termina el episodio con éxito
MAX_STEPS          =  300    # más horizonte
CUMULATIVE_REWARD_THRESHOLD = -20.0  # terminar si reward acumulado cae por debajo

# ── Bridge HTTP (Node.js ↔ Python) ────────────────────────────────────────────
RL_BRIDGE_PORT = 8766          # puerto del rl_agent.js

# ── Posición de reset ─────────────────────────────────────────────────────────
# Coordenadas del spawn de entrenamiento (bioma boscoso).
# Ajustar según el mapa cargado en el servidor.
SPAWN_X = 0
SPAWN_Y = 64
SPAWN_Z = 0
