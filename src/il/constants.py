"""
Constantes compartidas del pipeline de Imitation Learning.

Importar desde aquí evita tener que actualizar el mismo valor en varios archivos.
Uso: from constants import ACTIONS, BATCH_SIZE
"""

import math

# Espacio de acciones discretas (una acción por step).
# La cámara se predice como valores continuos (dyaw, dpitch) en una cabeza
# de regresión separada, no como acciones discretas.
ACTIONS = [
    "move_forward_jump",
    "move_forward_sprint",
    "move_backward_walk",
    "move_left",
    "move_right",
    "jump",
    "sneak",
    "attack",
    "equip_wooden_axe",
]

# Dimensión de la salida continua de cámara (dyaw, dpitch en radianes)
CAMERA_DIM = 2

# Bounds para normalización del camera_delta a [-1, 1].
# Rango estrecho (~±57°): los deltas reales son mayoritariamente < 0.3 rad;
# outliers > 1 rad son artefactos del pathfinding (giros bruscos del HTN)
# que no queremos que el modelo reproduzca.
CAMERA_BOUNDS = {
    "dyaw":   (-1.0, 1.0),
    "dpitch": (-1.0, 1.0),
}

# Longitud de la ventana temporal (nº de frames por secuencia)
SEQ_LEN = 4

# ── State vector ──────────────────────────────────────────────────────────────
# Orden canónico del vector de estado (9 componentes).
# dx/dz se calculan en load_dataset a partir de frames consecutivos.
STATE_KEYS = ["x", "y", "z", "yaw", "pitch", "dx", "dz",
              "tree_visible", "tree_distance"]

STATE_DIM = len(STATE_KEYS)  # 9

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
    "tree_visible":   (0,        1),
    "tree_distance":  (0,      100),
}

# Tamaño de batch por defecto para DataLoader
BATCH_SIZE = 32

# Resolución de entrada al modelo visual
IMG_SIZE = 128

# Normalización estándar (media y std de ImageNet, válida como baseline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]