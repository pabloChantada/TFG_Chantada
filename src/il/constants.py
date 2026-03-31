"""
Constantes compartidas del pipeline de Imitation Learning.

Importar desde aquí evita tener que actualizar el mismo valor en varios archivos.
Uso: from constants import ACTIONS, BATCH_SIZE
"""

# Espacio de acciones discretas (una acción por step)
ACTIONS = [
    "move_forward_walk",
    "move_forward_sprint",
    "move_backward_walk",
    "move_left",
    "move_right",
    "jump",
    "sneak",
    "camera_yaw_p15",
    "camera_yaw_m15",
    "camera_yaw_p45",
    "camera_yaw_m45",
    "camera_pitch_p15",
    "camera_pitch_m15",
    "camera_pitch_p45",
    "camera_pitch_m45",
    "attack",
    "equip_wooden_axe",
]

# Longitud de la ventana temporal (nº de frames por secuencia)
SEQ_LEN = 4

# Dimensión del vector de estado del bot (x, y, z, yaw, pitch)
STATE_DIM = 5

# Tamaño de batch por defecto para DataLoader
BATCH_SIZE = 32

# Normalización ImageNet (requerida por torchvision ResNet preentrenado)
# Ref: https://discuss.pytorch.org/t/147540
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
