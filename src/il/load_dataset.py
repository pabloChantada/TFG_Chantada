import os
import sys
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import numpy as np
import copy
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTIONS = [
    "idle",
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

BATCH_SIZE = 32

# Definimos una etiqueta especial para idle
IDLE_ACTION = "idle"


def vector_to_action_label(action):
    """
    action: lista o array de 11 ints (legacy multidiscrete)
    Devuelve una etiqueta discreta única.
    """
    try:
        move_forward = int(action[0])
        move_backward = int(action[1])
        move_lateral = int(action[2])
        move_vertical = int(action[3])
        camera_yaw = int(action[4])
        camera_pitch = int(action[5])
        attack = int(action[6])
        equip = int(action[10])
    except Exception:
        return IDLE_ACTION

    # Prioridad igual que en el tracker
    if attack == 1:
        return "attack"
    if equip == 4:
        return "equip_wooden_axe"

    if camera_yaw == 1:
        return "camera_yaw_p15"
    if camera_yaw == 2:
        return "camera_yaw_m15"
    if camera_yaw == 3:
        return "camera_yaw_p45"
    if camera_yaw == 4:
        return "camera_yaw_m45"

    if camera_pitch == 1:
        return "camera_pitch_p15"
    if camera_pitch == 2:
        return "camera_pitch_m15"
    if camera_pitch == 3:
        return "camera_pitch_p45"
    if camera_pitch == 4:
        return "camera_pitch_m45"

    if move_forward == 2:
        return "move_forward_sprint"
    if move_forward == 1:
        return "move_forward_walk"
    if move_backward == 1:
        return "move_backward_walk"
    if move_lateral == 1:
        return "move_left"
    if move_lateral == 2:
        return "move_right"
    if move_vertical == 1:
        return "jump"
    if move_vertical == 2:
        return "sneak"

    return IDLE_ACTION


def normalize_action_label(action):
    """Acepta formato discreto nuevo o vector legacy."""
    if isinstance(action, str):
        return action if action else IDLE_ACTION

    if isinstance(action, list) and len(action) == 11:
        return vector_to_action_label(action)

    if isinstance(action, dict):
        # Permite formatos tipo {"name":"attack"}
        name = action.get("name") or action.get("action") or action.get("type")
        if isinstance(name, str) and name:
            return name
        # o dict multidiscrete legado
        return vector_to_action_label([
            int(action.get("move_forward", 0) or 0),
            int(action.get("move_backward", 0) or 0),
            int(action.get("move_lateral", 0) or 0),
            int(action.get("move_vertical", 0) or 0),
            int(action.get("camera_yaw", 0) or 0),
            int(action.get("camera_pitch", 0) or 0),
            int(action.get("attack", 0) or 0),
            int(action.get("craft", 0) or 0),
            int(action.get("smelt", 0) or 0),
            int(action.get("place", 0) or 0),
            int(action.get("equip", 0) or 0),
        ])

    return IDLE_ACTION


class MinecraftDataset(Dataset):
    def __init__(self, jsonl_path, pair2id=None):
        self.data = []
        self.transforms = T.Compose([
            T.Resize((224, 224)),   # Resize para usar en ResNet
            T.ToTensor(),
            # Normalizacion para ResNet preentrenado con Imagenet (https://discuss.pytorch.org/t/what-is-the-correct-pytorch-resnet50-input-normalization-intensity-range/147540)
            # Es un poco magic numbers pero son las valores que se recomiendan
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

        # pair2id: dict externo opcional (para tener el mismo vocab en train/val/test)
        # si no se pasa, lo construiremos sobre la marcha
        self.pair2id = {} if pair2id is None else dict(pair2id)

        print(f"Loading dataset from {jsonl_path}.")
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    img_path = item.get("image")
                    # Saltamos la linea si no tiene imagen o la ruta no existe
                    # En principio nunca deberia pasar, pero por si acaso
                    if not img_path or not os.path.exists(img_path):
                        continue

                    action = item.get("action")
                    if action is None:
                        continue

                    action_label = normalize_action_label(action)

                    # asignar id si no existe
                    if action_label not in self.pair2id:
                        self.pair2id[action_label] = len(self.pair2id)
                    label_id = self.pair2id[action_label]

                    self.data.append({
                        "image_path": img_path,
                        "label": label_id,
                    })
                except Exception:
                    continue

        print(f"Dataset: {len(self.data)} valid samples")

        # distribución por etiqueta discreta
        if self.data:
            labels = [d["label"] for d in self.data]
            print("Label distribution:", Counter(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        label = item["label"]  
        return image, label

    def get_transforms(self):
        return self.transforms   



    # Reemplaza tu método load_dataset actual por este:
    def load_dataset(self, split_ratio=0.8):
        train_size = int(split_ratio * len(self.data))
        
        # Mezclar índices manualmente
        # indices = torch.randperm(len(self)).tolist()
        # train_indices = indices[:train_size]
        # val_indices = indices[train_size:]

        # split secuencial cronológico, el dataset esta ordanado por tiempo
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(self.data)))
        
        # Crear Dataset de Entrenamiento
        train_ds = copy.deepcopy(self)
        train_ds.data = [self.data[i] for i in train_indices]
        train_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Variación visual
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        
        # Crear Dataset de Validación (SIN ColorJitter)
        val_ds = copy.deepcopy(self)
        val_ds.data = [self.data[i] for i in val_indices]
        val_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        return train_loader, val_loader
