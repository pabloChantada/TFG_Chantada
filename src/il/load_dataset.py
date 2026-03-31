import os
import sys
import json
import random
import copy
from collections import Counter
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEQ_LEN, STATE_DIM

# Pares de labels que son imagen espejo entre sí (flip horizontal los intercambia)
MIRROR_PAIRS = [
    ("camera_yaw_p45", "camera_yaw_m45"),
    ("camera_yaw_p15", "camera_yaw_m15"),
]


def normalize_action_label(action):
    """Devuelve el string label si es válido, o None para descartar el sample."""
    if isinstance(action, str) and action:
        return action
    return None


class MinecraftDataset(Dataset):
    def __init__(self, jsonl_path, pair2id=None, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.data    = []
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.pair2id = {} if pair2id is None else dict(pair2id)
        self.use_mirror_flip   = False
        self.mirror_label_map: dict[int, int] = {}

        print(f"Loading dataset from {jsonl_path}.")
        error_count = 0
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    img_path = item.get("image")
                    if not img_path or not os.path.exists(img_path):
                        continue

                    action_label = normalize_action_label(item.get("action"))
                    if action_label is None:
                        continue

                    if action_label not in self.pair2id:
                        self.pair2id[action_label] = len(self.pair2id)
                    label_id = self.pair2id[action_label]

                    # Vector de estado (x, y, z, yaw, pitch) — raw, sin normalizar
                    state_dict = item.get("state") or {}
                    state_vec  = torch.tensor([
                        float(state_dict.get("x",     0.0)),
                        float(state_dict.get("y",     0.0)),
                        float(state_dict.get("z",     0.0)),
                        float(state_dict.get("yaw",   0.0)),
                        float(state_dict.get("pitch", 0.0)),
                    ], dtype=torch.float32)

                    session_id = item.get("session_id", "__unknown__")

                    self.data.append({
                        "image_path": img_path,
                        "label":      label_id,
                        "state_vec":  state_vec,
                        "session_id": session_id,
                    })
                except Exception as e:
                    error_count += 1
                    print(f"  [WARN] Línea {i}: {type(e).__name__}: {e}")
                    continue

        if error_count:
            print(f"  [{error_count} líneas con error omitidas]")
        print(f"Dataset: {len(self.data)} valid samples")

        if self.data:
            labels = [d["label"] for d in self.data]
            print("Label distribution:", Counter(labels))

        self._rebuild_sequence_index()

    def _rebuild_sequence_index(self):
        """Reconstruye el índice de sesiones y posiciones tras filtrar self.data."""
        # session_id → lista de índices globales en orden temporal
        self._session_frames: dict[str, list[int]] = {}
        for i, item in enumerate(self.data):
            sid = item["session_id"]
            self._session_frames.setdefault(sid, []).append(i)

        # Para cada sample: (session_id, posición dentro de la sesión)
        self._sample_pos = [None] * len(self.data)
        for sid, indices in self._session_frames.items():
            for pos, global_idx in enumerate(indices):
                self._sample_pos[global_idx] = (sid, pos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid, pos = self._sample_pos[idx]
        session_indices = self._session_frames[sid]

        # Obtener los T índices de la secuencia (padding al inicio si es necesario)
        start     = max(0, pos - self.seq_len + 1)
        positions = list(range(start, pos + 1))
        while len(positions) < self.seq_len:
            positions = [positions[0]] + positions
        global_seq = [session_indices[p] for p in positions]

        imgs_list, states_list, labels_list = [], [], []
        for g_idx in global_seq:
            item  = self.data[g_idx]
            image = Image.open(item["image_path"]).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            imgs_list.append(image)
            states_list.append(item["state_vec"])
            labels_list.append(item["label"])

        imgs   = torch.stack(imgs_list)                                    # (T, C, H, W)
        states = torch.stack(states_list)                                  # (T, STATE_DIM)
        labels = torch.tensor(labels_list, dtype=torch.long)               # (T,)

        # Flip horizontal consistente en toda la secuencia (solo en train)
        if self.use_mirror_flip and random.random() < 0.5:
            imgs   = torch.flip(imgs, dims=[3])
            labels = torch.tensor(
                [self.mirror_label_map.get(l.item(), l.item()) for l in labels],
                dtype=torch.long,
            )

        return imgs, states, labels

    def load_dataset(self, split_ratio=0.8, seed=42):
        """Split por sesión (80/20) para evitar leakage temporal."""
        sessions = sorted(self._session_frames.keys())
        rng = random.Random(seed)
        rng.shuffle(sessions)

        n_train       = max(1, int(len(sessions) * split_ratio))
        train_sessions = set(sessions[:n_train])
        val_sessions   = set(sessions[n_train:])

        print(f"\n  Session split (seed={seed}): {len(train_sessions)} train / {len(val_sessions)} val")
        print(f"  Train sessions: {sorted(train_sessions)}")
        print(f"  Val   sessions: {sorted(val_sessions)}")

        train_indices = [i for s in train_sessions for i in self._session_frames[s]]
        val_indices   = [i for s in val_sessions   for i in self._session_frames[s]]

        # Train dataset con augmentación
        train_ds      = copy.deepcopy(self)
        train_ds.data = [self.data[i] for i in train_indices]
        train_ds._rebuild_sequence_index()
        train_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_ds.use_mirror_flip = True
        mirror_label_map: dict[int, int] = {}
        for action_a, action_b in MIRROR_PAIRS:
            id_a = self.pair2id.get(action_a)
            id_b = self.pair2id.get(action_b)
            if id_a is not None and id_b is not None:
                mirror_label_map[id_a] = id_b
                mirror_label_map[id_b] = id_a
        train_ds.mirror_label_map = mirror_label_map

        # Val dataset sin augmentación
        val_ds      = copy.deepcopy(self)
        val_ds.data = [self.data[i] for i in val_indices]
        val_ds._rebuild_sequence_index()
        val_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        return train_loader, val_loader
