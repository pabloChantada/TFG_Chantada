import os
import sys
import json
import math
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

from constants import (BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
                        SEQ_LEN, STATE_DIM, STATE_KEYS, STATE_BOUNDS,
                        CAMERA_DIM, CAMERA_BOUNDS)

# Pares de labels que son imagen espejo entre sí (flip horizontal los intercambia)
MIRROR_PAIRS = [
    ("move_left", "move_right"),
]


def normalize_action_label(action):
    """Devuelve el string label si es válido, o None para descartar el sample."""
    if isinstance(action, str) and action:
        return action
    return None


def _norm(val, key):
    """Normaliza un escalar a [-1, 1] usando bounds fijos, con clamp."""
    lo, hi = STATE_BOUNDS[key]
    v = max(lo, min(hi, float(val)))
    return 2.0 * (v - lo) / (hi - lo) - 1.0 if hi != lo else 0.0


class MinecraftDataset(Dataset):
    def __init__(self, jsonl_path, pair2id=None, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.data    = []
        self.transforms = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
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

                    # Raw state: 7 valores que se almacenan tal cual.
                    # dx/dz se calculan en __getitem__ a partir de frames consecutivos.
                    state_dict = item.get("state") or {}
                    tv = item.get("tree_visible")
                    td = item.get("tree_distance")
                    state_raw = torch.tensor([
                        float(state_dict.get("x",     0.0)),
                        float(state_dict.get("y",     0.0)),
                        float(state_dict.get("z",     0.0)),
                        float(state_dict.get("yaw",   0.0)),
                        float(state_dict.get("pitch", 0.0)),
                        float(tv) if tv is not None else 0.0,
                        float(td) if td is not None else float(STATE_BOUNDS["tree_distance"][1]),
                    ], dtype=torch.float32)

                    # Camera delta continuo (dyaw, dpitch en radianes)
                    cd = item.get("camera_delta") or {}
                    raw_dyaw   = float(cd.get("dyaw",   0.0))
                    raw_dpitch = float(cd.get("dpitch", 0.0))
                    # Fix yaw wrapping: normalizar a [-π, π]
                    while raw_dyaw >  math.pi: raw_dyaw -= 2 * math.pi
                    while raw_dyaw < -math.pi: raw_dyaw += 2 * math.pi
                    camera_raw = torch.tensor([raw_dyaw, raw_dpitch], dtype=torch.float32)

                    session_id = item.get("session_id", "__unknown__")

                    self.data.append({
                        "image_path":  img_path,
                        "label":       label_id,
                        "state_raw":   state_raw,    # [x,y,z,yaw,pitch,tv,td]
                        "camera_raw":  camera_raw,   # [dyaw,dpitch]
                        "session_id":  session_id,
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

    def filter_min_samples(self, min_samples):
        """Elimina clases con menos de min_samples muestras y recalcula pair2id."""
        counts = Counter(d["label"] for d in self.data)
        id2name = {v: k for k, v in self.pair2id.items()}
        drop_ids = {lid for lid, n in counts.items() if n < min_samples}
        drop_names = [id2name[lid] for lid in sorted(drop_ids)]

        if not drop_ids:
            return

        before = len(self.data)
        self.data = [d for d in self.data if d["label"] not in drop_ids]

        # Recalcular pair2id solo con clases supervivientes
        kept = {k: v for k, v in self.pair2id.items() if v not in drop_ids}
        self.pair2id = {name: i for i, name in enumerate(sorted(kept.keys()))}

        # Reasignar label ids en data
        name_for_old_id = id2name
        for d in self.data:
            name = name_for_old_id[d["label"]]
            d["label"] = self.pair2id[name]

        self._rebuild_sequence_index()
        print(f"  Filtro min_samples={min_samples}: eliminadas {drop_names}, "
              f"{before} → {len(self.data)} muestras, {len(self.pair2id)} clases")

    def _rebuild_sequence_index(self):
        """Reconstruye el índice de sesiones y posiciones tras filtrar self.data."""
        self._session_frames: dict[str, list[int]] = {}
        for i, item in enumerate(self.data):
            sid = item["session_id"]
            self._session_frames.setdefault(sid, []).append(i)

        self._sample_pos = [None] * len(self.data)
        for sid, indices in self._session_frames.items():
            for pos, global_idx in enumerate(indices):
                self._sample_pos[global_idx] = (sid, pos)

    def __len__(self):
        return len(self.data)

    def _build_state_vector(self, raw_seq):
        """Construye el state vector normalizado de 9 dims a partir de raws de la secuencia.

        raw_seq: list of tensors [x, y, z, yaw, pitch, tree_visible, tree_distance]
        Returns: (T, STATE_DIM) tensor normalizado
        """
        T_len = len(raw_seq)
        states = torch.zeros(T_len, STATE_DIM)

        for t in range(T_len):
            r = raw_seq[t]
            x, y, z, yaw, pitch, tv, td = r[0], r[1], r[2], r[3], r[4], r[5], r[6]

            # dx, dz respecto al frame anterior
            if t > 0:
                dx = x - raw_seq[t - 1][0]
                dz = z - raw_seq[t - 1][2]
            else:
                dx = dz = 0.0

            vals = [x, y, z, yaw, pitch, dx, dz, tv, td]
            for j, (v, k) in enumerate(zip(vals, STATE_KEYS)):
                states[t, j] = _norm(v, k)

        return states

    def _normalize_camera(self, camera_raw):
        """Normaliza camera_delta a [-1, 1] usando CAMERA_BOUNDS."""
        dyaw_lo, dyaw_hi = CAMERA_BOUNDS["dyaw"]
        dp_lo, dp_hi     = CAMERA_BOUNDS["dpitch"]
        dyaw   = max(dyaw_lo, min(dyaw_hi, camera_raw[0].item()))
        dpitch = max(dp_lo,   min(dp_hi,   camera_raw[1].item()))
        norm_dyaw   = 2.0 * (dyaw   - dyaw_lo) / (dyaw_hi - dyaw_lo) - 1.0
        norm_dpitch = 2.0 * (dpitch - dp_lo)    / (dp_hi   - dp_lo)   - 1.0
        return torch.tensor([norm_dyaw, norm_dpitch], dtype=torch.float32)

    def __getitem__(self, idx):
        sid, pos = self._sample_pos[idx]
        session_indices = self._session_frames[sid]

        # Obtener los T índices de la secuencia (padding al inicio si es necesario)
        start     = max(0, pos - self.seq_len + 1)
        positions = list(range(start, pos + 1))
        while len(positions) < self.seq_len:
            positions = [positions[0]] + positions
        global_seq = [session_indices[p] for p in positions]

        imgs_list, raw_states, labels_list, camera_list = [], [], [], []
        for g_idx in global_seq:
            item  = self.data[g_idx]
            image = Image.open(item["image_path"]).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
            imgs_list.append(image)
            raw_states.append(item["state_raw"])
            labels_list.append(item["label"])
            camera_list.append(self._normalize_camera(item["camera_raw"]))

        imgs   = torch.stack(imgs_list)                                    # (T, C, H, W)
        states = self._build_state_vector(raw_states)                      # (T, STATE_DIM=9)
        labels = torch.tensor(labels_list, dtype=torch.long)               # (T,)
        camera = torch.stack(camera_list)                                  # (T, 2)

        # Flip horizontal consistente en toda la secuencia (solo en train)
        if self.use_mirror_flip and random.random() < 0.5:
            imgs   = torch.flip(imgs, dims=[3])
            labels = torch.tensor(
                [self.mirror_label_map.get(l.item(), l.item()) for l in labels],
                dtype=torch.long,
            )
            # Invertir dyaw al hacer flip horizontal (dpitch no cambia)
            camera[:, 0] = -camera[:, 0]

        return imgs, states, labels, camera

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
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train_ds.use_mirror_flip = False
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
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        return train_loader, val_loader