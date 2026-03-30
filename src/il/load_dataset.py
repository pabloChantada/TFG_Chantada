import os
import sys
import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import IDLE_ACTION, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Label pairs that are mirror images of each other (horizontal flip swaps them)
MIRROR_PAIRS = [
    ("camera_yaw_p45", "camera_yaw_m45"),
    ("camera_yaw_p15", "camera_yaw_m15"),
]


def normalize_action_label(action):
    """Normaliza una acción al string label discreto correspondiente."""
    if isinstance(action, str):
        return action if action else IDLE_ACTION
    return IDLE_ACTION


class MinecraftDataset(Dataset):
    def __init__(self, jsonl_path, pair2id=None):
        self.data = []
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self.pair2id = {} if pair2id is None else dict(pair2id)
        self.use_mirror_flip = False
        self.mirror_label_map: dict[int, int] = {}
        self.num_aux = 0  # set after loading

        print(f"Loading dataset from {jsonl_path}.")
        error_count = 0
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    img_path = item.get("image")
                    if not img_path or not os.path.exists(img_path):
                        continue

                    action = item.get("action")
                    if action is None:
                        continue

                    action_label = normalize_action_label(action)

                    if action_label not in self.pair2id:
                        self.pair2id[action_label] = len(self.pair2id)
                    label_id = self.pair2id[action_label]

                    # Tree-visibility aux features
                    tree_visible = item.get("tree_visible")
                    tree_distance = item.get("tree_distance")
                    has_tree_aux = tree_visible is not None or tree_distance is not None

                    self.data.append({
                        "image_path": img_path,
                        "label": label_id,
                        "tree_visible": bool(tree_visible) if tree_visible is not None else None,
                        "tree_distance": float(tree_distance) if tree_distance is not None else None,
                        "has_tree_aux": has_tree_aux,
                    })
                except Exception as e:
                    error_count += 1
                    print(f"  [WARN] Línea {i}: {type(e).__name__}: {e}")
                    continue

        if error_count > 0:
            print(f"  [{error_count} líneas con error omitidas]")
        print(f"Dataset: {len(self.data)} valid samples")

        if self.data:
            labels = [d["label"] for d in self.data]
            print("Label distribution:", Counter(labels))

        # Determine num_aux from whether tree aux fields are present
        aux_samples = [d for d in self.data if d["has_tree_aux"]]
        self.num_aux = 2 if aux_samples else 0
        if self.num_aux:
            pct_visible = 100.0 * sum(1 for d in self.data if d["tree_visible"]) / len(self.data)
            dist_vals = [d["tree_distance"] / 32.0 for d in self.data if d["tree_distance"] is not None]
            mean_dist = sum(dist_vals) / len(dist_vals) if dist_vals else 0.0
            print(f"Aux feat : 2  (tree_visible, tree_distance_norm)")
            print(f"  tree_visible=True : {pct_visible:.1f}%")
            print(f"  tree_distance mean: {mean_dist:.2f} (norm)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        label = item["label"]

        if self.transforms:
            image = self.transforms(image)

        # Build aux tensor: [tree_visible (0/1), tree_distance_norm (clipped 0-1)]
        if item["has_tree_aux"]:
            tv = 1.0 if item["tree_visible"] else 0.0
            td = min(max(item["tree_distance"] / 32.0, 0.0), 1.0) if item["tree_distance"] is not None else 0.0
            aux = torch.tensor([tv, td], dtype=torch.float32)
        else:
            aux = torch.zeros(2, dtype=torch.float32)

        # Label-aware mirror flip (train only, enabled via use_mirror_flip)
        # Flips the image horizontally and swaps camera_yaw_p <-> camera_yaw_m labels.
        if self.use_mirror_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            label = self.mirror_label_map.get(label, label)

        return image, aux, label

    def get_transforms(self):
        return self.transforms

    def load_dataset(self, split_ratio=0.8, seed=42):
        # --- Session-based split ---
        # Group sample indices by recording session (parent directory of the image).
        # Sessions are shuffled with a fixed seed and split 80/20 so that no session
        # bleeds across train/val, avoiding temporal leakage and world-state bias.
        session_to_indices: dict[str, list[int]] = {}
        for i, item in enumerate(self.data):
            parts = item["image_path"].replace("\\", "/").split("/")
            key = parts[-2] if len(parts) >= 2 else None
            if key is None:
                print(f"  [WARN] Cannot parse session from '{item['image_path']}', assigning to train")
                key = "__unresolved__"
            session_to_indices.setdefault(key, []).append(i)

        sessions = sorted(session_to_indices.keys())
        rng = random.Random(seed)
        rng.shuffle(sessions)

        n_train = max(1, int(len(sessions) * split_ratio))
        train_sessions = set(sessions[:n_train])
        val_sessions   = set(sessions[n_train:])

        print(f"\n  Session split (seed={seed}): {len(train_sessions)} train / {len(val_sessions)} val")
        print(f"  Train sessions: {sorted(train_sessions)}")
        print(f"  Val   sessions: {sorted(val_sessions)}")

        train_indices = [i for s in train_sessions for i in session_to_indices[s]]
        val_indices   = [i for s in val_sessions   for i in session_to_indices[s]]

        train_ds = copy.deepcopy(self)
        train_ds.data = [self.data[i] for i in train_indices]
        train_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # T.RandomGrayscale(p=0.1),
            # NOTE: No RandomHorizontalFlip — plain flipping corrupts camera_yaw labels.
            # Label-aware mirror flip is handled in __getitem__ via use_mirror_flip.
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
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

        val_ds = copy.deepcopy(self)
        val_ds.data = [self.data[i] for i in val_indices]
        val_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        return train_loader, val_loader
