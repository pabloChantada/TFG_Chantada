import os
import sys
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import IDLE_ACTION, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD


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

                    self.data.append({
                        "image_path": img_path,
                        "label": label_id,
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, item["label"]

    def get_transforms(self):
        return self.transforms

    def load_dataset(self, split_ratio=0.8):
        train_size = int(split_ratio * len(self.data))

        # Split cronológico: 80% inicial para train, 20% final para val.
        # No aleatorio para respetar la temporalidad de los episodios.
        train_indices = list(range(train_size))
        val_indices   = list(range(train_size, len(self.data)))

        train_ds = copy.deepcopy(self)
        train_ds.data = [self.data[i] for i in train_indices]
        train_ds.transforms = T.Compose([
            T.Resize((224, 224)),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

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
