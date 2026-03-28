import os
import sys
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTIONS = {
    0: "move_forward",  # 0=still, 1=walk, 2=sprint
    1: "move_backward", # 0=still, 1=walk
    2: "move_lateral",  # 0=still, 1=left, 2=right
    3: "move_vertical", # 0=still, 1=jump, 2=sneak
    4: "camera_yaw",    # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    5: "camera_pitch",  # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    6: "attack",        # 0=no, 1=yes
    7: "craft",         # 0=none,1=planks,2=stick,3=crafting_table,4=wpick,5=spick,6=ipick
    8: "smelt",         # 0=none, 1=iron_ingot
    9: "place",         # 0=none, 1=crafting_table, 2=furnace, 3=torch
    10: "equip"         # 0=none, 1=wpick, 2=spick, 3=ipick, 4=axe
}

BATCH_SIZE = 32

# Definimos un par especial para idle (todo ceros)
IDLE_PAIR = ("idle", 0)

# TODO: cambiar lo de >1 cuando hagamos que la camra se mueva en un solo eje
def vector_to_pair(action):
    """
    action: lista o array de 11 ints
    Devuelve:
      - ("idle", 0) si todo es 0
      - (slot, valor) si hay una posición != 0
    """
    # i -> accion a tomar, v -> valor de esa accion. i.e: (0,2) -> move_forward=2 (sprint), (4,3) -> camera_yaw=+45°
    nonzero = [(i, v) for i, v in enumerate(action) if v != 0]
    if len(nonzero) == 0:
        return IDLE_PAIR
    else:
        # si hubiera >1, nos quedamos con la primera por simplicidad
        return tuple(nonzero[0])  # (slot, valor)


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
                    # lista de 11 ints
                    if action is None or len(action) != 11:
                        continue

                    # vector (11,) -> par ("idle",0) o (slot, valor)
                    pair = vector_to_pair(action)

                    # asignar id si no existe
                    if pair not in self.pair2id:
                        self.pair2id[pair] = len(self.pair2id)
                    label_id = self.pair2id[pair]

                    self.data.append({
                        "image_path": img_path,
                        "label": label_id,
                    })
                except Exception:
                    continue

        print(f"Dataset: {len(self.data)} valid samples")

        # distribución por par (slot, valor) o idle
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

    def load_dataset(self, split_ratio=0.8):
        # No aplicamos una separacion de test, ya que la hacemos manualmente o con el bot
        train_size = int(split_ratio * len(self.data))
        val_size = len(self.data) - train_size
        # Las transformaciones se aplican en __getitem__
        train_ds, val_ds = random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        return train_loader, val_loader
    

if __name__ == "__main__":
    dataset = MinecraftDataset(jsonl_path='data\\train.jsonl')
    train_loader, val_loader = dataset.load_dataset()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("Sample batch:")

    for images, labels in train_loader:
        print("Images shape:", images.shape)  # [B,3,224,224]
        print("Labels shape:", labels.shape)  # [B]
        print("Labels:", labels)
        print(f"Image: {images[0]}, Label: {labels[0]}")
        print("Label id to (slot, value) mapping:")
        print()
        id2pair = {v: k for k, v in dataset.pair2id.items()}

        print("ID -> (action_name, value) -- At least 1 time:")
        for id_, (slot, value) in sorted(id2pair.items()):
            # como no tenemos el nombre de la accion para idle, lo ponemos a mano
            # en el test si encontramos idle podemos simplemente pasar a la siguiente iteracion hacer nada
            if slot == "idle":
                action_name = "idle"
            else:
                action_name = ACTIONS[slot]
            # La salida son 15 acciones que tienen un valor distinto de 0
            # Como en esta iteracion nunca craftea, funde, etc. esas acciones se "filtran"
            # en el dataset, ya que no se realizan nunca
            print(f"{id_}: ({action_name}, {value})")
        break