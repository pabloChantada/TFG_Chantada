#!/usr/bin/env python3
"""
save_pair2id.py — Genera el fichero <modelo>_pair2id.json a partir del dataset.

Necesario para modelos entrenados antes de que main.py lo guardara automáticamente.

Uso:
  python scripts/save_pair2id.py --dataset data/train_clean.jsonl --model src/il/models/minecraft_model.pth
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IL_DIR = PROJECT_ROOT / "src" / "il"
sys.path.insert(0, str(IL_DIR))

from load_dataset import MinecraftDataset


def parse_args():
    p = argparse.ArgumentParser(description="Exporta pair2id del dataset a JSON")
    p.add_argument("--dataset", required=True, help="Ruta al JSONL de entrenamiento")
    p.add_argument("--model",   required=True, help="Ruta al .pth del modelo (determina el nombre del output)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not dataset_path.exists():
        print(f"Error: dataset no encontrado: {dataset_path}")
        sys.exit(1)

    print(f"Cargando dataset desde {dataset_path}...")
    ds = MinecraftDataset(jsonl_path=str(dataset_path))

    out_path = model_path.parent / (model_path.stem + "_pair2id.json")
    with open(out_path, "w") as f:
        json.dump(ds.pair2id, f, indent=2)

    print(f"\nGuardado: {out_path}")
    print(f"Acciones ({len(ds.pair2id)}):")
    for name, idx in sorted(ds.pair2id.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {name}")
