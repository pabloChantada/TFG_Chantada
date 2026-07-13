#!/usr/bin/env python3
# --- bootstrap de rutas: modulos de IL organizados por rol ---
import sys as _sys
from pathlib import Path as _Path
_IL_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ('', 'models', 'data', 'train', 'serve', 'analysis'):
    _p = str(_IL_ROOT / _sub) if _sub else str(_IL_ROOT)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- fin bootstrap ---
"""
binary_confusion.py — Matriz de confusión BINARIA (atacar vs moverse) de un run IL.

Colapsa las tres clases del modelo (attack, move_forward_jump,
move_forward_sprint) en dos: "Atacar" y "Mover/Explorar". Sirve para justificar
que, ignorando la ambigüedad sprint/jump, el agente sí distingue acercarse del
tronco (mover) de talarlo (atacar).

Uso:
  python src/il/binary_confusion.py src/il/runs/<run> --out modelo-tfg-fic-v1.6_2223xun/imaxes/il_matriz_binaria.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Módulos core de IL (load_dataset, model, constants…) viven en src/il/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
})
import seaborn as sns
from sklearn.metrics import confusion_matrix

from load_dataset import MinecraftDataset
from model import RNNExtractor, ViTExtractor, MinecraftILModel
from model_convlstm import MinecraftConvLSTMModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from constants import SEQ_LEN, STATE_DIM, IMG_SIZE, CAMERA_DIM


def _is_attack(name: str) -> int:
    """0 = Atacar, 1 = Mover/Explorar."""
    return 0 if name == "attack" else 1


def load_run(run_dir: Path, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    model_path = next(run_dir.glob("model_*.pth"))
    pair2id = json.loads(next(run_dir.glob("*_pair2id.json")).read_text())
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    mt = cfg.get("model_type", "rnn")

    if mt == "convlstm":
        extractor = None
        model = MinecraftConvLSTMModel(num_actions=cfg["num_actions"],
                                       state_dim=cfg.get("state_dim", STATE_DIM),
                                       camera_dim=cfg.get("camera_dim", CAMERA_DIM))
        model.load_state_dict(ckpt["model"])
    elif mt == "state":
        extractor = None
        model = MinecraftILModel(num_actions=cfg["num_actions"], feat_dim=cfg["feat_dim"],
                                 state_dim=cfg.get("state_dim", STATE_DIM),
                                 lstm_hidden=cfg.get("lstm_hidden", 256),
                                 camera_dim=cfg.get("camera_dim", CAMERA_DIM))
        model.load_state_dict(ckpt["model"])
    else:
        if mt == "vit":
            extractor = ViTExtractor(img_size=cfg.get("img_size", IMG_SIZE),
                                     feat_dim=cfg["feat_dim"], pretrained=False)
        else:
            extractor = RNNExtractor(img_width=cfg.get("img_size", IMG_SIZE),
                                     hidden_size=cfg.get("hidden_size", 512))
        model = MinecraftILModel(num_actions=cfg["num_actions"], feat_dim=cfg["feat_dim"],
                                 state_dim=cfg.get("state_dim", STATE_DIM),
                                 lstm_hidden=cfg.get("lstm_hidden", 256),
                                 camera_dim=cfg.get("camera_dim", CAMERA_DIM))
        extractor.load_state_dict(ckpt["extractor"])
        model.load_state_dict(ckpt["model"])
        extractor = extractor.to(device).eval()
    model = model.to(device).eval()
    return extractor, model, mt, pair2id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="Directorio del run IL")
    ap.add_argument("--dataset", default="data/train.jsonl")
    ap.add_argument("--out", default="modelo-tfg-fic-v1.6_2223xun/imaxes/il_matriz_binaria.png")
    ap.add_argument("--split-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run)
    extractor, model, mt, pair2id = load_run(run_dir, device)
    id2pair = {v: k for k, v in pair2id.items()}

    dataset = MinecraftDataset(jsonl_path=args.dataset)
    _, val_loader = dataset.load_dataset(split_ratio=args.split_ratio, seed=args.seed)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs, states, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            B, T, C, H, W = imgs.shape
            if extractor is not None:
                feats = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
                logits, _ = model(feats, states)
            elif isinstance(model, MinecraftConvLSTMModel):
                logits, _ = model(imgs, states)
            else:
                logits, _ = model(None, states)
            preds = logits.argmax(dim=-1)[:, -1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels[:, -1].cpu().numpy())

    # Colapso binario: attack vs move. Las etiquetas vienen en el espacio de ids
    # del dataset (puede tener más clases que el modelo); las predicciones, en el
    # del modelo. Se mapea cada lado a NOMBRE con su propio pair2id antes de colapsar.
    ds_id2pair = {v: k for k, v in dataset.pair2id.items()}
    y_true = [_is_attack(ds_id2pair[i]) for i in all_labels]
    y_pred = [_is_attack(id2pair[i])    for i in all_preds]

    names = ["Atacar", "Mover/Explorar"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    acc = np.trace(cm) / cm.sum()
    print(f"Muestras val: {cm.sum()}  |  Accuracy binaria: {acc:.1%}")
    print("Matriz (filas=real, cols=predicho):")
    for i, n in enumerate(names):
        print(f"  {n:<16} {cm[i]}  ({cm_norm[i].round(3)})")

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, vmin=0, vmax=1,
                cbar_kws={"label": "Proporción por fila"})
    plt.title(f"Matriz de confusión binaria (acc. {acc:.1%})")
    plt.ylabel("Acción real")
    plt.xlabel("Acción predicha")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
