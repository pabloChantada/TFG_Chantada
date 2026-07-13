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
regen_confusion.py — Regenera confusion_matrix e il_matriz_binaria desde el
checkpoint guardado de un run IL, sin reentrenar.

Uso:
  python src/il/regen_confusion.py src/il/runs/20260616_224015_gru
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/il (módulos core)

from load_dataset import MinecraftDataset
from model import RNNExtractor, ViTExtractor, MinecraftILModel
from model_convlstm import MinecraftConvLSTMModel
from constants import SEQ_LEN, STATE_DIM, IMG_SIZE, CAMERA_DIM

OUT = Path("modelo-tfg-fic-v1.6_2223xun/imaxes/07_il")


def load_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    cfg = json.load(open(cfg_path))

    model_type = cfg.get("model_type", "rnn")
    feat_dim   = cfg.get("feat_dim",   512)
    hidden_size= cfg.get("hidden_size",512)
    lstm_hidden= cfg.get("lstm_hidden",256)
    state_dim  = cfg.get("state_dim",  STATE_DIM)
    num_actions= cfg.get("num_actions", 3)
    img_size   = cfg.get("img_size",   IMG_SIZE)

    # Buscar el .pth
    pth_files = list(run_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No se encontró .pth en {run_dir}")
    ckpt_path = pth_files[0]
    print(f"Cargando: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "convlstm":
        model = MinecraftConvLSTMModel(
            num_actions=num_actions, feat_dim=feat_dim,
            state_dim=state_dim, lstm_hidden=lstm_hidden,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        extractor = None
    elif model_type == "state":
        model = MinecraftILModel(
            num_actions=num_actions, feat_dim=feat_dim,
            state_dim=state_dim, lstm_hidden=lstm_hidden,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        extractor = None
    elif model_type == "vit":
        extractor = ViTExtractor(feat_dim=feat_dim, img_size=img_size).to(device)
        extractor.load_state_dict(ckpt["extractor"])
        model = MinecraftILModel(
            num_actions=num_actions, feat_dim=feat_dim,
            state_dim=state_dim, lstm_hidden=lstm_hidden,
        ).to(device)
        model.load_state_dict(ckpt["model"])
    else:  # rnn / gru
        extractor = RNNExtractor(hidden_size=hidden_size).to(device)
        extractor.load_state_dict(ckpt["extractor"])
        model = MinecraftILModel(
            num_actions=num_actions, feat_dim=feat_dim,
            state_dim=state_dim, lstm_hidden=lstm_hidden,
        ).to(device)
        model.load_state_dict(ckpt["model"])

    return extractor, model, model_type, device


def get_predictions(extractor, model, model_type, val_loader, device):
    if extractor is not None:
        extractor.eval()
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inferencia"):
            imgs, states, labels = batch[0], batch[1], batch[2]
            imgs, states, labels = imgs.to(device), states.to(device), labels.to(device)
            B, T, C, H, W = imgs.shape

            if model_type == "convlstm":
                logits, _ = model(imgs, states)
            elif model_type == "state":
                logits, _ = model(None, states)
            else:
                feats = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
                logits, _ = model(feats, states)

            preds = logits.argmax(dim=-1)[:, -1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels[:, -1].cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_full_confusion(preds, labels, id2action, out_path: Path):
    label_names = [id2action[i] for i in sorted(id2action)]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title("Matriz de confusión (normalizada)")
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Predicción")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_binary_confusion(preds, labels, id2action, out_path: Path):
    """Colapsa acciones de movimiento en una sola clase (Mover/Explorar)."""
    MOVE_NAMES = {"move_forward_sprint", "move_forward_jump"}

    def to_binary(idx):
        name = id2action.get(idx, "")
        action_part = name.split("|")[0].strip() if "|" in name else name
        return 1 if action_part in MOVE_NAMES else 0

    bin_preds  = np.array([to_binary(p) for p in preds])
    bin_labels = np.array([to_binary(l) for l in labels])

    cm = confusion_matrix(bin_labels, bin_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    bin_names = ["Atacar", "Mover/Explorar"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=bin_names, yticklabels=bin_names, ax=ax,
                annot_kws={"size": 13})
    ax.set_title("Confusión binaria (Atacar vs Mover)")
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Predicción")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data",  type=str, default="data/train.jsonl")
    args = ap.parse_args()

    extractor, model, model_type, device = load_run(args.run_dir)

    dataset = MinecraftDataset(jsonl_path=args.data)
    dataset.filter_min_samples(100)
    _, val_loader = dataset.load_dataset(split_ratio=0.6, seed=42)

    id2action = {v: k for k, v in dataset.pair2id.items()}

    preds, labels = get_predictions(extractor, model, model_type, val_loader, device)

    OUT.mkdir(parents=True, exist_ok=True)
    plot_full_confusion(preds, labels, id2action,
                        OUT / "il_confusion_matrix.png")
    plot_binary_confusion(preds, labels, id2action,
                          OUT / "il_matriz_binaria.png")


if __name__ == "__main__":
    main()
