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
regen_figures.py — Regenera las figuras de entrenamiento IL con fuente más grande.

Lee los history.json ya guardados de cada run (no necesita modelo ni Minecraft).
Escribe directamente en imaxes/07_il/.

Uso:
  python src/il/regen_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ── Fuente grande para la memoria ─────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
})

RUNS = {
    "GRU":      "src/il/runs/20260616_224015_gru",
    "ViT":      "src/il/runs/20260616_232018_vit",
    "ConvLSTM": "src/il/runs/20260616_230636_conv",
    "State":    "src/il/runs/20260616_221104_state",
}
OUT = Path("modelo-tfg-fic-v1.6_2223xun/imaxes/07_il")
OUT.mkdir(parents=True, exist_ok=True)


def load_history(run_dir: str) -> dict:
    return json.load(open(Path(run_dir) / "plots" / "history.json"))


# ── 1. Curvas de entrenamiento (una figura con todos los runs) ─────────────────
def plot_training_history_all():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, run_dir) in enumerate(RUNS.items()):
        try:
            h = load_history(run_dir)
        except FileNotFoundError:
            print(f"  [skip] {label}: no se encontró history.json")
            continue
        c = colors[i % len(colors)]
        axes[0].plot(h["train_loss"], label=f"{label} train", color=c, linestyle="-")
        axes[0].plot(h["val_loss"],   label=f"{label} val",   color=c, linestyle="--")
        axes[1].plot(h["train_acc"],  label=f"{label} train", color=c, linestyle="-")
        axes[1].plot(h["val_acc"],    label=f"{label} val",   color=c, linestyle="--")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(ncol=2, fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT / "il_training_history.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


# ── 2. Loss acción vs cámara del mejor run (GRU) ──────────────────────────────
def plot_loss_breakdown(run_label="GRU"):
    run_dir = RUNS[run_label]
    h = load_history(run_dir)
    if "train_action_loss" not in h:
        print(f"  [skip] {run_label}: sin desglose de loss")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(h["train_action_loss"], label="Train")
    axes[0].plot(h["val_action_loss"],   label="Val")
    axes[0].set_title(f"Action Loss (CE) — {run_label}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h["train_camera_loss"], label="Train")
    axes[1].plot(h["val_camera_loss"],   label="Val")
    axes[1].set_title(f"Camera Loss (MSE) — {run_label}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT / "il_loss_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    print("Regenerando figuras IL...")
    plot_training_history_all()
    plot_loss_breakdown("GRU")
    print("Hecho. Quedan por regenerar con modelo+dataset:")
    print("  il_confusion_matrix.png → python src/il/main.py (o cargar modelo)")
    print("  il_matriz_binaria.png   → python src/il/binary_confusion.py src/il/runs/20260616_224015_gru")
    print("  il_action_dist.png      → ver comando en el chat")
