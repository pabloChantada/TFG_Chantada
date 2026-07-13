# --- bootstrap de rutas: modulos de IL organizados por rol ---
import sys as _sys
from pathlib import Path as _Path
_IL_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ('', 'models', 'data', 'train', 'serve', 'analysis'):
    _p = str(_IL_ROOT / _sub) if _sub else str(_IL_ROOT)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- fin bootstrap ---
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
})
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch


def _save_path(filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


def plot_confusion_matrix(extractor, model, val_loader, dataset, device, normalize=True, save_dir="."):
    """Mapa de calor de confusion matrix."""
    if extractor is not None:
        extractor.eval()
    model.eval()
    all_preds, all_labels = [], []

    print("Calculando confusion matrix...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            # Compatibilidad: algunos loaders devuelven 3 items (imgs, states, labels)
            # y otros 4 (imgs, states, labels, camera_targets).
            if len(batch) < 3:
                raise ValueError(f"Batch inválido en val_loader: esperado >=3 elementos, recibido {len(batch)}")

            imgs, states, labels = batch[0], batch[1], batch[2]
            imgs, states, labels = imgs.to(device), states.to(device), labels.to(device)
            B, T, C, H, W = imgs.shape
            if extractor is not None:
                features = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
                logits, _ = model(features, states)
            elif type(model).__name__ == 'MinecraftConvLSTMModel':
                logits, _ = model(imgs, states)
            else:
                # Variante solo-estado: sin visión.
                logits, _ = model(None, states)
            preds = logits.argmax(dim=-1)[:, -1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels[:, -1].cpu().numpy())

    id2pair = {v: k for k, v in dataset.pair2id.items()}
    label_names = [id2pair[i] for i in range(len(id2pair))]

    cm = confusion_matrix(all_labels, all_preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix' + (' (Normalizada)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = _save_path('confusion_matrix.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")


def plot_training_history(history, save_dir="."):
    """Curvas de loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_path = _save_path('training_history.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")


def class_distribution(labels, dataset, save_dir="."):
    """Distribución de clases."""
    from collections import Counter
    id2pair = {v: k for k, v in dataset.pair2id.items()}

    counts = Counter(labels)
    pairs = [id2pair.get(i, i) for i in counts.keys()]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts.values())
    plt.xticks(range(len(counts)), pairs, rotation=45, ha='right')
    plt.title('Distribución de clases')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    output_path = _save_path('class_distribution.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")


def plot_camera_error(history, save_dir="."):
    """Curva de camera MSE loss (train vs val)."""
    if 'train_camera_loss' not in history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_action_loss'], label='Train Action Loss')
    ax1.plot(history['val_action_loss'],   label='Val Action Loss')
    ax1.set_title('Action Loss (CE)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_camera_loss'], label='Train Camera Loss')
    ax2.plot(history['val_camera_loss'],   label='Val Camera Loss')
    ax2.set_title('Camera Loss (MSE)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_path = _save_path('loss_breakdown.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")