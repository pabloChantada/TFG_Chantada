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
Comparativa RNN vanilla vs GRU como extractor visual.

Mismo dataset, mismos hiperparámetros, misma seed.
Solo cambia el tipo de celda recurrente del extractor.
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/il (módulos core)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from load_dataset import MinecraftDataset
from model import MinecraftILModel
from constants import SEQ_LEN, STATE_DIM, IMG_SIZE, CAMERA_DIM

# ── Extractores ──────────────────────────────────────────────────────────────

class VanillaRNNExtractor(nn.Module):
    """Extractor con nn.RNN (Elman network, tanh)."""
    def __init__(self, img_channels=3, img_width=128, hidden_size=512, num_layers=2):
        super().__init__()
        self.feat_dim = hidden_size
        self.rnn = nn.RNN(
            input_size=img_channels * img_width,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )

    def forward(self, imgs):
        N, C, H, W = imgs.shape
        x = imgs.permute(0, 2, 1, 3).reshape(N, H, C * W)
        _, h_n = self.rnn(x)
        return h_n[-1]


class GRUExtractor(nn.Module):
    """Extractor con nn.GRU (el que usamos en producción)."""
    def __init__(self, img_channels=3, img_width=128, hidden_size=512, num_layers=2):
        super().__init__()
        self.feat_dim = hidden_size
        self.gru = nn.GRU(
            input_size=img_channels * img_width,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, imgs):
        N, C, H, W = imgs.shape
        x = imgs.permute(0, 2, 1, 3).reshape(N, H, C * W)
        _, h_n = self.gru(x)
        return h_n[-1]


# ── Entrenamiento ────────────────────────────────────────────────────────────

def train_epoch(extractor, model, loader, criterion, cam_criterion,
                cam_weight, device, optimizer=None, training=True):
    extractor.train() if training else extractor.eval()
    model.train()     if training else model.eval()
    total_loss, correct, total = 0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, states, labels, cam_targets in loader:
            imgs, states = imgs.to(device), states.to(device)
            labels, cam_targets = labels.to(device), cam_targets.to(device)
            B, T, C, H, W = imgs.shape

            if training:
                optimizer.zero_grad()

            feats = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
            logits, cam_pred = model(feats, states)

            logits_flat = logits.reshape(B * T, -1)
            labels_flat = labels.reshape(B * T)
            loss = (criterion(logits_flat, labels_flat)
                    + cam_weight * cam_criterion(
                        cam_pred.reshape(B * T, -1),
                        cam_targets.reshape(B * T, -1)))

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct    += (logits_flat.argmax(1) == labels_flat).sum().item()
            total      += B * T

    return total_loss / len(loader), correct / total


def run_experiment(name, extractor, model, train_loader, val_loader,
                   criterion, cam_criterion, cam_weight, device,
                   epochs=30, lr=1e-4, patience=5):
    params = list(extractor.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-6)

    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"  {name}  ({n_params:,} params)")
    print(f"{'='*50}")

    best_val_acc, patience_ctr = 0, 0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(
            extractor, model, train_loader, criterion, cam_criterion,
            cam_weight, device, optimizer, training=True)
        v_loss, v_acc = train_epoch(
            extractor, model, val_loader, criterion, cam_criterion,
            cam_weight, device, training=False)
        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        marker = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = ep
            patience_ctr = 0
            marker = " *BEST*"
        else:
            patience_ctr += 1

        print(f"  Ep {ep:2d}/{epochs}  "
              f"Train {t_acc:.1%} ({t_loss:.3f})  "
              f"Val {v_acc:.1%} ({v_loss:.3f})  "
              f"lr={lr_now:.1e}{marker}")

        if patience_ctr >= patience:
            print(f"  Early stopping en epoch {ep}")
            break

    print(f"\n  >> Mejor val accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    return best_val_acc


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEED        = 42
    HIDDEN_SIZE = 512
    LSTM_HIDDEN = 256
    EPOCHS      = 30
    LR          = 1e-4
    PATIENCE    = 5
    CAM_WEIGHT  = 1.0
    MIN_SAMPLES = 100

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (compartido)
    dataset = MinecraftDataset(jsonl_path="data/train.jsonl")
    if MIN_SAMPLES > 0:
        dataset.filter_min_samples(MIN_SAMPLES)
    train_loader, val_loader = dataset.load_dataset(split_ratio=0.6, seed=SEED)
    num_actions = len(dataset.pair2id)
    print(f"Dataset: {len(dataset.data)} muestras, {num_actions} clases")

    # Class weights
    all_labels  = [d["label"] for d in dataset.data]
    present     = np.unique(all_labels)
    raw_w       = compute_class_weight("balanced", classes=present, y=all_labels)
    class_w     = np.full(num_actions, 10.0)
    for cls, w in zip(present, raw_w):
        class_w[cls] = min(w, 10.0)
    class_weights = torch.tensor(class_w, dtype=torch.float).to(device)

    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.001)
    cam_criterion = nn.MSELoss()

    # ── RNN vanilla ──────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    rnn_ext = VanillaRNNExtractor(img_width=IMG_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    rnn_model = MinecraftILModel(
        num_actions=num_actions, feat_dim=rnn_ext.feat_dim,
        state_dim=STATE_DIM, lstm_hidden=LSTM_HIDDEN, camera_dim=CAMERA_DIM,
    ).to(device)
    rnn_acc = run_experiment(
        "RNN vanilla (nn.RNN)", rnn_ext, rnn_model,
        train_loader, val_loader, criterion, cam_criterion,
        CAM_WEIGHT, device, EPOCHS, LR, PATIENCE)

    # ── GRU ──────────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    gru_ext = GRUExtractor(img_width=IMG_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    gru_model = MinecraftILModel(
        num_actions=num_actions, feat_dim=gru_ext.feat_dim,
        state_dim=STATE_DIM, lstm_hidden=LSTM_HIDDEN, camera_dim=CAMERA_DIM,
    ).to(device)
    gru_acc = run_experiment(
        "GRU (nn.GRU)", gru_ext, gru_model,
        train_loader, val_loader, criterion, cam_criterion,
        CAM_WEIGHT, device, EPOCHS, LR, PATIENCE)

    # ── Resumen ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  RESUMEN")
    print(f"{'='*50}")
    print(f"  RNN vanilla : {rnn_acc:.1%}")
    print(f"  GRU         : {gru_acc:.1%}")
    diff = gru_acc - rnn_acc
    print(f"  Diferencia  : {diff:+.1%} {'(GRU gana)' if diff > 0 else '(RNN gana)' if diff < 0 else '(empate)'}")
