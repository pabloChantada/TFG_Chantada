#!/usr/bin/env python3
"""
Hyperparameter sweep for the Minecraft IL pipeline.

Usage:
    python scripts/sweep.py                          # run SWEEP_CONFIGS in order
    python scripts/sweep.py --random 10             # random search, 10 samples
    python scripts/sweep.py --dataset data/my.jsonl # custom dataset
    python scripts/sweep.py --dry-run               # print configs and exit

Ctrl+C cancels after the current epoch finishes and saves all results up to that point.
"""

import sys
import os
import json
import argparse
import signal
import copy
import math
import random as pyrandom
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IL_DIR       = PROJECT_ROOT / "src" / "il"
sys.path.insert(0, str(IL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import MinecraftDataset
from model import ResNetExtractor, MinecraftILModel
from constants import STATE_DIM

# -- Cancellation flag ----------------------------------------------------------
_cancel = False

def _handle_sigint(sig, frame):
    global _cancel
    if _cancel:
        print("\n[sweep] Segunda interrupción — saliendo inmediatamente.")
        sys.exit(1)
    _cancel = True
    print("\n[sweep] Ctrl+C recibido. Se terminará el epoch actual y se guardarán los resultados.")

signal.signal(signal.SIGINT, _handle_sigint)

# -- Sweep grid -----------------------------------------------------------------
# Cada dict = un run de entrenamiento. Añade/elimina filas libremente.
# Baseline primero, luego una variable cambia a la vez.
SWEEP_CONFIGS = [
    # -- Baseline --------------------------------------------------------------
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    # -- LR --------------------------------------------------------------------
    dict(lr=5e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    dict(lr=1e-3, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    dict(lr=5e-5, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    # -- Batch size -------------------------------------------------------------
    dict(lr=1e-4, batch_size=16, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    dict(lr=1e-4, batch_size=64, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    # -- Weight decay ----------------------------------------------------------
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=1e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=1e-1, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
    # -- Label smoothing --------------------------------------------------------
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.0,  dropout=0.6, epochs=30, patience=3),
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.1,  dropout=0.6, epochs=30, patience=3),
    # -- Dropout ----------------------------------------------------------------
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.3, epochs=30, patience=3),
    dict(lr=1e-4, batch_size=32, backbone="resnet18",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.5, epochs=30, patience=3),
    # -- Backbone ---------------------------------------------------------------
    dict(lr=1e-4, batch_size=32, backbone="resnet34",  weight_decay=5e-2, label_smoothing=0.05, dropout=0.6, epochs=30, patience=3),
]

# -- Random search ranges -------------------------------------------------------
RANDOM_RANGES = {
    "lr":              ("log",    1e-4, 5e-3),
    "batch_size":      ("choice", [16, 32, 64]),
    "backbone":        ("choice", ["resnet18", "resnet34"]),
    "weight_decay":    ("log",    1e-3, 2e-1),
    "label_smoothing": ("uniform", 0.0, 0.15),
    "dropout":         ("uniform", 0.3, 0.7),
    "epochs":          ("fixed",  30),
    "patience":        ("fixed",  3),
}


# -- Helpers --------------------------------------------------------------------

def _backbone_abbr(backbone: str) -> str:
    return backbone.replace("resnet", "r")


def _fmt(val) -> str:
    """Human-readable value for run name."""
    if isinstance(val, float):
        if val == 0.0:
            return "0"
        exp = math.floor(math.log10(abs(val)))
        man = val / (10 ** exp)
        if abs(man - round(man)) < 1e-9:
            return f"{int(round(man))}e{exp}"
        return f"{man:.1f}e{exp}"
    return str(val)


def make_run_name(cfg: dict) -> str:
    """Produce a short, readable name from a config dict.

    Example: r18_lr1e-4_bs32_wd5e-2_ls0.05_do0.6
    """
    return (
        f"{_backbone_abbr(cfg['backbone'])}"
        f"_lr{_fmt(cfg['lr'])}"
        f"_bs{cfg['batch_size']}"
        f"_wd{_fmt(cfg['weight_decay'])}"
        f"_ls{_fmt(cfg['label_smoothing'])}"
        f"_do{_fmt(cfg['dropout'])}"
    )


def sample_random_config(rng: pyrandom.Random) -> dict:
    cfg = {}
    for key, spec in RANDOM_RANGES.items():
        kind = spec[0]
        if kind == "fixed":
            cfg[key] = spec[1]
        elif kind == "choice":
            cfg[key] = rng.choice(spec[1])
        elif kind == "uniform":
            cfg[key] = rng.uniform(spec[1], spec[2])
        elif kind == "log":
            log_lo = math.log10(spec[1])
            log_hi = math.log10(spec[2])
            cfg[key] = 10 ** rng.uniform(log_lo, log_hi)
    return cfg


# -- Core training loop ---------------------------------------------------------

LSTM_HIDDEN = 256


def _run_epoch(extractor, model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    extractor.train() if training else extractor.eval()
    model.train()     if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, states, labels in loader:
            # imgs:   (B, T, C, H, W)
            # states: (B, T, STATE_DIM)
            # labels: (B, T)
            imgs, states, labels = imgs.to(device), states.to(device), labels.to(device)
            B, T, C, H, W = imgs.shape

            if training:
                optimizer.zero_grad()

            features    = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
            logits      = model(features, states)        # (B, T, num_actions)
            logits_flat = logits.reshape(B * T, -1)
            labels_flat = labels.reshape(B * T)

            loss = criterion(logits_flat, labels_flat)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct    += (logits_flat.argmax(1) == labels_flat).sum().item()
            total      += B * T

    return total_loss / len(loader), correct / total


def train_one_config(cfg, dataset, device, run_dir):
    """Train a single config. Returns a result dict."""
    global _cancel

    # -- DataLoaders -----------------------------------------------------------
    base_train, base_val = dataset.load_dataset()
    train_loader = DataLoader(
        base_train.dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        base_val.dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    # -- Model -----------------------------------------------------------------
    extractor = ResNetExtractor(backbone=cfg["backbone"]).to(device)
    il_model  = MinecraftILModel(
        num_actions=len(dataset.pair2id),
        feat_dim=extractor.feat_dim,
        state_dim=STATE_DIM,
        lstm_hidden=LSTM_HIDDEN,
    ).to(device)

    # Override dropout in the head
    for m in il_model.head:
        if isinstance(m, nn.Dropout):
            m.p = cfg["dropout"]

    # -- Loss / optimizer / scheduler ------------------------------------------
    all_labels    = [d["label"] for d in dataset.data]
    class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights = np.clip(class_weights, None, 10)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["label_smoothing"])
    trainable = (list(filter(lambda p: p.requires_grad, extractor.parameters())) +
                 list(il_model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-6
    )

    # -- Training --------------------------------------------------------------
    model_path   = os.path.join(run_dir, "best_model.pth")
    config_path  = os.path.join(run_dir, "best_model_config.json")
    pair2id_path = os.path.join(run_dir, "best_model_pair2id.json")
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 0
    patience_ctr = 0
    status       = "completed"

    for epoch in range(cfg["epochs"]):
        if _cancel:
            status = "cancelled"
            break

        train_loss, train_acc = _run_epoch(extractor, il_model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc   = _run_epoch(extractor, il_model, val_loader,   criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  ep {epoch+1:02d}/{cfg['epochs']}  "
              f"train {train_acc:.1%} ({train_loss:.3f})  "
              f"val {val_acc:.1%} ({val_loss:.3f})  "
              f"lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save({"extractor": extractor.state_dict(),
                        "model":     il_model.state_dict()}, model_path)
            with open(pair2id_path, "w") as fh:
                json.dump(dataset.pair2id, fh, indent=2)
            with open(config_path, "w") as fh:
                json.dump({
                    "backbone":    cfg["backbone"],
                    "feat_dim":    extractor.feat_dim,
                    "state_dim":   STATE_DIM,
                    "lstm_hidden": LSTM_HIDDEN,
                    "num_actions": len(dataset.pair2id),
                    "seq_len":     dataset.seq_len,
                }, fh, indent=2)
            print(f"    [BEST] {best_val_acc:.1%}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"    [early stop @ ep {epoch+1}]")
                break

    return {
        "best_val_acc":   best_val_acc,
        "best_epoch":     best_epoch,
        "epochs_trained": len(history["val_acc"]),
        "status":         status,
        "history":        history,
        "model_path":     model_path if os.path.exists(model_path) else None,
    }


# -- Sweep runner ---------------------------------------------------------------

def run_sweep(configs, dataset, device, sweep_dir):
    global _cancel

    results_path = os.path.join(sweep_dir, "sweep_results.jsonl")
    results      = []

    print(f"\n{'='*60}")
    print(f"  SWEEP: {len(configs)} configs  —  {sweep_dir}")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(configs, 1):
        if _cancel:
            break

        name     = make_run_name(cfg)
        run_id   = f"{i:03d}_{name}"
        run_dir  = os.path.join(sweep_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'-'*60}")
        print(f"  Run {i}/{len(configs)}: {name}")
        print(f"  {cfg}")
        print(f"{'-'*60}")

        try:
            result = train_one_config(cfg, dataset, device, run_dir)
        except KeyboardInterrupt:
            _cancel = True
            print("\n[sweep] Interrupted during training — saving partial results.")
            result = {
                "best_val_acc": 0.0, "best_epoch": 0,
                "epochs_trained": 0, "status": "interrupted",
                "history": {}, "model_path": None,
            }

        entry = {
            "rank":          i,
            "run_id":        run_id,
            "name":          name,
            "config":        cfg,
            "best_val_acc":  result["best_val_acc"],
            "best_epoch":    result["best_epoch"],
            "epochs_trained": result["epochs_trained"],
            "status":        result["status"],
        }
        results.append(entry)

        # Persist after every run so Ctrl+C never loses data
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        # Save history separately for plotting
        hist_path = os.path.join(run_dir, "history.json")
        with open(hist_path, "w") as fh:
            json.dump(result["history"], fh)

        print(f"\n  → best_val_acc={result['best_val_acc']:.1%}  "
              f"epoch={result['best_epoch']}  status={result['status']}")

    # -- Final leaderboard ------------------------------------------------------
    completed = [r for r in results if r["status"] in ("completed", "cancelled")]
    if completed:
        completed.sort(key=lambda r: r["best_val_acc"], reverse=True)
        print(f"\n{'='*60}")
        print(f"  LEADERBOARD  ({len(completed)} runs completados)")
        print(f"{'='*60}")
        print(f"  {'#':<4}  {'val_acc':>7}  {'ep':>3}  {'name'}")
        print(f"  {'-'*52}")
        for rank, r in enumerate(completed, 1):
            flag = " *" if rank == 1 else ""
            print(f"  {rank:<4}  {r['best_val_acc']:>6.1%}  {r['best_epoch']:>3}  {r['name']}{flag}")
        print()

    print(f"Resultados guardados en: {results_path}")
    return results


# -- Entry point ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sweep de hiperparámetros IL")
    p.add_argument("--dataset",  default="data/train.jsonl")
    p.add_argument("--random",   type=int, default=0,
                   help="Número de configs aleatorias (0 = usar SWEEP_CONFIGS)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--dry-run",  action="store_true",
                   help="Mostrar configs y salir sin entrenar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -- Configs ---------------------------------------------------------------
    if args.random > 0:
        rng     = pyrandom.Random(args.seed)
        configs = [sample_random_config(rng) for _ in range(args.random)]
        print(f"Random search: {args.random} configs (seed={args.seed})")
    else:
        configs = SWEEP_CONFIGS

    if args.dry_run:
        print(f"\n{'-'*60}")
        print(f"  DRY RUN  {len(configs)} configs:")
        print(f"{'-'*60}")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i:>3}  {make_run_name(cfg)}")
            print(f"       {cfg}")
        sys.exit(0)

    # -- Dataset (cargado una sola vez) ----------------------------------------
    jsonl_path = str(PROJECT_ROOT / args.dataset)
    if not os.path.exists(jsonl_path):
        print(f"Error: dataset no encontrado en {jsonl_path}")
        sys.exit(1)

    dataset = MinecraftDataset(jsonl_path=jsonl_path)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweep_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"sweep_{sweep_ts}_{'rand' if args.random else 'grid'}"
    sweep_dir  = str(IL_DIR / "runs" / sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    # Save the full config list for reproducibility
    with open(os.path.join(sweep_dir, "configs.json"), "w") as fh:
        json.dump(configs, fh, indent=2)

    run_sweep(configs, dataset, device, sweep_dir)
