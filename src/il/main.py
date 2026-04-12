import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os
import argparse
from datetime import datetime
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

from plots import *
from load_dataset import MinecraftDataset, MIRROR_PAIRS
from model import RNNExtractor, ViTExtractor, MinecraftILModel
from model_convlstm import MinecraftConvLSTMModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from constants import SEQ_LEN, STATE_DIM, IMG_SIZE, CAMERA_DIM

LSTM_HIDDEN = 256


class TeeLogger:
    """Redirige sys.stdout al terminal Y a un fichero simultáneamente."""
    def __init__(self, log_path: str):
        self._terminal = sys.stdout
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def isatty(self):
        return False

    def close(self):
        sys.stdout = self._terminal
        self._file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Minecraft IL: Train")
    parser.add_argument('--dataset',     type=str,   default='data/train.jsonl')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--hidden-size', type=int,   default=512,
                        help='Tamaño del estado oculto del GRU extractor')
    parser.add_argument('--min-samples', type=int,   default=0,
                        help='Eliminar clases con menos de N muestras (default: 0 = desactivado)')
    parser.add_argument('--model',       type=str,   default='rnn',
                        choices=['rnn', 'convlstm', 'vit'],
                        help='Arquitectura del modelo: rnn (GRU+LSTM), convlstm (CNN+ConvLSTM) o vit (ViT+LSTM)')
    parser.add_argument('--vit-model',   type=str,   default='vit_small_patch16_224',
                        help='Nombre del modelo timm para el extractor ViT')
    parser.add_argument('--vit-feat-dim', type=int,  default=256,
                        help='Dimensión del embedding de salida del ViTExtractor')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Entrenar el ViT desde cero (sin pesos ImageNet)')
    parser.add_argument('--vit-unfreeze-blocks', type=int, default=0,
                        help='Nº de bloques ViT finales a entrenar (0=solo proj, -1=todo)')
    return parser.parse_args()


def get_run_dir(run_timestamp):
    il_dir  = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(il_dir, "runs", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir, model_type, hidden_size, feat_dim, num_actions):
    config = {
        "model_type":  model_type,
        "img_size":    IMG_SIZE,
        "hidden_size": hidden_size,
        "feat_dim":    feat_dim,
        "state_dim":   STATE_DIM,
        "lstm_hidden": LSTM_HIDDEN,
        "num_actions": num_actions,
        "camera_dim":  CAMERA_DIM,
        "seq_len":     SEQ_LEN,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


def print_run_summary(args, run_timestamp, extractor, model, dataset,
                      train_loader, val_loader, class_weights, criterion,
                      weight_decay, patience, device, camera_loss_weight=1.0):
    SEP  = "=" * 56
    SEP2 = "-" * 56

    id2pair      = {v: k for k, v in dataset.pair2id.items()}
    ext_trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad) if extractor else 0
    ext_total     = sum(p.numel() for p in extractor.parameters()) if extractor else 0
    trainable    = ext_trainable + sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = ext_total     + sum(p.numel() for p in model.parameters())
    label_counts = Counter(d["label"] for d in dataset.data)

    active_pairs = [(a, b) for a, b in MIRROR_PAIRS
                    if a in dataset.pair2id and b in dataset.pair2id]

    train_sessions = {item["session_id"] for item in train_loader.dataset.data}
    val_sessions   = {item["session_id"] for item in val_loader.dataset.data}

    print(SEP)
    print("  RUN SUMMARY")
    print(SEP)
    print(f"  Run ID     : {run_timestamp}")
    print(f"  Fecha      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dispositivo: {device}")
    print()

    print(f"  {SEP2}")
    print(f"  DATASET")
    print(f"  {SEP2}")
    print(f"  Archivo  : {args.dataset}")
    print(f"  Total    : {len(dataset.data)} muestras")
    print(f"  Train    : {len(train_loader.dataset)} muestras ({len(train_sessions)} sesiones)")
    print(f"  Val      : {len(val_loader.dataset)} muestras ({len(val_sessions)} sesiones)")
    print(f"  Seq len  : {dataset.seq_len}")
    print()
    print(f"  Distribución de clases:")
    for label_id in sorted(id2pair):
        name = id2pair[label_id]
        n    = label_counts.get(label_id, 0)
        w    = class_weights[label_id].item()
        bar  = "#" * int(n / max(label_counts.values()) * 20)
        print(f"    {label_id}  {name:<22} n={n:5d}  w={w:.3f}  {bar}")
    print()

    print(f"  {SEP2}")
    print(f"  MODELO")
    print(f"  {SEP2}")
    if extractor is not None:
        ext_name = type(extractor).__name__
        if ext_name == 'ViTExtractor':
            print(f"  Extractor    : ViTExtractor({extractor.vit.default_cfg['architecture'] if hasattr(extractor.vit, 'default_cfg') else ''}, img_size={IMG_SIZE}, feat_dim={extractor.feat_dim})")
        else:
            print(f"  Extractor    : RNNExtractor(img_size={IMG_SIZE}, hidden={extractor.feat_dim})")
        print(f"  state_proj   : Linear({STATE_DIM} → {extractor.feat_dim})")
        print(f"  LSTM         : ({extractor.feat_dim} → {model.lstm_hidden})")
        print(f"  action_head  : Dropout(0.5) → Linear({model.lstm_hidden} → {len(id2pair)})")
        print(f"  camera_head  : Dropout(0.3) → Linear({model.lstm_hidden} → {model.camera_dim}) → Tanh")
    else:
        print(f"  Arquitectura : ConvLSTM (CNN + ConvLSTMCell)")
        print(f"  CNN channels : {model.cnn_channels}")
        print(f"  ConvLSTM h   : {model.hidden_channels}")
        print(f"  state_dim    : {model.state_dim}")
    print(f"  Parámetros   : {trainable:,} entrenables / {total_params:,} total")
    print()

    print(f"  {SEP2}")
    print(f"  HIPERPARÁMETROS")
    print(f"  {SEP2}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LR inicial   : {args.lr:.2e}")
    print(f"  Weight decay : {weight_decay}")
    print(f"  Batch size   : {train_loader.batch_size}")
    print(f"  Patience     : {patience}")
    ls = getattr(criterion, "label_smoothing", 0)
    print(f"  Loss action  : CrossEntropyLoss(label_smoothing={ls}, class_weights=balanced)")
    print(f"  Loss camera  : MSELoss")
    print(f"  Camera weight: {camera_loss_weight}")
    print(f"  Optimizer    : AdamW")
    print(f"  Scheduler    : ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)")
    print()

    if active_pairs:
        print(f"  {SEP2}")
        print(f"  AUGMENTACIONES (train)")
        print(f"  {SEP2}")
        pairs_str = ", ".join(f"{a}<->{b}" for a, b in active_pairs)
        print(f"  MirrorFlip(p=0.5, swap_labels=[{pairs_str}])")
        print()

    print(SEP)
    print()


def train_epoch(extractor, model, loader, criterion, camera_criterion,
                device, camera_loss_weight=1.0, optimizer=None, is_training=True):
    if extractor is not None:
        extractor.train() if is_training else extractor.eval()
    model.train()     if is_training else model.eval()
    total_loss, total_action_loss, total_camera_loss = 0, 0, 0
    correct, total = 0, 0

    for imgs, states, labels, camera_targets in loader:
        imgs   = imgs.to(device)
        states = states.to(device)
        labels = labels.to(device)
        camera_targets = camera_targets.to(device)
        B, T, C, H, W = imgs.shape

        if is_training:
            optimizer.zero_grad()

        if extractor is not None:
            features = extractor(imgs.reshape(B * T, C, H, W)).view(B, T, -1)
            logits, camera_pred = model(features, states)
        else:
            logits, camera_pred = model(imgs, states)

        # Loss acción discreta
        logits_flat = logits.reshape(B * T, -1)
        labels_flat = labels.reshape(B * T)
        action_loss = criterion(logits_flat, labels_flat)

        # Loss cámara continua
        camera_loss = camera_criterion(
            camera_pred.reshape(B * T, -1),
            camera_targets.reshape(B * T, -1),
        )

        loss = action_loss + camera_loss_weight * camera_loss

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss        += loss.item()
        total_action_loss += action_loss.item()
        total_camera_loss += camera_loss.item()
        correct           += (logits_flat.argmax(1) == labels_flat).sum().item()
        total             += B * T

    n = len(loader)
    return (total_loss / n, total_action_loss / n, total_camera_loss / n,
            correct / total)


def train(extractor, model, dataset, train_loader, val_loader, optimizer, scheduler,
          criterion, camera_criterion, camera_loss_weight, epochs, patience,
          model_path, plots_dir):
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_action_loss': [], 'train_camera_loss': [],
        'val_action_loss': [],   'val_camera_loss': [],
    }
    best_val_acc = 0
    patience_ctr = 0

    for epoch in range(epochs):
        train_loss, train_act_loss, train_cam_loss, train_acc = train_epoch(
            extractor, model, train_loader, criterion, camera_criterion,
            device, camera_loss_weight, optimizer, is_training=True)
        val_loss, val_act_loss, val_cam_loss, val_acc = train_epoch(
            extractor, model, val_loader, criterion, camera_criterion,
            device, camera_loss_weight, is_training=False)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_action_loss'].append(train_act_loss)
        history['train_camera_loss'].append(train_cam_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_action_loss'].append(val_act_loss)
        history['val_camera_loss'].append(val_cam_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train {train_acc:.1%} (act={train_act_loss:.3f} cam={train_cam_loss:.3f})  "
              f"Val {val_acc:.1%} (act={val_act_loss:.3f} cam={val_cam_loss:.3f})  "
              f"lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {'model': model.state_dict()}
            if extractor is not None:
                checkpoint['extractor'] = extractor.state_dict()
            torch.save(checkpoint, model_path)
            pair2id_path = os.path.splitext(model_path)[0] + "_pair2id.json"
            with open(pair2id_path, "w") as f:
                json.dump(dataset.pair2id, f, indent=2)
            patience_ctr = 0
            print(f"  [BEST] Mejor modelo guardado: {best_val_acc:.1%}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping en epoch {epoch+1}")
                break

    plot_training_history(history, save_dir=plots_dir)
    plot_confusion_matrix(extractor, model, val_loader, dataset, device, save_dir=plots_dir)
    class_distribution([d['label'] for d in dataset.data], dataset, save_dir=plots_dir)
    plot_camera_error(history, save_dir=plots_dir)

    return history


if __name__ == "__main__":
    args = parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = get_run_dir(run_timestamp)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    log_path  = os.path.join(run_dir, "log.txt")
    logger    = TeeLogger(log_path)

    HIDDEN_SIZE       = args.hidden_size
    WEIGHT_DECAY      = 1e-2
    PATIENCE          = 5
    CAMERA_LOSS_WEIGHT = 1.0
    SPLIT_RATIO       = 0.6
    
    model_path = os.path.join(run_dir, f"model_rnn_{run_timestamp}.pth")

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} no existe.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MinecraftDataset(jsonl_path=args.dataset)
    if args.min_samples > 0:
        dataset.filter_min_samples(args.min_samples)
    train_loader, val_loader = dataset.load_dataset(split_ratio=SPLIT_RATIO, seed=42)
    num_actions = len(dataset.pair2id)

    if args.model == 'convlstm':
        extractor = None
        model = MinecraftConvLSTMModel(
            num_actions=num_actions,
            state_dim=STATE_DIM,
            camera_dim=CAMERA_DIM,
        ).to(device)
        model_path = os.path.join(run_dir, f"model_convlstm_{run_timestamp}.pth")
    elif args.model == 'vit':
        extractor = ViTExtractor(
            img_size=IMG_SIZE,
            feat_dim=args.vit_feat_dim,
            model_name=args.vit_model,
            pretrained=not args.no_pretrained,
            unfreeze_blocks=args.vit_unfreeze_blocks,
        ).to(device)
        model = MinecraftILModel(
            num_actions=num_actions,
            feat_dim=extractor.feat_dim,
            state_dim=STATE_DIM,
            lstm_hidden=LSTM_HIDDEN,
            camera_dim=CAMERA_DIM,
        ).to(device)
        model_path = os.path.join(run_dir, f"model_vit_{run_timestamp}.pth")
    else:
        extractor = RNNExtractor(img_width=IMG_SIZE, hidden_size=HIDDEN_SIZE).to(device)
        model = MinecraftILModel(
            num_actions=num_actions,
            feat_dim=extractor.feat_dim,
            state_dim=STATE_DIM,
            lstm_hidden=LSTM_HIDDEN,
            camera_dim=CAMERA_DIM,
        ).to(device)

    save_run_config(run_dir, args.model, HIDDEN_SIZE,
                    extractor.feat_dim if extractor else 0, num_actions)

    all_labels    = [d['label'] for d in dataset.data]
    present       = np.unique(all_labels)
    raw_weights   = compute_class_weight('balanced', classes=present, y=all_labels)
    # Asignar peso máximo a clases sin muestras
    class_weights = np.full(num_actions, 10.0)
    for cls, w in zip(present, raw_weights):
        class_weights[cls] = min(w, 10.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion        = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.001)
    camera_criterion = nn.MSELoss()
    if extractor is not None:
        trainable_params = (
            list(filter(lambda p: p.requires_grad, extractor.parameters())) +
            list(model.parameters())
        )
    else:
        trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, min_lr=1e-6,
    )

    print_run_summary(args, run_timestamp, extractor, model, dataset,
                      train_loader, val_loader, class_weights, criterion,
                      WEIGHT_DECAY, PATIENCE, device, CAMERA_LOSS_WEIGHT)

    id2pair = {v: k for k, v in dataset.pair2id.items()}
    print("Mappings ID -> Accion:")
    for id_, name in sorted(id2pair.items()):
        print(f"  {id_}: {name}")
    print()

    try:
        train(extractor, model, dataset, train_loader, val_loader,
              optimizer, scheduler, criterion, camera_criterion,
              CAMERA_LOSS_WEIGHT, args.epochs, PATIENCE,
              model_path, plots_dir)

        import shutil
        shutil.copy(os.path.join(run_dir, "config.json"),
                    os.path.splitext(model_path)[0] + "_config.json")
        print(f"\nLog guardado en: {log_path}")
    finally:
        logger.close()