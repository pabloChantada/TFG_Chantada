import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import sys
import json
from plots import *
from load_dataset import MinecraftDataset, MIRROR_PAIRS
from model import MinecraftILModel
import os
from PIL import Image
import argparse
from datetime import datetime
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


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
        return False  # deshabilita animaciones de tqdm en el fichero

    def close(self):
        sys.stdout = self._terminal
        self._file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Minecraft IL: Train o Inference")
    parser.add_argument('--mode',     choices=['train', 'eval', 'inference'], default='train')
    parser.add_argument('--dataset',  type=str,   default='data/train.jsonl',
                        help='Ruta al dataset JSONL')
    parser.add_argument('--model',    type=str,   default='src/il/models/minecraft_model.pth',
                        help='Ruta al modelo')
    parser.add_argument('--epochs',   type=int,   default=30)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--backbone', type=str,   default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--image',    type=str,   help='Imagen para inferencia única')
    return parser.parse_args()


def get_run_dir(run_timestamp):
    il_dir    = os.path.dirname(os.path.abspath(__file__))
    run_dir   = os.path.join(il_dir, "runs", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_model_plots_dir(run_dir):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def write_run_summary(args, run_timestamp, model, dataset, train_loader, val_loader,
                      class_weights, criterion, optimizer, scheduler,
                      weight_decay, patience, device):
    """Imprime el resumen de configuración del run al inicio del log."""
    SEP  = "=" * 56
    SEP2 = "-" * 56

    id2pair      = {v: k for k, v in dataset.pair2id.items()}
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    label_counts = Counter(d["label"] for d in dataset.data)

    # Frozen / trainable layers
    frozen, unfrozen = [], []
    for name, param in model.model.named_parameters():
        layer = name.split(".")[0]
        (unfrozen if param.requires_grad else frozen).append(layer)
    frozen_layers   = sorted(set(frozen))
    unfrozen_layers = sorted(set(unfrozen))

    # Mirror pairs actually active
    active_pairs = [(a, b) for a, b in MIRROR_PAIRS
                    if a in dataset.pair2id and b in dataset.pair2id]

    # Session counts
    from load_dataset import MIRROR_PAIRS as _  # already imported
    train_sessions = set()
    val_sessions   = set()
    for item in train_loader.dataset.data:
        parts = item["image_path"].replace("\\", "/").split("/")
        train_sessions.add(parts[-2] if len(parts) >= 2 else "?")
    for item in val_loader.dataset.data:
        parts = item["image_path"].replace("\\", "/").split("/")
        val_sessions.add(parts[-2] if len(parts) >= 2 else "?")

    print(SEP)
    print("  RUN SUMMARY")
    print(SEP)
    print(f"  Run ID   : {run_timestamp}")
    print(f"  Fecha    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modo     : {args.mode}")
    print(f"  Dispositivo: {device}")
    print()

    print(f"  {SEP2}")
    print(f"  DATASET")
    print(f"  {SEP2}")
    print(f"  Archivo  : {args.dataset}")
    print(f"  Total    : {len(dataset.data)} muestras")
    print(f"  Train    : {len(train_loader.dataset)} muestras ({len(train_sessions)} sesiones)")
    print(f"  Val      : {len(val_loader.dataset)} muestras ({len(val_sessions)} sesiones)")
    print(f"  Aux feat : {dataset.num_aux}  (tree_visible, tree_distance_norm)" if dataset.num_aux else "  Aux feat : ninguna")
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
    print(f"  Backbone     : {model.backbone_name}")
    print(f"  Capas frozen : {', '.join(frozen_layers)}")
    print(f"  Capas libres : {', '.join(unfrozen_layers)}")
    print(f"  Parámetros   : {trainable:,} entrenables / {total_params:,} total")
    print(f"  Cabeza       : Dropout(0.5) -> Linear({512 if 'resnet18' in model.backbone_name or 'resnet34' in model.backbone_name else 2048} + {model.num_aux}, {len(id2pair)})")
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
    print(f"  Loss         : CrossEntropyLoss(label_smoothing={ls}, class_weights=balanced)")
    print(f"  Optimizer    : AdamW")
    print(f"  Scheduler    : ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)")
    print()

    print(f"  {SEP2}")
    print(f"  AUGMENTACIONES (train)")
    print(f"  {SEP2}")
    print(f"  ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)")
    print(f"  RandomGrayscale(p=0.1)")
    if active_pairs:
        pairs_str = ", ".join(f"{a}<->{b}" for a, b in active_pairs)
        print(f"  MirrorFlip(p=0.5, swap_labels=[{pairs_str}], flip_yaw_delta=True)")
    print()

    print(SEP)
    print()


def train_epoch(model, loader, criterion, device, optimizer=None, is_training=True):
    model.train() if is_training else model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, aux, labels in loader:
        imgs, aux, labels = imgs.to(device), aux.to(device), labels.to(device)

        if is_training:
            optimizer.zero_grad()
            logits = model(imgs, aux)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs, aux)
                loss   = criterion(logits, labels)

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, epochs, patience=5, model_path="", plots_dir="."):
    history      = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_ctr = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, device,
                                            optimizer, is_training=True)
        val_loss,   val_acc   = train_epoch(model, val_loader,   criterion, device,
                                            is_training=False)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Ajustar LR si val_loss se estanca
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train {train_acc:.1%} ({train_loss:.3f})  "
              f"Val {val_acc:.1%} ({val_loss:.3f})  "
              f"lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            # Guardar pair2id junto al modelo para que test_model.py pueda cargarlo
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
    plot_confusion_matrix(model, val_loader, dataset, device, save_dir=plots_dir)
    class_distribution([d['label'] for d in dataset.data], dataset, save_dir=plots_dir)

    return history


def eval_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, aux, labels in val_loader:
            imgs, aux, labels = imgs.to(device), aux.to(device), labels.to(device)
            logits = model(imgs, aux)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f"VALIDATION FINAL: Loss {avg_loss:.3f}, Accuracy {accuracy:.1%}")
    return avg_loss, accuracy


def inference_step(model, image_path, dataset):
    model.eval()
    with torch.no_grad():
        image    = dataset.transforms(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        pred_id  = model(image).argmax(1).item()
        id2pair  = {v: k for k, v in dataset.pair2id.items()}
        return id2pair[pred_id]


def gradcam_single(model, dataset, image_path, plots_dir="."):
    plot_gradcam(model, image_path, dataset, save_dir=plots_dir)


if __name__ == "__main__":
    args = parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR    = get_run_dir(run_timestamp)
    PLOTS_DIR  = get_model_plots_dir(RUN_DIR)

    # Log file: captura todo el output del run (summary + entrenamiento)
    LOG_PATH = os.path.join(RUN_DIR, "log.txt")
    logger   = TeeLogger(LOG_PATH) if args.mode != "inference" else None

    JSONL_PATH   = args.dataset
    BACKBONE     = args.backbone
    EPOCHS       = args.epochs
    LR           = args.lr
    WEIGHT_DECAY = 5e-2
    PATIENCE     = 3

    backbone_name = BACKBONE.replace("resnet", "r")
    MODEL_PATH    = os.path.join(RUN_DIR, f"model_{backbone_name}_{run_timestamp}.pth")

    if not os.path.exists(JSONL_PATH):
        print(f"Error: {JSONL_PATH} no existe.")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MinecraftDataset(jsonl_path=JSONL_PATH)
    train_loader, val_loader = dataset.load_dataset()
    num_actions = len(dataset.pair2id)

    model = MinecraftILModel(num_actions=num_actions, backbone=BACKBONE, num_aux=dataset.num_aux).to(device)

    # Solo parámetros entrenables (backbone parcialmente congelado en model.py)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    # Pesos de clase inversamente proporcionales a su frecuencia
    all_labels    = [d['label'] for d in dataset.data]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = np.clip(class_weights, a_min=None, a_max=10)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    CRITERION = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    OPTIMIZER = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    # Reduce LR x0.5 si val_loss no mejora en 2 epochs consecutivos
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMIZER, mode='min', patience=2, factor=0.5, min_lr=1e-6
    )

    write_run_summary(args, run_timestamp, model, dataset, train_loader, val_loader,
                      class_weights, CRITERION, OPTIMIZER, SCHEDULER,
                      WEIGHT_DECAY, PATIENCE, device)

    id2pair = {v: k for k, v in dataset.pair2id.items()}
    print("Mappings ID -> Accion:")
    for id_, name in sorted(id2pair.items()):
        print(f"  {id_}: {name}")
    print()

    try:
        if args.mode == "train":
            print("MODO TRAIN\n")
            train_model(model, train_loader, val_loader, OPTIMIZER, SCHEDULER,
                        CRITERION, EPOCHS, PATIENCE, MODEL_PATH, PLOTS_DIR)
            print(f"\nLog guardado en: {LOG_PATH}")

        elif args.mode == "eval":
            print("MODO EVAL\n")
            if os.path.exists(args.model):
                model.load_state_dict(torch.load(args.model, map_location=device))
                eval_model(model, val_loader, CRITERION, device)
            else:
                print(f"Modelo no encontrado: {args.model}")

        else:  # inference — no logger, interactive
            print("MODO INFERENCE\n")
            model_path = args.model if os.path.exists(args.model) else MODEL_PATH
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Modelo cargado")
            else:
                print("WARNING: Sin modelo entrenado. Usando pesos aleatorios.")

            while True:
                img_path = input("\nRuta imagen (o 'q' para salir): ").strip()
                if img_path.lower() == 'q':
                    break
                if not os.path.exists(img_path):
                    print("Imagen no encontrada")
                    continue
                try:
                    action_label = inference_step(model, img_path, dataset)
                    gradcam_single(model, dataset, img_path, PLOTS_DIR)
                    print(f"\nPrediccion: {action_label}")
                except Exception as e:
                    print(f"Error: {e}")
    finally:
        if logger:
            logger.close()
