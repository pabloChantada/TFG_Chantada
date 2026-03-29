import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from plots import *
from load_dataset import MinecraftDataset
from model import MinecraftILModel
import os
from PIL import Image
import argparse
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight


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


def train_epoch(model, loader, criterion, device, optimizer=None, is_training=True):
    model.train() if is_training else model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if is_training:
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs)
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
            patience_ctr = 0
            print(f"  ★ Mejor modelo guardado: {best_val_acc:.1%}")
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
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
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

    JSONL_PATH   = args.dataset
    BACKBONE     = args.backbone
    EPOCHS       = args.epochs
    LR           = args.lr
    WEIGHT_DECAY = 1e-2
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

    model = MinecraftILModel(num_actions=num_actions, backbone=BACKBONE).to(device)

    # Solo parámetros entrenables (backbone parcialmente congelado en model.py)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    # Pesos de clase inversamente proporcionales a su frecuencia
    all_labels    = [d['label'] for d in dataset.data]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = np.clip(class_weights, a_min=None, a_max=10)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    CRITERION = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    OPTIMIZER = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    # Reduce LR x0.5 si val_loss no mejora en 2 epochs consecutivos
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
        OPTIMIZER, mode='min', patience=2, factor=0.5, min_lr=1e-6
    )

    id2pair = {v: k for k, v in dataset.pair2id.items()}

    print(f"=== Minecraft IL ===")
    print(f"Run:        {run_timestamp}")
    print(f"Dataset:    {JSONL_PATH}")
    print(f"Backbone:   {BACKBONE}  ({trainable:,} / {total:,} params entrenables)")
    print(f"Clases:     {num_actions}")
    print(f"Dispositivo:{device}")
    print("\nMappings ID → Acción:")
    for id_, name in sorted(id2pair.items()):
        print(f"  {id_}: {name}")

    if args.mode == "train":
        print("\nMODO TRAIN")
        train_model(model, train_loader, val_loader, OPTIMIZER, SCHEDULER,
                    CRITERION, EPOCHS, PATIENCE, MODEL_PATH, PLOTS_DIR)

    elif args.mode == "eval":
        print("\nMODO EVAL")
        if os.path.exists(args.model):
            model.load_state_dict(torch.load(args.model, map_location=device))
            eval_model(model, val_loader, CRITERION, device)
        else:
            print(f"Modelo no encontrado: {args.model}")

    else:  # inference
        print("\nMODO INFERENCE")
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
                print(f"\nPredicción: {action_label}")
            except Exception as e:
                print(f"Error: {e}")
