import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from plots import * # Usamos todas asi que le ponemos *
from load_dataset import MinecraftDataset
from model import MinecraftILModel
import os, sys
from PIL import Image
import argparse
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

def parse_args():
    parser = argparse.ArgumentParser(description="Minecraft IL: Train o Inference")
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], 
                        default='train', help='Modo: train, eval, inference')
    parser.add_argument('--dataset', type=str, default='data\\train.jsonl',
                        help='Ruta al dataset JSONL')
    parser.add_argument('--model', type=str, default='src\\il\\models\\minecraft_model.pth',
                        help='Ruta al modelo')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs para training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image', type=str, help='Imagen para inferencia única')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='Backbone ResNet a usar (default: resnet50)')
    return parser.parse_args()


def get_run_dir(run_timestamp):
    """Create a unique directory for this run."""
    il_dir = os.path.dirname(os.path.abspath(__file__))
    runs_root = os.path.join(il_dir, "runs")
    run_dir = os.path.join(runs_root, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_model_plots_dir(run_dir):
    """Get plots subdirectory within run."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def train_epoch(model, loader, criterion, device, optimizer=None, is_training=True):
    # Repetimos la misma funcion tanto para el train como para el val
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss, correct, total = 0, 0, 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if is_training:
            # Training: con gradientes
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            # Validation: sin gradientes
            with torch.no_grad():
                logits = model(imgs)
                loss = criterion(logits, labels)
        
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss/len(loader), correct/total

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience=5, model_path="", plots_dir="."):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, device, optimizer, is_training=True)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Val
        val_loss, val_acc = train_epoch(model, val_loader, criterion, device, is_training=False)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train {train_acc:.1%} ({train_loss:.3f}), "
              f"Val {val_acc:.1%} ({val_loss:.3f})")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f"Mejor modelo guardado: {best_val_acc:.1%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
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
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f"VALIDATION FINAL: Loss {avg_loss:.3f}, Accuracy {accuracy:.1%}")
    return avg_loss, accuracy


def inference_step(model, image_path, dataset):  
    model.eval()
    
    with torch.no_grad():
        image = dataset.transforms(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        logits = model(image)  
        pred_id = logits.argmax(1).item()
        
        # id → action label
        id2pair = {v: k for k, v in dataset.pair2id.items()}
        action_label = id2pair[pred_id]

        return action_label
    
def gradcam_single(model, dataset, image_path, plots_dir="."):
    """Para modo inference"""
    plot_gradcam(model, image_path, dataset, save_dir=plots_dir)

# TEST CONFIG
SCREEN_REGION = (0, 30, 1920, 1040)
DELAY = 0.1  # 100ms loop


if __name__ == "__main__":
    args = parse_args()
    
    # Create unique run directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = get_run_dir(run_timestamp)
    
    JSONL_PATH = args.dataset
    BACKBONE = args.backbone
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = 1e-2
    PATIENCE = 3
    
    # Model path in run directory (independent per run)
    backbone_name = BACKBONE.replace("resnet", "r")
    MODEL_PATH = os.path.join(RUN_DIR, f"model_{backbone_name}_{run_timestamp}.pth")
    PLOTS_DIR = get_model_plots_dir(RUN_DIR)
    
    if not os.path.exists(JSONL_PATH):
        print(f"Error: El archivo {JSONL_PATH} no existe. Asegúrate de que el dataset esté en la ruta correcta.")
        exit(1)
    
    # Config device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carga dataset
    dataset = MinecraftDataset(jsonl_path=JSONL_PATH)
    train_loader, val_loader = dataset.load_dataset()
    num_actions = len(dataset.pair2id)
    
    # Crea modelo
    model = MinecraftILModel(num_actions=num_actions, backbone=BACKBONE).to(device)
    
    # Config optimizer y criterion
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Usar pesos de clase para balancear minimamente el entrenamiento debido al imbalance del dataset
    all_labels = [d['label'] for d in dataset.data]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = np.clip(class_weights, a_min=None, a_max=10)  # Evitar pesos extremos
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    CRITERION = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"=== Minecraft IL ===")
    print(f"Run ID: {run_timestamp}")
    print(f"Run Directory: {RUN_DIR}")
    print(f"Dataset: {JSONL_PATH}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Backbone: {BACKBONE}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Dispositivo: {device}")
    print(f"Clases: {num_actions}")
    
    # Print mappings
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    print("\nMappings ID → Acción:")
    for id_, action_name in sorted(id2pair.items()):
        print(f"  {id_}: {action_name}")
    
    if args.mode == "train":
        print("\nMODO TRAIN")
        train_model(model, train_loader, val_loader, OPTIMIZER, CRITERION, EPOCHS, PATIENCE, MODEL_PATH, PLOTS_DIR)
    
    elif args.mode == "eval":
        print("\nMODO EVAL")
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            eval_model(model, val_loader, CRITERION, device)
        else:
            print("No hay modelo entrenado para evaluar")
    
    else:  # inference
        print("\nMODO INFERENCE")
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print("Modelo cargado")
        else:
            print("WARNING: No hay modelo entrenado. Usando modelo random.")
        
        print("\n" + "="*50)
        
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

                print(f"\nPredicción:")
                print(f"  Acción: {action_label}")
                
            except Exception as e:
                print(f"Error: {e}")