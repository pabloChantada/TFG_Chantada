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
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image', type=str, help='Imagen para inferencia única')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='Backbone ResNet a usar (default: resnet50)')
    return parser.parse_args()


def get_model_plots_dir(model_path):
    il_dir = os.path.dirname(os.path.abspath(__file__))
    plots_root = os.path.join(il_dir, "plots")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_plots_dir = os.path.join(plots_root, model_name)
    os.makedirs(model_plots_dir, exist_ok=True)
    return model_plots_dir


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
        
        # id → (slot, valor)
        id2pair = {v: k for k, v in dataset.pair2id.items()}
        slot, value = id2pair[pred_id]
        
        # Reconstruir vector 11
        action_vector = [0] * 11
        if slot != "idle":
            action_vector[slot] = value
            
        return action_vector, slot, value
    
def gradcam_single(model, dataset, image_path, plots_dir="."):
    """Para modo inference"""
    plot_gradcam(model, image_path, dataset, save_dir=plots_dir)

# TEST CONFIG
SCREEN_REGION = (0, 30, 1920, 1040)  
ACTIONS = {
    0: "move_forward",  # 0=still, 1=walk, 2=sprint
    1: "move_backward", # 0=still, 1=walk
    2: "move_lateral",  # 0=still, 1=left, 2=right
    3: "move_vertical", # 0=still, 1=jump, 2=sneak
    4: "camera_yaw",    # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    5: "camera_pitch",  # 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    6: "attack",        # 0=no, 1=yes
    7: "craft",         # 0=none,1=planks,2=stick,3=crafting_table,4=wpick,5=spick,6=ipick
    8: "smelt",         # 0=none, 1=iron_ingot
    9: "place",         # 0=none, 1=crafting_table, 2=furnace, 3=torch
    10: "equip"         # 0=none, 1=wpick, 2=spick, 3=ipick, 4=axe
}
DELAY = 0.1  # 100ms loop


if __name__ == "__main__":
    args = parse_args()
    
    JSONL_PATH = args.dataset
    MODEL_PATH = args.model
    BACKBONE = args.backbone
    PLOTS_DIR = get_model_plots_dir(MODEL_PATH)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    
    MODEL_DIR = os.path.dirname(MODEL_PATH)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
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
    CRITERION = nn.CrossEntropyLoss()
    
    print(f"=== Minecraft IL ===")
    print(f"Dataset: {JSONL_PATH}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Backbone: {BACKBONE}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Dispositivo: {device}")
    print(f"Clases: {num_actions}")
    
    # Print mappings
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    print("\nMappings ID → Acción:")
    for id_, (slot, val) in sorted(id2pair.items()):
        name = "idle" if slot == "idle" else ACTIONS.get(int(slot), f"slot_{slot}")
        print(f"  {id_}: {name}={val}")
    
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
                action_vec, slot, value = inference_step(model, img_path, dataset)
                gradcam_single(model, dataset, img_path, PLOTS_DIR)
                
                name = "idle" if slot == "idle" else ACTIONS.get(int(slot), f"slot_{slot}")
                
                print(f"\nPredicción:")
                print(f"  Acción: {name}={value}")
                print(f"  Vector: {action_vec}")
                print(f"  Slots:  {list(ACTIONS.keys())}")
                
            except Exception as e:
                print(f"Error: {e}")