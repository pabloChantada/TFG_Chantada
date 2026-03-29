import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import cv2
from PIL import Image
from gradcam import MinecraftGradCAM, get_last_conv_layer

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

def _ensure_output_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def _get_save_path(filename, save_dir):
    output_dir = _ensure_output_dir(save_dir)
    return os.path.join(output_dir, filename)

def simple_progress(current, total, desc=""):
    """Progress bar"""
    progress = current / total
    bar_length = 20
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r{desc} [{bar}] {progress:.1%} ({current}/{total})", end="", flush=True)
    if current == total:
        print()

def plot_confusion_matrix(model, val_loader, dataset, device, normalize=True, save_dir="."):
    """Mapa de calor de confusion matrix"""
    model.eval()
    all_preds, all_labels = [], []
    
    print("Calculando confusion matrix...")
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(val_loader, desc="Eval")):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Mappings
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    labels = [f"{id2pair[i]}" for i in range(len(id2pair))]
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix' + (' (Normalizada)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path = _get_save_path('confusion_matrix.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")

def plot_training_history(history, save_dir="."):
    """Curvas de loss/accuracy"""
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
    output_path = _get_save_path('training_history.png', save_dir)
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")

def class_distribution(labels, dataset, save_dir="."):
    """Distribución de clases"""
    from collections import Counter
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    
    counts = Counter(labels)
    pairs = [id2pair.get(i, i) for i in counts.keys()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts.values())
    plt.xticks(range(len(counts)), pairs, rotation=45, ha='right')
    plt.title('Distribución de clases (Imbalance)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    output_path = _get_save_path('class_distribution.png', save_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Guardado: {output_path}")

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap sobre imagen original"""
    # Heatmap a 3 canales + resize
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (heatmap * alpha + image * (1-alpha)).astype(np.uint8)
    return overlay

def plot_gradcam(model, image_path, dataset, save_dir=".", img_size=(448, 448)):
    """Grad-CAM en ALTA RESOLUCIÓN"""
    # Carga imagen ORIGINAL (alta resolución)
    orig_image = Image.open(image_path).convert("RGB")
    
    # Resize A MÁS RESOLUCIÓN para mejor detalle
    high_res_img = orig_image.resize(img_size, Image.Resampling.LANCZOS)
    input_tensor = dataset.transforms(high_res_img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Target layer + Grad-CAM
    target_layer = get_last_conv_layer(model)
    cam_extractor = MinecraftGradCAM(model, target_layer)
    
    # Genera Grad-CAM
    heatmap, pred_class = cam_extractor.generate(input_tensor)
    
    # Mappings
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    pred_action = id2pair[pred_class]
    if isinstance(pred_action, tuple):
        # Legacy format: (slot, value)
        name = "idle" if pred_action[0] == "idle" else ACTIONS.get(pred_action[0], f"slot_{pred_action[0]}")
        pred_label = f"{name}={pred_action[1]}"
    else:
        # New format: discrete string label
        pred_label = str(pred_action)
    
    # Imagen para visualización ALTA RES
    img_array = np.array(high_res_img)  # 448x448 ahora
    
    # INTERPOLA heatmap a alta resolución
    heatmap_highres = cv2.resize(heatmap, (img_size[1], img_size[0]), 
                                interpolation=cv2.INTER_CUBIC)
    
    # Overlay
    overlay = overlay_heatmap(img_array, heatmap_highres)
    
    # Plot AÚN MÁS GRANDE
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))  # Más ancho
    
    axes[0].imshow(img_array)
    axes[0].set_title(f'Imagen original ({img_size[0]}x{img_size[1]})', fontsize=14)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(heatmap_highres, cmap='jet')
    axes[1].set_title(f'Grad-CAM HD: {pred_label}', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Alta Resolución', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    output_path = _get_save_path(f'gradcam_hd_{image_stem}.png', save_dir)
    plt.savefig(output_path, dpi=400, bbox_inches='tight')  # DPI más alto
    plt.show()
    
    cam_extractor.release()
    print(f"Grad-CAM HD guardado: {output_path}")
    print(f"Predicción: {pred_label}")