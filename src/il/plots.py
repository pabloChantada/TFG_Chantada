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


def _ensure_output_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def _get_save_path(filename, save_dir):
    output_dir = _ensure_output_dir(save_dir)
    return os.path.join(output_dir, filename)

def simple_progress(current, total, desc=""):
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
        for imgs, labels in tqdm(val_loader, desc="Eval"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    id2pair = {v: k for k, v in dataset.pair2id.items()}
    label_names = [f"{id2pair[i]}" for i in range(len(id2pair))]

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
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + image * (1 - alpha)).astype(np.uint8)

def plot_gradcam(model, image_path, dataset, save_dir=".", img_size=(448, 448)):
    """Grad-CAM en alta resolución"""
    orig_image   = Image.open(image_path).convert("RGB")
    high_res_img = orig_image.resize(img_size, Image.Resampling.LANCZOS)
    input_tensor = dataset.transforms(high_res_img).unsqueeze(0).to(next(model.parameters()).device)

    target_layer  = get_last_conv_layer(model)
    cam_extractor = MinecraftGradCAM(model, target_layer)
    heatmap, pred_class = cam_extractor.generate(input_tensor)

    id2pair    = {v: k for k, v in dataset.pair2id.items()}
    pred_label = str(id2pair[pred_class])

    img_array      = np.array(high_res_img)
    heatmap_highres = cv2.resize(heatmap, (img_size[1], img_size[0]),
                                 interpolation=cv2.INTER_CUBIC)
    overlay = overlay_heatmap(img_array, heatmap_highres)

    _, axes = plt.subplots(1, 3, figsize=(20, 7))

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
    image_stem  = os.path.splitext(os.path.basename(image_path))[0]
    output_path = _get_save_path(f'gradcam_hd_{image_stem}.png', save_dir)
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.show()

    cam_extractor.release()
    print(f"Grad-CAM HD guardado: {output_path}")
    print(f"Predicción: {pred_label}")
