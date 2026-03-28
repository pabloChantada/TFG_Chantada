import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def get_last_conv_layer(model):
    """Última capa convolucional de ResNet18: layer4[-1]"""
    return model.model.layer4[-1]

class MinecraftGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar activations y gradientes
        self.forward_hook = target_layer.register_forward_hook(self.forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        """Captura activations del forward pass"""
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        """Captura gradientes del backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Genera Grad-CAM heatmap para imagen de entrada"""
        self.model.eval()
        
        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass para la clase objetivo
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM: pesos = media global de gradientes por canal
        gradients = self.gradients[0]  # [C, H, W]
        weights = torch.mean(gradients, dim=(1,2), keepdim=True)  # [C, 1, 1]
        cam = torch.sum(weights * self.activations[0], dim=0)  # [H, W]
        
        # Post-procesado: ReLU + normalización [0,1]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), target_class
    
    def release(self):
        """Limpia hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Superpone heatmap sobre imagen original"""
    # Resize heatmap al tamaño de la imagen
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Colormap JET (azul→rojo)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (heatmap_colored * alpha + image * (1 - alpha)).astype(np.uint8)
    return overlay

def plot_gradcam(model, image_path, dataset, ACTIONS):
    """Visualización completa Grad-CAM"""
    # Carga y preprocess imagen
    orig_image = Image.open(image_path).convert("RGB")
    input_tensor = dataset.transforms(orig_image).unsqueeze(0).to(next(model.parameters()).device)
    
    # Target layer + Grad-CAM
    target_layer = get_last_conv_layer(model)
    cam_extractor = MinecraftGradCAM(model, target_layer)
    
    heatmap, pred_class = cam_extractor.generate(input_tensor)
    
    # Mapeo a nombre de acción
    id2pair = {v: k for k, v in dataset.pair2id.items()}
    pred_action = id2pair[pred_class]
    slot_name = "idle" if pred_action[0] == "idle" else ACTIONS.get(pred_action[0], f"slot_{pred_action[0]}")
    
    # Imagen para visualización (resize sin normalizar)
    img_array = np.array(orig_image.resize((224, 224)))
    
    # Overlay heatmap
    overlay = overlay_heatmap(img_array, heatmap)
    
    # Plot 3 paneles
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Imagen original')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'Grad-CAM: {slot_name}={pred_action[1]}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (α=0.4)')
    axes[2].axis('off')
    
    plt.tight_layout()
    filename = f'gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    cam_extractor.release()
    print(f"Grad-CAM guardado: {filename}")
    print(f"Predicción: {slot_name}={pred_action[1]}")
    
    return pred_class, pred_action