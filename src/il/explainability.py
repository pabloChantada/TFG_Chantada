"""
explainability.py — Técnicas de explicabilidad para MinecraftILModel + ResNetExtractor.

Técnicas:
  - FullModelGradCAM    : GradCAM con gradientes desde la salida final del pipeline
  - IntegratedGradients : atribución pixel a pixel (baseline → imagen real)
  - state_importance    : importancia de cada variable de estado del bot

Uso rápido:
    from explainability import FullModelGradCAM, IntegratedGradients, state_importance
    from explainability import plot_explainability

    cam = FullModelGradCAM(extractor, il_model)
    heatmap, cls = cam.generate(image_tensor, feat_buffer, state_buffer)
    cam.release()

Convenciones de dimensiones:
    image_tensor  : (1, C, H, W)
    feat_buffer   : (1, T-1, feat_dim)  — features de frames anteriores, sin grad
    state_buffer  : (1, T, state_dim)   — estados completos de la secuencia

# Imágenes directas (historial = frame repetido)
python src/il/generate_explainability.py --model src/il/models/model.pth --images frame1.png frame2.png

# Dataset jsonl (usa la secuencia temporal real de cada muestra)
python src/il/generate_explainability.py --model src/il/models/model.pth --jsonl data/train.jsonl --n-samples 10 --seed 42


--techniques gradcam state      # solo esas dos, sin IG (más rápido)
--ig-steps 20                   # IG más rápido (menos preciso)
--state 120.5,64,88.3,90,0      # estado del bot manual (solo con --images)
--output-dir mi_carpeta/        # directorio de salida personalizado
--show                          # abrir ventana interactiva además de guardar

"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


STATE_LABELS = ["x", "y", "z", "yaw", "pitch"]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _relu_normalize(t: torch.Tensor) -> np.ndarray:
    """ReLU + normalización [0, 1] → numpy."""
    t = F.relu(t)
    t = (t - t.min()) / (t.max() + 1e-8)
    return t.squeeze().cpu().numpy()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Superpone un heatmap normalizado [0,1] sobre una imagen RGB (H, W, 3).
    El heatmap se redimensiona automáticamente al tamaño de la imagen.
    """
    h = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    h = np.uint8(255 * h)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (colored * alpha + image_np * (1 - alpha)).astype(np.uint8)


def _run_full_pipeline(extractor, il_model, image_tensor, feat_buffer, state_buffer):
    """Forward completo image → extractor → LSTM → head. Retorna logits (1, T, A)."""
    feat  = extractor(image_tensor)                          # (1, feat_dim)
    feats = torch.cat([feat_buffer, feat.unsqueeze(1)], dim=1)  # (1, T, feat_dim)
    return il_model(feats, state_buffer), feat               # logits, feat_actual


# ---------------------------------------------------------------------------
# 1. GradCAM — pipeline completo
# ---------------------------------------------------------------------------

class FullModelGradCAM:
    """
    GradCAM sobre la última capa convolucional de ResNet (layer4[-1]) con
    gradientes que fluyen por el pipeline completo:

        imagen → ResNet(layer4[-1]) → GAP → state_proj → LSTM → head → logits

    Al propagar desde los logits finales, la importancia de cada región
    refleja la decisión del modelo completo, no solo del backbone visual.
    """

    def __init__(self, extractor, il_model):
        self.extractor = extractor
        self.il_model  = il_model
        self._grads    = None
        self._acts     = None

        target   = extractor.net.layer4[-1]
        self._fh = target.register_forward_hook(self._save_acts)
        self._bh = target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out):  self._acts  = out.detach()
    def _save_grads(self, m, gi, go):   self._grads = go[0].detach()

    def generate(self, image_tensor, feat_buffer, state_buffer,
                 target_class: int | None = None):
        """
        Returns:
            heatmap (H', W') float32 normalizado [0, 1]
            target_class int
        """
        # train() necesario para que cuDNN permita backward a través del LSTM
        self.extractor.train()
        self.il_model.train()
        self.extractor.zero_grad()
        self.il_model.zero_grad()

        logits, _ = _run_full_pipeline(extractor=self.extractor,
                                       il_model=self.il_model,
                                       image_tensor=image_tensor,
                                       feat_buffer=feat_buffer,
                                       state_buffer=state_buffer)
        last = logits[:, -1, :]  # (1, num_actions)

        if target_class is None:
            target_class = last.argmax(dim=1).item()

        one_hot = torch.zeros_like(last)
        one_hot[0, target_class] = 1.0
        last.backward(gradient=one_hot)

        # CAM = suma ponderada de activaciones por los gradientes medios por canal
        weights = self._grads[0].mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam     = (weights * self._acts[0]).sum(dim=0)            # (H', W')

        return _relu_normalize(cam), target_class

    def release(self):
        """Elimina los hooks. Llamar siempre al terminar."""
        self._fh.remove()
        self._bh.remove()


# ---------------------------------------------------------------------------
# 2. Integrated Gradients
# ---------------------------------------------------------------------------

class IntegratedGradients:
    """
    Atribución pixel a pixel mediante integración de gradientes
    (Sundararajan et al., 2017).

    Integra el gradiente de la salida respecto a la imagen a lo largo del
    camino lineal desde una imagen base (ceros) hasta la imagen real.
    Satisface los axiomas de implementación-invarianza y completitud.

    El resultado es más preciso que GradCAM a nivel de píxel, aunque más
    lento (requiere `steps` forward+backward passes).
    """

    def __init__(self, extractor, il_model, steps: int = 50):
        self.extractor = extractor
        self.il_model  = il_model
        self.steps     = steps

    def generate(self, image_tensor, feat_buffer, state_buffer,
                 target_class: int | None = None, baseline=None):
        """
        Args:
            baseline : (1, C, H, W) imagen de referencia; None → zeros
        Returns:
            attribution (H, W) float32 normalizado [0, 1]
            target_class int
        """
        if baseline is None:
            baseline = torch.zeros_like(image_tensor)

        # Obtener target_class sin gradientes
        if target_class is None:
            self.extractor.eval()
            self.il_model.eval()
            with torch.no_grad():
                logits, _ = _run_full_pipeline(self.extractor, self.il_model,
                                               image_tensor, feat_buffer, state_buffer)
                target_class = logits[0, -1].argmax().item()

        accumulated_grads = torch.zeros_like(image_tensor)

        self.extractor.train()
        self.il_model.train()

        for i in range(self.steps + 1):
            alpha  = i / self.steps
            interp = (baseline + alpha * (image_tensor - baseline)).detach().requires_grad_(True)

            self.extractor.zero_grad()
            self.il_model.zero_grad()

            logits, _ = _run_full_pipeline(self.extractor, self.il_model,
                                           interp, feat_buffer, state_buffer)
            score = logits[0, -1, target_class]
            score.backward()

            accumulated_grads = accumulated_grads + interp.grad.detach()

        # IG = (imagen − baseline) × gradiente_medio
        ig = (image_tensor - baseline) * accumulated_grads / (self.steps + 1)

        # Colapsar canales RGB → magnitud escalar por píxel (H, W)
        attribution = ig.squeeze(0).abs().sum(dim=0)          # (H, W)
        attribution = (attribution - attribution.min()) / (attribution.max() + 1e-8)
        return attribution.cpu().numpy(), target_class


# ---------------------------------------------------------------------------
# 3. State importance
# ---------------------------------------------------------------------------

def state_importance(extractor, il_model, image_tensor, feat_buffer,
                     state_buffer, target_class: int | None = None):
    """
    Importancia de cada variable de estado (x, y, z, yaw, pitch) calculada
    como la magnitud del gradiente de la salida respecto al vector de estado.

    Responde a: "¿cuánto cambia la confianza en la acción si perturbamos
    ligeramente cada variable de estado?"

    Returns:
        importance  np.ndarray (state_dim,) — valores en [0, 1]
        target_class int
    """
    extractor.eval()
    il_model.train()  # train() necesario para backward a través del LSTM en cuDNN

    # Extraer features sin gradientes (no necesitamos explicar la imagen aquí)
    with torch.no_grad():
        feat  = extractor(image_tensor)
        feats = torch.cat([feat_buffer, feat.unsqueeze(1)], dim=1)  # (1, T, feat_dim)

    # Clonar el state_buffer con grad habilitado
    state_w_grad = state_buffer.detach().requires_grad_(True)

    il_model.zero_grad()
    logits = il_model(feats, state_w_grad)  # (1, T, num_actions)
    last   = logits[:, -1, :]               # (1, num_actions)

    if target_class is None:
        target_class = last.argmax(dim=1).item()

    last[0, target_class].backward()

    # Gradiente en el último timestep: (state_dim,)
    grad = state_w_grad.grad[0, -1].abs().cpu().numpy()
    importance = grad / (grad.max() + 1e-8)  # normalizar [0, 1]

    return importance, target_class


# ---------------------------------------------------------------------------
# Visualización combinada
# ---------------------------------------------------------------------------

def plot_explainability(orig_image_np: np.ndarray, results: dict,
                        action_name: str, save_path: str | None = None):
    """
    Panel combinado con todas las técnicas disponibles.

    Args:
        orig_image_np : imagen RGB (H, W, 3) sin normalizar
        results       : dict con claves opcionales:
                          'gradcam' → np.ndarray (H, W)
                          'ig'      → np.ndarray (H, W)
                          'state'   → np.ndarray (state_dim,)
        action_name   : nombre de la acción predicha para el título
        save_path     : ruta de guardado (None = no guardar)
    """
    has_gradcam = "gradcam" in results
    has_ig      = "ig"      in results
    has_state   = "state"   in results

    visual_panels = 1 + sum([has_gradcam, has_ig])  # orig + los que haya
    n_rows = 2 if has_state else 1

    fig = plt.figure(figsize=(5 * visual_panels, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, visual_panels, figure=fig,
                            hspace=0.45, wspace=0.25)

    img_224 = cv2.resize(orig_image_np, (224, 224))
    col = 0

    # Imagen original
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img_224)
    ax.set_title("Imagen original", fontsize=11)
    ax.axis("off")
    col += 1

    # GradCAM overlay
    if has_gradcam:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(overlay_heatmap(img_224, results["gradcam"]))
        ax.set_title("GradCAM (pipeline completo)", fontsize=11)
        ax.axis("off")
        col += 1

    # Integrated Gradients overlay
    if has_ig:
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(overlay_heatmap(img_224, results["ig"]))
        ax.set_title("Integrated Gradients", fontsize=11)
        ax.axis("off")

    # State importance (segunda fila, span completo)
    if has_state:
        ax  = fig.add_subplot(gs[1, :])
        imp = results["state"]
        labels = STATE_LABELS[:len(imp)]

        colors = ["#e74c3c" if v > 0.5 else "#3498db" for v in imp]
        bars   = ax.bar(labels, imp, color=colors, edgecolor="white", width=0.5)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Importancia relativa", fontsize=10)
        ax.set_title(f"Importancia de variables de estado — acción: {action_name}",
                     fontsize=11)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        for bar, val in zip(bars, imp):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(f"Explicabilidad del modelo — acción predicha: {action_name}",
                 fontsize=13, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Guardado: {save_path}")

    plt.show()
    return fig
