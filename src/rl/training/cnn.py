"""
YOLOv8 Tree Detector para bot Minecraft.
INFO para bot: coordenadas árboles + confianza.


DATASET: https://universe.roboflow.com/project-rslmo/minecraft-tree-detection-qrp0d
"""
from ultralytics import YOLO
from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "minecraft_trees" / "weights" / "best.pt"


def _find_latest_best_weights() -> Path:
    weights = sorted(
        (PROJECT_ROOT / "runs" / "detect").glob("*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not weights:
        raise FileNotFoundError(
            f"No se encontró ningún modelo YOLO en {(PROJECT_ROOT / 'runs' / 'detect')}"
        )
    return weights[0]


def _resolve_model_path(model_path=None) -> Path:
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"No existe el modelo YOLO especificado: {candidate}")
        return candidate

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    return _find_latest_best_weights()

class YOLOTreeDetector:
    def __init__(self, model_path=None):
        self.model_path = _resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def predict(self, image_path):
        """
        Devuelve lista de árboles detectados con formato:
        [{"bbox": [x1,y1,x2,y2], "confidence": 0.85}, ...]
        Esto lo podemos usar para que el bot mueva la camara hacia el árbol más cercano.
        Usando los radianes
        """
        results = self.model(image_path)
        
        trees = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            conf = box.conf[0].cpu().item()
            trees.append({"bbox": [x1,y1,x2,y2], "confidence": conf})
        
        print(f"Detectados {len(trees)} árboles")
        return trees
