"""
YOLOv8 para detección de árboles Minecraft (coordenadas + confianza).
Reemplaza completamente tu ResNet classifier.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "Minecraft Tree Detection.yolov8"
DEFAULT_TEST_DIR = PROJECT_ROOT / "src" / "metrics" / "agent_metrics" / "Rec_5_f1ly7q_screenshots"


def _find_latest_best_weights() -> Path:
    weights = sorted(
        (PROJECT_ROOT / "runs" / "detect").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not weights:
        raise FileNotFoundError("No se encontró ningún best.pt en runs/detect/*/weights")
    return weights[0]

def train_yolo():
    # Entrena desde pesos base de YOLOv8n
    model = YOLO('yolo26s.pt') 
    data_yaml = DATASET_PATH / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe data.yaml en: {data_yaml}")
    
    # ENTRENAR
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        name='minecraft_trees',
        patience=5
    )
    
    best_weights = _find_latest_best_weights()
    print(f"Modelo en: {best_weights}")
    return YOLO(str(best_weights))

def test_model(model, test_dir: Path, max_images: int = 8):
    if not test_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de test: {test_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in image_exts])

    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en: {test_dir}")

    images = images[:max_images]
    print(f"Testeando {len(images)} imágenes de: {test_dir}")

    results = model.predict(
        source=[str(p) for p in images],
        save=True,
        imgsz=640,
        conf=0.25,
        name="minecraft_trees_test",
        exist_ok=True,
    )

    for img, res in zip(images, results):
        print(f"{img.name}: {len(res.boxes)} detecciones")

    print("Detección completada")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento/Test de YOLOv8 para detección de árboles")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--weights", type=str, default="", help="Ruta a best.pt para modo test")
    parser.add_argument("--test_dir", type=str, default=str(DEFAULT_TEST_DIR), help="Carpeta con imágenes para test")
    parser.add_argument("--max_images", type=int, default=8, help="Número máximo de imágenes a testear")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = None

    if args.mode in {"train", "both"}:
        model = train_yolo()

    if args.mode in {"test", "both"}:
        if model is None:
            weights_path = Path(args.weights).expanduser().resolve() if args.weights else _find_latest_best_weights()
            if not weights_path.exists():
                raise FileNotFoundError(f"No existe el modelo: {weights_path}")
            model = YOLO(str(weights_path))

        test_model(model, Path(args.test_dir).expanduser().resolve(), max_images=args.max_images)