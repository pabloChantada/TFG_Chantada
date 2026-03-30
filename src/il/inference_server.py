#!/usr/bin/env python3
"""
inference_server.py — HTTP server que carga el modelo IL y sirve predicciones.

Uso:
  python src/il/inference_server.py --model src/il/models/minecraft_model.pth [--port 8765]

Endpoint:
  POST /predict   body: PNG en bytes (Content-Type: image/png o application/octet-stream)
  Respuesta:      {"action": "chop_tree", "confidence": 0.87, "top5": [["chop_tree", 0.87], ...]}
"""

import sys
import json
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IL_DIR       = Path(__file__).resolve().parent
sys.path.insert(0, str(IL_DIR))

from model     import MinecraftILModel
from constants import IMAGENET_MEAN, IMAGENET_STD

# ── Transform (idéntico al de entrenamiento, sin augmentación) ────────────────
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Globals (cargados en main) ────────────────────────────────────────────────
_model    = None
_id2action = None
_device   = None


def load_model_and_mapping(model_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device, weights_only=True)

    head_w = state.get("head.1.weight")
    legacy = head_w is None and "model.fc.weight" in state
    if legacy:
        head_w = state["model.fc.weight"]
        state["head.1.weight"] = state.pop("model.fc.weight")
        state["head.1.bias"]   = state.pop("model.fc.bias")

    if head_w is None:
        raise ValueError("No se encontró la clave 'head.1.weight' en el checkpoint.")

    num_actions = head_w.shape[0]
    linear_in   = head_w.shape[1]

    backbone_map = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}
    model, backbone, num_aux = None, None, None
    for bb, feat_dim in backbone_map.items():
        candidate_aux = linear_in - feat_dim
        if candidate_aux >= 0:
            try:
                m = MinecraftILModel(num_actions=num_actions, backbone=bb, num_aux=candidate_aux)
                m.load_state_dict(state)
                model, backbone, num_aux = m, bb, candidate_aux
                break
            except Exception:
                continue

    if model is None:
        raise RuntimeError("No se pudo determinar el backbone del modelo.")

    model = model.to(device).eval()
    print(f"Modelo cargado: backbone={backbone}, num_actions={num_actions}, num_aux={num_aux}")

    pair2id_path = Path(model_path).parent / (Path(model_path).stem + "_pair2id.json")
    if not pair2id_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{pair2id_path.name}'. "
            f"Genera el mapping con: python scripts/save_pair2id.py --model {model_path}"
        )
    with open(pair2id_path) as f:
        pair2id = json.load(f)

    id2action = {v: k for k, v in pair2id.items()}
    return model, id2action


def predict_from_bytes(png_bytes: bytes):
    """Devuelve (action_name, confidence, top5_list)."""
    image  = Image.open(BytesIO(png_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    k    = min(5, len(_id2action))
    top  = probs.topk(k)
    top5 = [((_id2action[i.item()]), round(p.item(), 4))
            for i, p in zip(top.indices, top.values)]
    best_id = probs.argmax().item()
    return _id2action[best_id], round(probs[best_id].item(), 4), top5


# ── HTTP handler ──────────────────────────────────────────────────────────────

class PredictHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silenciar el log de requests por defecto

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"error": f"Unknown endpoint: {self.path}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send(400, {"error": "Empty body"})
            return

        try:
            png_bytes = self.rfile.read(length)
            action, confidence, top5 = predict_from_bytes(png_bytes)
            self._send(200, {"action": action, "confidence": confidence, "top5": top5})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IL Inference Server")
    p.add_argument("--model", default="src/il/models/minecraft_model.pth")
    p.add_argument("--port",  type=int, default=8765)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {_device}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    _model, _id2action = load_model_and_mapping(str(model_path), _device)
    print(f"Acciones ({len(_id2action)}): {list(_id2action.values())}")

    server = HTTPServer(("localhost", args.port), PredictHandler)
    print(f"\nInference server listo en http://localhost:{args.port}/predict")
    print("Esperando peticiones... (Ctrl+C para salir)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
