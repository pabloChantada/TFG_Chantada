#!/usr/bin/env python3
"""
inference_server.py — HTTP server que carga el modelo IL y sirve predicciones.

Uso:
  python src/il/inference_server.py --model src/il/models/minecraft_model.pth [--port 8765]

Endpoint:
  POST /predict
    Body    : PNG en bytes (image/png)
    Header  : X-Bot-State: x,y,z,yaw,pitch  (floats separados por coma)
    Respuesta: {"action": "attack", "confidence": 0.87, "top5": [...]}

El servidor mantiene un buffer de los últimos SEQ_LEN frames. Cada petición añade
un nuevo frame y predice usando la secuencia completa (último timestep).
"""

import sys
import json
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from collections import deque

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IL_DIR       = Path(__file__).resolve().parent
sys.path.insert(0, str(IL_DIR))

from model     import ResNetExtractor, MinecraftILModel
from constants import IMAGENET_MEAN, IMAGENET_STD, SEQ_LEN, STATE_DIM

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Globals
_extractor    = None
_model        = None
_id2action    = None
_device       = None
_feat_buffer  = None   # deque de tensors (feat_dim,)
_state_buffer = None   # deque de tensors (STATE_DIM,)


def load_model(model_path: str, device: torch.device):
    """Carga extractor + modelo temporal desde checkpoint y config JSON."""
    model_p  = Path(model_path)
    config_p = model_p.parent / (model_p.stem + "_config.json")
    pair2id_p = model_p.parent / (model_p.stem + "_pair2id.json")

    if not config_p.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_p}")
    if not pair2id_p.exists():
        raise FileNotFoundError(f"pair2id no encontrado: {pair2id_p}")

    with open(config_p) as f:
        cfg = json.load(f)
    with open(pair2id_p) as f:
        pair2id = json.load(f)

    extractor = ResNetExtractor(backbone=cfg["backbone"])
    model     = MinecraftILModel(
        num_actions=cfg["num_actions"],
        feat_dim=cfg["feat_dim"],
        state_dim=cfg.get("state_dim", STATE_DIM),
        lstm_hidden=cfg.get("lstm_hidden", 256),
    )

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    extractor.load_state_dict(ckpt["extractor"])
    model.load_state_dict(ckpt["model"])

    extractor = extractor.to(device).eval()
    model     = model.to(device).eval()

    id2action = {v: k for k, v in pair2id.items()}
    seq_len   = cfg.get("seq_len", SEQ_LEN)

    print(f"Modelo cargado: backbone={cfg['backbone']}, "
          f"num_actions={cfg['num_actions']}, seq_len={seq_len}")
    return extractor, model, id2action, seq_len


def _parse_state(header_val: str) -> torch.Tensor:
    """Parsea 'x,y,z,yaw,pitch' del header X-Bot-State."""
    try:
        vals = [float(v) for v in header_val.split(",")]
        if len(vals) == STATE_DIM:
            return torch.tensor(vals, dtype=torch.float32)
    except Exception:
        pass
    return torch.zeros(STATE_DIM, dtype=torch.float32)


def predict_from_bytes(png_bytes: bytes, state_header: str):
    """Añade el frame al buffer y predice la acción del timestep actual."""
    global _feat_buffer, _state_buffer

    # Extraer features del frame actual
    image  = Image.open(BytesIO(png_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(_device)  # (1, C, H, W)
    with torch.no_grad():
        feat = _extractor(tensor)[0]  # (feat_dim,)

    state = _parse_state(state_header).to(_device)  # (STATE_DIM,)

    _feat_buffer.append(feat)
    _state_buffer.append(state)

    # Rellenar si el buffer no está lleno todavía
    seq_len = _feat_buffer.maxlen
    feats_list  = list(_feat_buffer)
    states_list = list(_state_buffer)
    while len(feats_list) < seq_len:
        feats_list  = [feats_list[0]]  + feats_list
        states_list = [states_list[0]] + states_list

    # (1, T, feat_dim) y (1, T, STATE_DIM)
    feats_t  = torch.stack(feats_list).unsqueeze(0)   # (1, T, feat_dim)
    states_t = torch.stack(states_list).unsqueeze(0)  # (1, T, STATE_DIM)

    with torch.no_grad():
        logits = _model(feats_t, states_t)          # (1, T, num_actions)
        probs  = F.softmax(logits[0, -1], dim=0)    # último timestep → (num_actions,)

    k    = min(5, len(_id2action))
    top  = probs.topk(k)
    top5 = [(_id2action[i.item()], round(p.item(), 4))
            for i, p in zip(top.indices, top.values)]
    best_id = probs.argmax().item()
    return _id2action[best_id], round(probs[best_id].item(), 4), top5


class PredictHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"error": f"Unknown endpoint: {self.path}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send(400, {"error": "Empty body"})
            return

        try:
            png_bytes    = self.rfile.read(length)
            state_header = self.headers.get("X-Bot-State", "")
            action, confidence, top5 = predict_from_bytes(png_bytes, state_header)
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


def parse_args():
    p = argparse.ArgumentParser(description="IL Inference Server")
    p.add_argument("--model", default="src/il/models/minecraft_model.pth")
    p.add_argument("--port",  type=int, default=8765)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {_device}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    _extractor, _model, _id2action, seq_len = load_model(str(model_path), _device)
    print(f"Acciones ({len(_id2action)}): {list(_id2action.values())}")

    _feat_buffer  = deque(maxlen=seq_len)
    _state_buffer = deque(maxlen=seq_len)

    server = HTTPServer(("localhost", args.port), PredictHandler)
    print(f"\nInference server listo en http://localhost:{args.port}/predict")
    print("Esperando peticiones... (Ctrl+C para salir)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
