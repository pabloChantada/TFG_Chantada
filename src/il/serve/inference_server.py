#!/usr/bin/env python3
# --- bootstrap de rutas: modulos de IL organizados por rol ---
import sys as _sys
from pathlib import Path as _Path
_IL_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ('', 'models', 'data', 'train', 'serve', 'analysis'):
    _p = str(_IL_ROOT / _sub) if _sub else str(_IL_ROOT)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- fin bootstrap ---
"""
inference_server.py — Servidor HTTP de inferencia para modelos IL (LSTM / ConvLSTM).

Uso:
  python src/il/inference_server.py --model src/il/runs/<timestamp>/model_*.pth [--port 8765]

Endpoint:
  POST /predict
    Body   : PNG en bytes (image/png)
    Header : X-Bot-State: x,y,z,yaw,pitch,tree_visible,tree_distance
    Resp   : {"action": "attack", "confidence": 0.87, "camera_delta": {"dyaw": 0.1, "dpitch": -0.05}, "top5": [...]}

Auto-detecta el tipo de modelo (rnn/convlstm) desde config.json.
"""

import sys
import json
import math
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from collections import deque

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

IL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IL_DIR))

from model import RNNExtractor, ViTExtractor, MinecraftILModel
from model_convlstm import MinecraftConvLSTMModel
from constants import (IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEQ_LEN,
                       STATE_DIM, STATE_KEYS, STATE_BOUNDS,
                       CAMERA_BOUNDS, CAMERA_DIM)

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Globals ───────────────────────────────────────────────────────────────────
_extractor   = None   # None para convlstm
_model       = None
_model_type  = None   # "rnn" o "convlstm"
_id2action   = None
_device      = None
_seq_len     = None
_img_buffer  = None   # deque de tensors (C, H, W) — solo para convlstm
_feat_buffer = None   # deque de tensors (feat_dim,) — solo para rnn
_state_buffer = None  # deque de listas [x,y,z,yaw,pitch,tree_visible,tree_distance]


# ── Normalización (idéntica a load_dataset._norm) ─────────────────────────────

def _norm(val, key):
    lo, hi = STATE_BOUNDS[key]
    v = max(lo, min(hi, float(val)))
    return 2.0 * (v - lo) / (hi - lo) - 1.0 if hi != lo else 0.0


def _denorm_camera(norm_dyaw, norm_dpitch):
    """Convierte predicción [-1,1] a radianes."""
    dyaw_lo, dyaw_hi = CAMERA_BOUNDS["dyaw"]
    dp_lo, dp_hi     = CAMERA_BOUNDS["dpitch"]
    dyaw   = (norm_dyaw + 1.0) / 2.0 * (dyaw_hi - dyaw_lo) + dyaw_lo
    dpitch = (norm_dpitch + 1.0) / 2.0 * (dp_hi - dp_lo) + dp_lo
    return dyaw, dpitch


def _build_state_tensor(raw_states):
    """Construye (T, STATE_DIM) normalizado a partir de lista de raw states."""
    T_len = len(raw_states)
    states = torch.zeros(T_len, STATE_DIM)

    for t in range(T_len):
        r = raw_states[t]
        # x, z solo para derivar dx, dz; la posición absoluta no entra al vector.
        x, z, yaw, pitch, tv, td = r[0], r[2], r[3], r[4], r[5], r[6]

        if t > 0:
            dx = x - raw_states[t - 1][0]
            dz = z - raw_states[t - 1][2]
        else:
            dx = dz = 0.0

        vals = [yaw, pitch, dx, dz, tv, td]
        for j, (v, k) in enumerate(zip(vals, STATE_KEYS)):
            states[t, j] = _norm(v, k)

    return states


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    model_p   = Path(model_path)
    config_p  = model_p.parent / "config.json"
    pair2id_p = model_p.parent / (model_p.stem + "_pair2id.json")

    # Fallback: buscar pair2id con el patron *_pair2id.json
    if not pair2id_p.exists():
        candidates = list(model_p.parent.glob("*_pair2id.json"))
        if candidates:
            pair2id_p = candidates[0]

    if not config_p.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_p}")
    if not pair2id_p.exists():
        raise FileNotFoundError(f"pair2id no encontrado: {pair2id_p}")

    with open(config_p) as f:
        cfg = json.load(f)
    with open(pair2id_p) as f:
        pair2id = json.load(f)

    model_type = cfg.get("model_type", "rnn")
    seq_len    = cfg.get("seq_len", SEQ_LEN)
    id2action  = {int(v): k for k, v in pair2id.items()}

    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    if model_type == "convlstm":
        model = MinecraftConvLSTMModel(
            num_actions=cfg["num_actions"],
            state_dim=cfg.get("state_dim", STATE_DIM),
            camera_dim=cfg.get("camera_dim", CAMERA_DIM),
        )
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        extractor = None
        print(f"Modelo ConvLSTM cargado: num_actions={cfg['num_actions']}, seq_len={seq_len}")
    elif model_type == "state":
        # Solo-estado: sin extractor visual. El modelo recibe únicamente el vector
        # de estado (features=None en el forward); la imagen del visor se ignora.
        model = MinecraftILModel(
            num_actions=cfg["num_actions"],
            feat_dim=cfg["feat_dim"],
            state_dim=cfg.get("state_dim", STATE_DIM),
            lstm_hidden=cfg.get("lstm_hidden", 256),
            camera_dim=cfg.get("camera_dim", CAMERA_DIM),
        )
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        extractor = None
        print(f"Modelo solo-estado cargado (sin visión): num_actions={cfg['num_actions']}, seq_len={seq_len}")
    else:
        # rnn (GRU por filas) o vit: extractor visual + MinecraftILModel.
        if model_type == "vit":
            extractor = ViTExtractor(
                img_size=cfg.get("img_size", IMG_SIZE),
                feat_dim=cfg["feat_dim"],
                pretrained=False,   # los pesos se cargan del checkpoint
            )
        else:
            extractor = RNNExtractor(
                img_width=cfg.get("img_size", IMG_SIZE),
                hidden_size=cfg.get("hidden_size", 512),
            )
        model = MinecraftILModel(
            num_actions=cfg["num_actions"],
            feat_dim=cfg["feat_dim"],
            state_dim=cfg.get("state_dim", STATE_DIM),
            lstm_hidden=cfg.get("lstm_hidden", 256),
            camera_dim=cfg.get("camera_dim", CAMERA_DIM),
        )
        extractor.load_state_dict(ckpt["extractor"])
        model.load_state_dict(ckpt["model"])
        extractor = extractor.to(device).eval()
        model = model.to(device).eval()
        print(f"Modelo {model_type.upper()}+LSTM cargado: "
              f"num_actions={cfg['num_actions']}, seq_len={seq_len}")

    return extractor, model, model_type, id2action, seq_len


# ── Parsing del estado ────────────────────────────────────────────────────────

_BASE_STATE_LEN = 7  # x, y, z, yaw, pitch, tree_visible, tree_distance


def _parse_state(header_val: str) -> list:
    """Parsea el header X-Bot-State → lista de 7 floats."""
    try:
        vals = [float(v) for v in header_val.split(",")]
        if len(vals) >= _BASE_STATE_LEN:
            return vals[:_BASE_STATE_LEN]
        # Si solo envía 5 (x,y,z,yaw,pitch), rellenar tree con 0
        while len(vals) < _BASE_STATE_LEN:
            vals.append(0.0)
        return vals
    except Exception:
        return [0.0] * _BASE_STATE_LEN


# ── Predicción ────────────────────────────────────────────────────────────────

def predict_from_bytes(png_bytes: bytes, state_header: str):
    global _feat_buffer, _img_buffer, _state_buffer

    image  = Image.open(BytesIO(png_bytes)).convert("RGB")
    tensor = TRANSFORM(image)  # (C, H, W)
    state  = _parse_state(state_header)

    _state_buffer.append(state)

    # Padding del buffer si no está lleno
    states_list = list(_state_buffer)
    while len(states_list) < _seq_len:
        states_list = [states_list[0]] + states_list

    states_t = _build_state_tensor(states_list).unsqueeze(0).to(_device)  # (1, T, STATE_DIM)

    with torch.no_grad():
        if _model_type == "state":
            # Solo-estado: la imagen se ignora, el modelo usa solo el estado.
            logits, camera_pred = _model(None, states_t)
        elif _model_type == "convlstm":
            _img_buffer.append(tensor)
            imgs_list = list(_img_buffer)
            while len(imgs_list) < _seq_len:
                imgs_list = [imgs_list[0]] + imgs_list
            imgs_t = torch.stack(imgs_list).unsqueeze(0).to(_device)  # (1, T, C, H, W)
            logits, camera_pred = _model(imgs_t, states_t)
        else:
            feat = _extractor(tensor.unsqueeze(0).to(_device))[0]  # (feat_dim,)
            _feat_buffer.append(feat)
            feats_list = list(_feat_buffer)
            while len(feats_list) < _seq_len:
                feats_list = [feats_list[0]] + feats_list
            feats_t = torch.stack(feats_list).unsqueeze(0)  # (1, T, feat_dim)
            logits, camera_pred = _model(feats_t, states_t)

        # Ultimo timestep
        probs = F.softmax(logits[0, -1], dim=0)
        cam   = camera_pred[0, -1]  # (2,) normalizado [-1,1]

    dyaw, dpitch = _denorm_camera(cam[0].item(), cam[1].item())

    k    = min(5, len(_id2action))
    top  = probs.topk(k)
    top5 = [(_id2action[i.item()], round(p.item(), 4))
            for i, p in zip(top.indices, top.values)]

    best_id = probs.argmax().item()
    return {
        "action":       _id2action[best_id],
        "confidence":   round(probs[best_id].item(), 4),
        "camera_delta": {"dyaw": round(dyaw, 5), "dpitch": round(dpitch, 5)},
        "top5":         top5,
    }


# ── HTTP Handler ──────────────────────────────────────────────────────────────

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
            result       = predict_from_bytes(png_bytes, state_header)
            self._send(200, result)
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IL Inference Server (LSTM / ConvLSTM)")
    p.add_argument("--model", required=True,
                   help="Ruta al checkpoint .pth (con config.json y pair2id.json en el mismo dir)")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {_device}")

    _extractor, _model, _model_type, _id2action, _seq_len = load_model(args.model, _device)
    print(f"Acciones ({len(_id2action)}): {list(_id2action.values())}")

    _state_buffer = deque(maxlen=_seq_len)
    if _model_type == "convlstm":
        _img_buffer  = deque(maxlen=_seq_len)
    else:
        _feat_buffer = deque(maxlen=_seq_len)

    server = HTTPServer(("localhost", args.port), PredictHandler)
    print(f"\nInference server listo en http://localhost:{args.port}/predict")
    print(f"Modelo: {_model_type}  |  seq_len: {_seq_len}")
    print("Esperando peticiones... (Ctrl+C para salir)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
