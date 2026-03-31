#!/usr/bin/env python3
"""
generate_explainability.py — Genera plots de explicabilidad para el modelo IL.

Técnicas disponibles:
  gradcam  : GradCAM sobre layer4[-1] con gradientes del pipeline completo
  ig       : Integrated Gradients (pixel-level)
  state    : Importancia de variables de estado (x, y, z, yaw, pitch)

Modos de entrada:
  --images  : una o varias rutas de imagen directamente
  --jsonl   : toma N muestras aleatorias de un dataset .jsonl (con secuencia real)

Uso:
  # Una imagen concreta, todas las técnicas
  python generate_explainability.py --model src/il/models/model.pth --images frame.png

  # Varias imágenes, solo GradCAM + state
  python generate_explainability.py --model ... --images a.png b.png --techniques gradcam state

  # 10 muestras aleatorias del dataset con secuencia temporal real
  python generate_explainability.py --model ... --jsonl data/train.jsonl --n-samples 10

  # Estado del bot personalizado
  python generate_explainability.py --model ... --images frame.png --state 120.5,64,88.3,90,0
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IL_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT = IL_DIR.parents[1]
sys.path.insert(0, str(IL_DIR))

from model          import ResNetExtractor, MinecraftILModel
from constants      import IMAGENET_MEAN, IMAGENET_STD, SEQ_LEN, STATE_DIM
from explainability import (FullModelGradCAM, IntegratedGradients,
                            state_importance, plot_explainability)

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

ALL_TECHNIQUES = ("gradcam", "ig", "state")


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: torch.device):
    """Carga extractor + il_model + mapeo de acciones desde checkpoint."""
    config_p  = model_path.parent / (model_path.stem + "_config.json")
    pair2id_p = model_path.parent / (model_path.stem + "_pair2id.json")

    if not config_p.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_p}")
    if not pair2id_p.exists():
        raise FileNotFoundError(f"pair2id no encontrado: {pair2id_p}")

    with open(config_p)  as f: cfg     = json.load(f)
    with open(pair2id_p) as f: pair2id = json.load(f)

    extractor = ResNetExtractor(backbone=cfg["backbone"])
    il_model  = MinecraftILModel(
        num_actions=cfg["num_actions"],
        feat_dim=cfg["feat_dim"],
        state_dim=cfg.get("state_dim", STATE_DIM),
        lstm_hidden=cfg.get("lstm_hidden", 256),
    )

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    extractor.load_state_dict(ckpt["extractor"])
    il_model.load_state_dict(ckpt["model"])

    extractor = extractor.to(device).eval()
    il_model  = il_model.to(device).eval()

    id2action = {int(v): k for k, v in pair2id.items()}
    seq_len   = cfg.get("seq_len", SEQ_LEN)

    print(f"Modelo: backbone={cfg['backbone']}  acciones={cfg['num_actions']}  seq_len={seq_len}")
    return extractor, il_model, id2action, seq_len


# ---------------------------------------------------------------------------
# Construcción de buffers
# ---------------------------------------------------------------------------

def build_buffers_from_image(image_tensor, extractor, seq_len, state_vec, device):
    """
    Construye feat_buffer y state_buffer repitiendo el frame actual seq_len veces.
    Equivalente al modo de inferencia simple (sin historial real).
    """
    with torch.no_grad():
        feat = extractor(image_tensor)  # (1, feat_dim)

    # Repetir el mismo feature para los T-1 frames anteriores
    feat_buffer  = feat.unsqueeze(1).expand(1, seq_len - 1, -1).clone()  # (1, T-1, feat_dim)

    # Repetir el mismo estado para todos los T frames
    state_buffer = state_vec.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1).clone().to(device)

    return feat_buffer, state_buffer


def build_buffers_from_sequence(seq_tensors, seq_states, extractor, device):
    """
    Construye buffers a partir de una secuencia temporal real de frames.
    seq_tensors : lista de (1, C, H, W), longitud T
    seq_states  : lista de (STATE_DIM,),  longitud T
    """
    with torch.no_grad():
        feats = [extractor(t.to(device)) for t in seq_tensors]  # lista de (1, feat_dim)

    # feat_buffer = todos los frames menos el último → (1, T-1, feat_dim)
    feat_buffer = torch.stack(feats[:-1], dim=1)  # (1, T-1, feat_dim)

    # state_buffer = todos los estados → (1, T, STATE_DIM)
    state_buffer = torch.stack(seq_states, dim=0).unsqueeze(0).to(device)  # (1, T, STATE_DIM)

    return feat_buffer, state_buffer, seq_tensors[-1].to(device)


# ---------------------------------------------------------------------------
# Procesado de una muestra
# ---------------------------------------------------------------------------

def process_sample(extractor, il_model, id2action, device,
                   image_tensor, feat_buffer, state_buffer,
                   orig_image_np, label_hint, techniques, ig_steps,
                   output_dir, sample_name, show):
    """Ejecuta las técnicas seleccionadas y genera el plot para una muestra."""
    results = {}

    # -- GradCAM --
    if "gradcam" in techniques:
        cam = FullModelGradCAM(extractor, il_model)
        heatmap, pred_cls = cam.generate(image_tensor, feat_buffer, state_buffer)
        cam.release()
        results["gradcam"] = heatmap
        target_class = pred_cls
    else:
        # Necesitamos target_class de alguna manera
        with torch.no_grad():
            feat  = extractor(image_tensor)
            feats = torch.cat([feat_buffer, feat.unsqueeze(1)], dim=1)
            logits = il_model(feats, state_buffer)
            target_class = logits[0, -1].argmax().item()

    # -- Integrated Gradients --
    if "ig" in techniques:
        ig = IntegratedGradients(extractor, il_model, steps=ig_steps)
        attr, _ = ig.generate(image_tensor, feat_buffer, state_buffer,
                              target_class=target_class)
        results["ig"] = attr

    # -- State importance --
    if "state" in techniques:
        imp, _ = state_importance(extractor, il_model, image_tensor,
                                  feat_buffer, state_buffer,
                                  target_class=target_class)
        results["state"] = imp

    action_name = id2action.get(target_class, str(target_class))
    hint_str    = f"  (gt: {label_hint})" if label_hint else ""
    print(f"  Predicción: {action_name}{hint_str}")

    save_path = str(output_dir / f"{sample_name}.png")
    fig = plot_explainability(orig_image_np, results, action_name,
                              save_path=save_path)
    if not show:
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Modos de entrada
# ---------------------------------------------------------------------------

def run_from_images(args, extractor, il_model, id2action, seq_len, device, output_dir):
    """Modo --images: cada imagen se procesa de forma independiente."""
    state_vec = parse_state(args.state, device)

    for img_path in args.images:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[WARN] No encontrada: {img_path}")
            continue

        print(f"\n[{img_path.name}]")
        orig_np      = np.array(Image.open(img_path).convert("RGB"))
        image_tensor = TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        feat_buffer, state_buffer = build_buffers_from_image(
            image_tensor, extractor, seq_len, state_vec, device)

        process_sample(
            extractor, il_model, id2action, device,
            image_tensor, feat_buffer, state_buffer,
            orig_np, label_hint=None,
            techniques=args.techniques,
            ig_steps=args.ig_steps,
            output_dir=output_dir,
            sample_name=img_path.stem,
            show=args.show,
        )


def run_from_jsonl(args, extractor, il_model, id2action, seq_len, device, output_dir):
    """Modo --jsonl: toma N muestras con su secuencia temporal real."""
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL no encontrado: {jsonl_path}")

    # Agrupar frames por sesión manteniendo orden temporal
    sessions: dict[str, list[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if not item.get("image") or not Path(item["image"]).exists():
                    continue
                sid = item.get("session_id", "__unknown__")
                sessions.setdefault(sid, []).append(item)
            except Exception:
                continue

    # Recopilar posiciones válidas (necesitamos al menos seq_len frames en la sesión)
    candidates = []
    for sid, frames in sessions.items():
        for pos in range(len(frames)):
            candidates.append((sid, pos, frames))

    if not candidates:
        print("No se encontraron muestras válidas en el JSONL.")
        return

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.n_samples, len(candidates)))

    for i, (sid, pos, frames) in enumerate(selected):
        # Construir secuencia de T frames con padding al inicio si hace falta
        start     = max(0, pos - seq_len + 1)
        positions = list(range(start, pos + 1))
        while len(positions) < seq_len:
            positions = [positions[0]] + positions

        seq_items = [frames[p] for p in positions]

        seq_tensors, seq_states = [], []
        ok = True
        for item in seq_items:
            img_p = Path(item["image"])
            if not img_p.exists():
                ok = False
                break
            img = Image.open(img_p).convert("RGB")
            seq_tensors.append(TRANSFORM(img).unsqueeze(0))

            sv = item.get("state") or {}
            seq_states.append(torch.tensor([
                float(sv.get("x",     0.0)),
                float(sv.get("y",     0.0)),
                float(sv.get("z",     0.0)),
                float(sv.get("yaw",   0.0)),
                float(sv.get("pitch", 0.0)),
            ], dtype=torch.float32))

        if not ok:
            print(f"[WARN] Imagen faltante en sesión {sid}, pos {pos}")
            continue

        feat_buffer, state_buffer, image_tensor = build_buffers_from_sequence(
            seq_tensors, seq_states, extractor, device)

        orig_np     = np.array(Image.open(Path(seq_items[-1]["image"])).convert("RGB"))
        label_hint  = seq_items[-1].get("action", "")
        sample_name = f"sample_{i:03d}_{sid[:8]}_pos{pos}"

        print(f"\n[{i+1}/{len(selected)}] sesión={sid[:8]}  pos={pos}  gt={label_hint}")

        process_sample(
            extractor, il_model, id2action, device,
            image_tensor, feat_buffer, state_buffer,
            orig_np, label_hint=label_hint,
            techniques=args.techniques,
            ig_steps=args.ig_steps,
            output_dir=output_dir,
            sample_name=sample_name,
            show=args.show,
        )


# ---------------------------------------------------------------------------
# Utilidades CLI
# ---------------------------------------------------------------------------

def parse_state(state_str: str | None, device) -> torch.Tensor:
    """Parsea 'x,y,z,yaw,pitch' o devuelve zeros."""
    if state_str:
        try:
            vals = [float(v) for v in state_str.split(",")]
            if len(vals) == STATE_DIM:
                return torch.tensor(vals, dtype=torch.float32).to(device)
        except ValueError:
            pass
        print(f"[WARN] --state inválido: '{state_str}'. Usando zeros.")
    return torch.zeros(STATE_DIM, dtype=torch.float32, device=device)


def parse_args():
    p = argparse.ArgumentParser(
        description="Genera plots de explicabilidad para el modelo IL de Minecraft.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Modelo
    p.add_argument("--model", required=True,
                   help="Ruta al checkpoint .pth (necesita _config.json y _pair2id.json)")

    # Entrada (mutuamente exclusivos)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", metavar="IMG",
                     help="Una o varias rutas de imagen")
    src.add_argument("--jsonl", metavar="PATH",
                     help="Dataset .jsonl; usa secuencias temporales reales")

    # Opciones jsonl
    p.add_argument("--n-samples", type=int, default=5,
                   help="Nº de muestras a tomar del jsonl (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para selección aleatoria del jsonl (default: 42)")

    # Técnicas
    p.add_argument("--techniques", nargs="+", choices=ALL_TECHNIQUES,
                   default=list(ALL_TECHNIQUES),
                   help="Técnicas a aplicar (default: todas)")

    # Opciones de técnicas
    p.add_argument("--ig-steps", type=int, default=50,
                   help="Pasos de interpolación para Integrated Gradients (default: 50)")
    p.add_argument("--state", default=None,
                   help="Vector de estado manual 'x,y,z,yaw,pitch' (solo con --images)")

    # Salida
    p.add_argument("--output-dir", default=None,
                   help="Directorio de salida (default: directorio del modelo/explainability/)")
    p.add_argument("--show", action="store_true",
                   help="Mostrar plots interactivos además de guardarlos")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print(f"Error: modelo no encontrado: {model_path}")
        sys.exit(1)

    extractor, il_model, id2action, seq_len = load_model(model_path, device)

    # Directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando en: {output_dir}")
    print(f"Técnicas: {args.techniques}")

    if args.images:
        run_from_images(args, extractor, il_model, id2action, seq_len, device, output_dir)
    else:
        run_from_jsonl(args, extractor, il_model, id2action, seq_len, device, output_dir)

    print(f"\nListo. Plots guardados en: {output_dir}")


if __name__ == "__main__":
    main()
