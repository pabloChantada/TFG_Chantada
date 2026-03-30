#!/usr/bin/env python3
"""
test_model.py — Prueba el modelo de Imitation Learning en tres modos:

  --mode manual  : captura pantalla y muestra la predicción (sin ejecutar)
  --mode bot     : loop continuo que predice y ejecuta las acciones via teclado/ratón (pyautogui)
  --mode server  : loop continuo que predice y ejecuta las acciones via HTTP al servidor RL
                   (requiere: node src/rl/server.js corriendo)

Uso:
  python scripts/test_model.py --model src/il/models/minecraft_model.pth --mode manual
  python scripts/test_model.py --model src/il/models/minecraft_model.pth --mode bot --region 0 0 854 480
  python scripts/test_model.py --model src/il/models/minecraft_model.pth --mode server --server http://localhost:3001

Dependencias bot:
  pip install pyautogui
Dependencias server:
  pip install requests
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import mss
import mss.tools

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IL_DIR = PROJECT_ROOT / "src" / "il"
sys.path.insert(0, str(IL_DIR))

from model import MinecraftILModel
from constants import IMAGENET_MEAN, IMAGENET_STD

# ── Transform (idéntico al val/inference del dataset, sin augmentación) ───────
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Duración de tecla pulsada en modo bot ─────────────────────────────────────
HOLD_DURATION = 0.06  # segundos


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_mapping(model_path: str, device: torch.device):
    """
    Carga state_dict, infiere backbone y num_aux, devuelve (model, id2action).

    Búsqueda de pair2id (orden de clases):
      1. <model_dir>/pair2id.json   — guardado automáticamente por main.py
      2. Fallback: orden de ACTIONS (constants.py)
    """
    state = torch.load(model_path, map_location=device, weights_only=True)

    # Detectar formato del checkpoint:
    #   head.1.weight → arquitectura actual (Dropout + Linear en self.head)
    #   head.0.weight → arquitectura actual sin Dropout
    #   model.fc.weight → formato antiguo (fc era el clasificador directamente)
    head_w = state.get("head.1.weight")
    legacy = head_w is None and "model.fc.weight" in state

    if legacy:
        # Checkpoint antiguo: convertir model.fc → head.1 para poder cargarlo
        head_w = state["model.fc.weight"]
        print("[WARN] Checkpoint antiguo (model.fc). Convirtiendo para compatibilidad.")
        state["head.1.weight"] = state.pop("model.fc.weight")
        state["head.1.bias"]   = state.pop("model.fc.bias")

    if head_w is None:
        available = [k for k in state if "weight" in k]
        raise ValueError(
            f"No se encontró la clave de la cabeza ('head.1.weight' ni 'head.0.weight').\n"
            f"Claves con 'weight' disponibles: {available}"
        )

    num_actions = head_w.shape[0]
    linear_in   = head_w.shape[1]

    # Detectar backbone por dimensión de features (512 = r18/r34, 2048 = r50/r101)
    backbone_map = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}
    backbone, num_aux = None, None
    for bb, feat_dim in backbone_map.items():
        candidate_aux = linear_in - feat_dim
        if candidate_aux >= 0:
            try:
                m = MinecraftILModel(num_actions=num_actions, backbone=bb, num_aux=candidate_aux)
                m.load_state_dict(state)
                backbone, num_aux = bb, candidate_aux
                break
            except Exception:
                continue

    if backbone is None:
        raise RuntimeError("No se pudo determinar el backbone del modelo.")

    model = MinecraftILModel(num_actions=num_actions, backbone=backbone, num_aux=num_aux)
    model.load_state_dict(state)
    model = model.to(device).eval()

    print(f"Modelo: backbone={backbone}, num_actions={num_actions}, num_aux={num_aux}")

    # Intentar cargar pair2id desde JSON compañero (<modelo>_pair2id.json)
    pair2id_path = Path(model_path).parent / (Path(model_path).stem + "_pair2id.json")
    if pair2id_path.exists():
        with open(pair2id_path) as f:
            pair2id = json.load(f)
        print(f"Mapping cargado desde {pair2id_path}")
    else:
        raise FileNotFoundError(
            f"No se encontró el mapping de clases '{pair2id_path.name}'.\n"
            f"Los modelos entrenados con main.py o sweep.py generan este archivo automáticamente.\n"
            f"Si tienes un checkpoint antiguo, genera el mapping con:\n"
            f"  python scripts/save_pair2id.py --dataset data/train_clean.jsonl --model {model_path}"
        )

    id2action = {v: k for k, v in pair2id.items()}
    return model, id2action


# ─────────────────────────────────────────────────────────────────────────────
# Captura de pantalla
# ─────────────────────────────────────────────────────────────────────────────

def capture_screen(region=None) -> Image.Image:
    """
    Captura la pantalla completa o una región y devuelve PIL.Image RGB.

    region: (x, y, width, height) o None para pantalla completa.
    """
    with mss.mss() as sct:
        if region:
            mon = {"left": region[0], "top": region[1],
                   "width": region[2], "height": region[3]}
        else:
            mon = sct.monitors[1]  # monitor principal
        shot = sct.grab(mon)
    # mss devuelve BGRA; convertimos a RGB
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, image: Image.Image, device, id2action: dict):
    """Devuelve (best_action, best_conf, top5_list)."""
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)  # aux=None → zeros (graceful degradation)
        probs  = torch.softmax(logits, dim=1)[0]

    k = min(5, len(id2action))
    top = probs.topk(k)
    top5 = [(id2action[i.item()], p.item()) for i, p in zip(top.indices, top.values)]
    best_id = probs.argmax().item()
    return id2action[best_id], probs[best_id].item(), top5


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de acciones (modo bot)
# ─────────────────────────────────────────────────────────────────────────────

def _press(key, ctrl=False):
    import pyautogui
    if ctrl:
        pyautogui.keyDown("ctrl")
    pyautogui.keyDown(key)
    time.sleep(HOLD_DURATION)
    pyautogui.keyUp(key)
    if ctrl:
        pyautogui.keyUp("ctrl")


def _mouse_rel(dx, dy):
    import pyautogui
    pyautogui.moveRel(dx, dy, duration=0)


def _click():
    import pyautogui
    pyautogui.click(button="left")


# Controles por defecto Minecraft Java Edition
ACTION_EXEC = {
    "idle":                lambda: None,
    "move_forward_walk":   lambda: _press("w"),
    "move_forward_sprint": lambda: _press("w", ctrl=True),   # Ctrl = sprint
    "move_backward_walk":  lambda: _press("s"),
    "move_left":           lambda: _press("a"),
    "move_right":          lambda: _press("d"),
    "jump":                lambda: _press("space"),
    "sneak":               lambda: _press("shift"),
    "camera_yaw_p15":      lambda: _mouse_rel(15, 0),
    "camera_yaw_m15":      lambda: _mouse_rel(-15, 0),
    "camera_yaw_p45":      lambda: _mouse_rel(45, 0),
    "camera_yaw_m45":      lambda: _mouse_rel(-45, 0),
    "camera_pitch_p15":    lambda: _mouse_rel(0, 15),
    "camera_pitch_m15":    lambda: _mouse_rel(0, -15),
    "camera_pitch_p45":    lambda: _mouse_rel(0, 45),
    "camera_pitch_m45":    lambda: _mouse_rel(0, -45),
    "attack":              lambda: _click(),
    "equip_wooden_axe":    lambda: _press("1"),  # slot 1 del hotbar
}


def execute_action(action_name: str):
    fn = ACTION_EXEC.get(action_name)
    if fn:
        fn()
    else:
        print(f"[WARN] Acción desconocida: {action_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Modo manual
# ─────────────────────────────────────────────────────────────────────────────

def run_manual(model, device, id2action, region, interval):
    """Captura pantalla y muestra predicción. No ejecuta nada."""
    print(f"MODO MANUAL — captura cada {interval:.2f}s  |  Ctrl+C para salir\n")
    try:
        n = 0
        while True:
            img = capture_screen(region)
            action, conf, top5 = predict(model, img, device, id2action)
            n += 1

            os.system("cls" if os.name == "nt" else "clear")
            print(f"  Frame #{n}   [{time.strftime('%H:%M:%S')}]\n")
            print(f"  {'Acción':<26}  {'Conf':>6}")
            print(f"  {'-'*34}")
            for name, p in top5:
                marker = " ◄" if name == action else "  "
                print(f"  {name:<26}  {p:>5.1%}{marker}")
            print()
            print(f"  Predicción: {action}  ({conf:.1%})")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nSaliendo.")


# ─────────────────────────────────────────────────────────────────────────────
# Modo bot
# ─────────────────────────────────────────────────────────────────────────────

def run_bot(model, device, id2action, region, interval):
    """Loop: captura → predice → ejecuta acción en el juego."""
    try:
        import pyautogui
    except ImportError:
        print("ERROR: pyautogui no instalado. Ejecuta: pip install pyautogui")
        sys.exit(1)

    pyautogui.FAILSAFE = True  # mover ratón a esquina sup-izq para parar

    print("MODO BOT")
    print("  • El modelo controla el juego vía teclado/ratón")
    print("  • Para PARAR: mueve el ratón a la esquina SUPERIOR IZQUIERDA")
    print("  • O pulsa Ctrl+C en esta terminal\n")
    print("Empezando en 5 segundos — pon el foco en la ventana de Minecraft...\n")
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  ¡Arrancando!\n")

    try:
        frame = 0
        while True:
            t0 = time.time()

            img = capture_screen(region)
            action, conf, top5 = predict(model, img, device, id2action)
            execute_action(action)

            frame += 1
            elapsed = time.time() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"\r  [#{frame:05d}] {action:<26}  {conf:>5.1%}  {fps:.1f} fps   ", end="", flush=True)

            remaining = interval - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\nBot detenido.")
    except pyautogui.FailSafeException:
        print("\n\nEmergencia: ratón en esquina — bot detenido.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Test del modelo IL de Minecraft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode",     choices=["manual", "bot"], default="manual",
                   help="manual: sólo predice | bot: predice y ejecuta (default: manual)")
    p.add_argument("--model",    default="src/il/models/minecraft_model.pth",
                   help="Ruta al .pth del modelo entrenado")
    p.add_argument("--region",   nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="Región de captura en píxeles: X Y Ancho Alto "
                        "(omitir = pantalla completa del monitor principal)")
    p.add_argument("--interval", type=float, default=0.1,
                   help="Segundos entre capturas / acciones (default: 0.1 → ~10 fps)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print(f"Error: modelo no encontrado en {model_path}")
        sys.exit(1)

    model, id2action = load_model_and_mapping(str(model_path), device)

    print(f"\nAcciones disponibles ({len(id2action)}):")
    for i, name in sorted(id2action.items()):
        print(f"  {i:2d}: {name}")
    print()

    region = tuple(args.region) if args.region else None

    if args.mode == "manual":
        run_manual(model, device, id2action, region, args.interval)
    else:
        run_bot(model, device, id2action, region, args.interval)
