import argparse
import json
import os
import glob
import math
from collections import Counter


def _load_valid_items(jsonl_files, filter_actions):
    """Carga items válidos de los JSONL y calcula displacement por frame."""
    items = []
    skipped = 0

    for jf in jsonl_files:
        prev_x, prev_z = None, None
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  [WARN] {jf}:{lineno} — JSON error: {e}")
                        skipped += 1
                        continue

                    img_path = item.get("image")
                    action   = item.get("action")

                    if not img_path or not isinstance(action, str) or not action:
                        skipped += 1
                        continue
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    if filter_actions and action not in filter_actions:
                        skipped += 1
                        continue

                    # Displacement respecto al frame anterior (dentro del mismo fichero)
                    state = item.get("state") or {}
                    x = float(state.get("x", 0))
                    z = float(state.get("z", 0))
                    if prev_x is not None:
                        item["_displacement"] = math.hypot(x - prev_x, z - prev_z)
                    else:
                        item["_displacement"] = 0.0
                    prev_x, prev_z = x, z

                    items.append(item)
        except Exception as e:
            print(f"Saltando {jf}: {e}")

    return items, skipped


def build_dataset(recordings_dir, output_jsonl, filter_actions=None,
                  input_jsonl=None, max_attack_displacement=None):
    """Merge DatasetRecorder JSONL recordings into a single training file."""
    jsonl_files = [input_jsonl] if input_jsonl else sorted(
        glob.glob(os.path.join(recordings_dir, "*.jsonl"))
    )

    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    items, skipped = _load_valid_items(jsonl_files, filter_actions)

    # Filtrar frames de attack con displacement alto (etiqueta ruidosa)
    filtered_attack = 0
    if max_attack_displacement is not None:
        before = len(items)
        items = [it for it in items
                 if not (it["action"] == "attack"
                         and it["_displacement"] > max_attack_displacement)]
        filtered_attack = before - len(items)
        print(f"  Filtrados {filtered_attack} frames attack con displacement > {max_attack_displacement}")

    action_dist = Counter()
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for item in items:
            action = item["action"]
            entry = {
                "image":         item["image"],
                "state":         item.get("state", {}),
                "action":        action,
                "camera_delta":  item.get("camera_delta", {"dyaw": 0.0, "dpitch": 0.0}),
                "tree_visible":  item.get("tree_visible"),
                "tree_distance": item.get("tree_distance"),
            }
            if "session_id" in item:
                entry["session_id"] = item["session_id"]
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            action_dist[action] += 1

    print(f"Dataset generado : {output_jsonl}")
    print(f"Ficheros leídos  : {len(jsonl_files)}")
    print(f"Ejemplos válidos : {len(items)}  (saltados: {skipped}, attack filtrados: {filtered_attack})")
    print("Distribución de acciones:")
    for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        print(f"  {action:<25} {count:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge DatasetRecorder JSONL recordings into a training dataset."
    )
    parser.add_argument("--recordings_dir", default="data/recordings",
                        help="Directorio con los *.jsonl de grabación (default: data/recordings)")
    parser.add_argument("--output_jsonl", default="data/train.jsonl",
                        help="Ruta del JSONL de salida (default: data/train.jsonl)")
    parser.add_argument("--filter_actions", nargs="+", default=None, metavar="ACTION",
                        help="Mantener solo estas acciones")
    parser.add_argument("--input_jsonl", default=None,
                        help="Leer de un único JSONL en lugar de recordings_dir")
    parser.add_argument("--max_attack_displacement", type=float, default=None,
                        help="Eliminar frames 'attack' con displacement > este umbral (ej: 0.5)")
    args = parser.parse_args()

    build_dataset(
        recordings_dir=args.recordings_dir,
        output_jsonl=args.output_jsonl,
        filter_actions=set(args.filter_actions) if args.filter_actions else None,
        input_jsonl=args.input_jsonl,
        max_attack_displacement=args.max_attack_displacement,
    )