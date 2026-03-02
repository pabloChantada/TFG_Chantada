import argparse
import json
import os
import glob
from collections import defaultdict
from datetime import datetime

def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def build_dataset(metrics_dir, output_jsonl, root_dir="."):
    metrics_files = glob.glob(os.path.join(metrics_dir, "metrics_*.json"))
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    total = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mf in metrics_files:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            start_time = parse_iso(data["start_time"])
            counts = defaultdict(int)

            # Nueva estructura: control_tracking.control_sequence
            control_tracking = data.get("control_tracking", {})
            control_sequence = control_tracking.get("control_sequence", [])

            for step in control_sequence:
                # Nueva estructura: control + action (pressed/released)
                control_name = step.get("control")
                action_type = step.get("action")  # "pressed" o "released"
                screenshot = step.get("screenshot")

                if not control_name or not screenshot:
                    continue

                # Formar nombre de acción: control_action (ej: mine_pressed)
                action_name = f"{control_name}_{action_type}"

                # Obtener posición y cámara
                position = step.get("position", {})
                camera = step.get("camera", {})

                # estado con contadores acumulados HASTA este paso
                state = {
                    # Redondeamos ya que los numeros siguientes no son relevantes (F3 trabaja con 3 decimales)
                    "x": round(position.get("x", 0), 3),
                    "y": round(position.get("y", 0), 3),
                    "z": round(position.get("z", 0), 3),
                    "yaw": round(camera.get("yaw", 0), 4),
                    "pitch": round(camera.get("pitch", 0), 4),
                    "action_counts": dict(counts)
                }

                # normaliza la ruta a Windows y la hace relativa al root si es necesario
                screenshot_path = os.path.normpath(screenshot)
                if not os.path.isabs(screenshot_path):
                    screenshot_path = os.path.normpath(os.path.join(root_dir, screenshot_path))

                example = {
                    "image": screenshot_path,
                    "state": state,
                    "action": action_name
                }

                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1

                # actualiza contadores después de usar el estado
                counts[action_name] += 1

    print(f" Dataset generado: {output_jsonl}")
    print(f" Ejemplos: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=False, default="src/metrics/agent_metrics", help="Directorio con metrics_*.json")
    parser.add_argument("--output_jsonl", default="data/train.jsonl")
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()

    build_dataset(args.metrics_dir, args.output_jsonl, args.root_dir)