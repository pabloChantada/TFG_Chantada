import argparse
import json
import os
import glob
from collections import defaultdict
from datetime import datetime

def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def build_dataset(metrics_dir, output_jsonl, root_dir="."):
    metrics_files = glob.glob(os.path.join(metrics_dir, "*_metrics.json"))
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    total = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mf in metrics_files:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            start_time = parse_iso(data["start_time"])
            counts = defaultdict(int)

            for step in data.get("world_model", []):
                action = step.get("action") or {}
                action_name = action.get("name")
                screenshot = step.get("screenshot")

                if not action_name or not screenshot:
                    continue

                # estado con contadores acumulados HASTA este paso
                state = {
                    # Redondeamos ya que los numeros siguientes no son relevantes (F3 trabaja con 3 decimales)
                    "x": round(step.get("x"), 3),
                    "y": round(step.get("y"), 3),
                    "z": round(step.get("z"), 3),
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
    parser.add_argument("--metrics_dir", required=True, help="Directorio con metrics_*.json")
    parser.add_argument("--output_jsonl", default="data/train.jsonl")
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()

    build_dataset(args.metrics_dir, args.output_jsonl, args.root_dir)