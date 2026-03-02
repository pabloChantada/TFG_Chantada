import argparse
import json
import os
import glob
from collections import defaultdict


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_dataset(metrics_dir, output_jsonl, root_dir="."):
    # NEW SCHEMA: BotX_metrics.json
    metrics_files = sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.json")))

    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = 0
    used_files = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for mf in metrics_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Saltando {mf}: {e}")
                continue

            # NEW SCHEMA: control_tracking.control_sequence
            control_sequence = data.get("control_tracking", {}).get("control_sequence", [])
            if not isinstance(control_sequence, list) or not control_sequence:
                continue

            used_files += 1
            counts = defaultdict(int)

            for step in control_sequence:
                control_name = step.get("control")
                action_type = step.get("action")  # pressed/released
                screenshot = step.get("screenshot")

                # Ignore non-control snapshots
                if not control_name or not action_type or not screenshot:
                    continue

                action_name = f"{control_name}_{action_type}"

                position = step.get("position", {}) or {}
                camera = step.get("camera", {}) or {}

                state = {
                    "x": round(_to_float(position.get("x", 0.0)), 3),
                    "y": round(_to_float(position.get("y", 0.0)), 3),
                    "z": round(_to_float(position.get("z", 0.0)), 3),
                    "yaw": round(_to_float(camera.get("yaw", 0.0)), 4),
                    "pitch": round(_to_float(camera.get("pitch", 0.0)), 4),
                    "action_counts": dict(counts),
                }

                screenshot_path = os.path.normpath(screenshot)
                if not os.path.isabs(screenshot_path):
                    screenshot_path = os.path.normpath(os.path.join(root_dir, screenshot_path))

                example = {
                    "image": screenshot_path,
                    "state": state,
                    "action": action_name,
                }

                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1

                counts[action_name] += 1

    print(f"Dataset generado: {output_jsonl}")
    print(f"Ficheros usados: {used_files}/{len(metrics_files)}")
    print(f"Ejemplos: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="src/metrics/agent_metrics")
    parser.add_argument("--output_jsonl", default="data/train.jsonl")
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()

    build_dataset(args.metrics_dir, args.output_jsonl, args.root_dir)