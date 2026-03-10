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


# Action dimension names (matches rl_action_tracker.js MultiDiscrete space)
ACTION_DIMS = [
    "move_forward",   # Discrete(3): 0=still, 1=walk, 2=sprint
    "move_backward",  # Discrete(2): 0=still, 1=walk
    "move_lateral",   # Discrete(3): 0=still, 1=left, 2=right
    "move_vertical",  # Discrete(3): 0=still, 1=jump, 2=sneak
    "camera_yaw",     # Discrete(5): 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    "camera_pitch",   # Discrete(5): 0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
    "attack",         # Discrete(2): 0=no, 1=yes
    "craft",          # Discrete(7): 0=none..6=iron_pickaxe
    "smelt",          # Discrete(2): 0=none, 1=iron_ingot
    "place",          # Discrete(4): 0=none..3=torch
    "equip",          # Discrete(5): 0=none..4=axe
]


def build_dataset(metrics_dir, output_jsonl, root_dir="."):
    """Build training dataset from metrics JSON files.

    Reads from control_tracking.rl_actions.action_sequence which contains
    the MultiDiscrete action vector + screenshot for each timestep.
    """
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

            # Read from rl_actions.action_sequence
            rl_actions = data.get("control_tracking", {}).get("rl_actions", {})
            action_sequence = rl_actions.get("action_sequence", [])
            if not isinstance(action_sequence, list) or not action_sequence:
                continue

            used_files += 1

            for step in action_sequence:
                screenshot = step.get("screenshot")

                # Skip steps without screenshot (no visual observation)
                if not screenshot:
                    continue

                action_dict = step.get("action", {})
                position = step.get("position", {}) or {}
                camera = step.get("camera", {}) or {}

                # Build action vector (MultiDiscrete)
                action_vector = [int(action_dict.get(dim, 0)) for dim in ACTION_DIMS]

                # Build state — prefer full 13-dim vector when available
                state_vector = step.get("state_vector")
                if isinstance(state_vector, list) and len(state_vector) == 13:
                    # Full state from extractState() — already normalized
                    state = {"vector": [round(float(v), 4) for v in state_vector]}
                else:
                    # Legacy fallback: only position + camera
                    state = {
                        "x": round(_to_float(position.get("x", 0.0)), 3),
                        "y": round(_to_float(position.get("y", 0.0)), 3),
                        "z": round(_to_float(position.get("z", 0.0)), 3),
                        "yaw": round(_to_float(camera.get("yaw", 0.0)), 4),
                        "pitch": round(_to_float(camera.get("pitch", 0.0)), 4),
                    }

                screenshot_path = os.path.normpath(screenshot)
                if not os.path.isabs(screenshot_path):
                    screenshot_path = os.path.normpath(os.path.join(root_dir, screenshot_path))

                example = {
                    "image": screenshot_path,
                    "state": state,
                    "action": action_vector,
                }

                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1

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