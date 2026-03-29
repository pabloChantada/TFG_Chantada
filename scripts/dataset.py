import argparse
import json
import os
import glob


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_discrete_action(action_value):
    """Normalize action from metrics to a single discrete label string."""
    if isinstance(action_value, str):
        return action_value or "idle"

    # Compatibility with old multidiscrete dict format
    if isinstance(action_value, dict):
        attack = int(action_value.get("attack", 0) or 0)
        equip = int(action_value.get("equip", 0) or 0)
        camera_yaw = int(action_value.get("camera_yaw", 0) or 0)
        camera_pitch = int(action_value.get("camera_pitch", 0) or 0)
        move_forward = int(action_value.get("move_forward", 0) or 0)
        move_backward = int(action_value.get("move_backward", 0) or 0)
        move_lateral = int(action_value.get("move_lateral", 0) or 0)
        move_vertical = int(action_value.get("move_vertical", 0) or 0)

        if attack == 1:
            return "attack"
        if equip == 4:
            return "equip_wooden_axe"

        if camera_yaw == 1:
            return "camera_yaw_p15"
        if camera_yaw == 2:
            return "camera_yaw_m15"
        if camera_yaw == 3:
            return "camera_yaw_p45"
        if camera_yaw == 4:
            return "camera_yaw_m45"

        if camera_pitch == 1:
            return "camera_pitch_p15"
        if camera_pitch == 2:
            return "camera_pitch_m15"
        if camera_pitch == 3:
            return "camera_pitch_p45"
        if camera_pitch == 4:
            return "camera_pitch_m45"

        if move_forward == 2:
            return "move_forward_sprint"
        if move_forward == 1:
            return "move_forward_walk"
        if move_backward == 1:
            return "move_backward_walk"
        if move_lateral == 1:
            return "move_left"
        if move_lateral == 2:
            return "move_right"
        if move_vertical == 1:
            return "jump"
        if move_vertical == 2:
            return "sneak"

        return "idle"

    return "idle"


def build_dataset(metrics_dir, output_jsonl):
    """Build training dataset from metrics JSON files.

    Reads from control_tracking.rl_actions.action_sequence and stores
    one discrete action label per timestep.
    """
    root_dir = "."
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

                action_label = _normalize_discrete_action(step.get("action", "idle"))
                position = step.get("position", {}) or {}
                camera = step.get("camera", {}) or {}

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
                    "action": action_label,
                }

                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                total += 1

    print(f"Dataset generado: {output_jsonl}")
    print(f"Ficheros usados: {used_files}/{len(metrics_files)}")
    print(f"Ejemplos: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir",  default="src/metrics/agent_metrics",
                        help="Directorio con los *_metrics.json")
    parser.add_argument("--output_jsonl", default="data/train.jsonl",
                        help="Ruta del JSONL de salida")
    args = parser.parse_args()

    build_dataset(args.metrics_dir, args.output_jsonl)