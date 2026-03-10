"""
Dataset balancing and filtering for RL behavioral cloning.

Reads a raw train.jsonl and produces a balanced version by:
  1. Removing excessive runs of identical actions (e.g., 30 consecutive "attack-only" frames)
  2. Keeping all "interesting" transitions (movement, camera, crafting, etc.)
  3. Optionally upsampling rare actions to reduce class imbalance

Usage:
    python src/evaluation/balance_dataset.py                           # defaults
    python src/evaluation/balance_dataset.py --input data/train.jsonl --output data/train_balanced.jsonl
    python src/evaluation/balance_dataset.py --max_repeat 3 --upsample_rare
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


# ── Action dimension names (must match rl_action_tracker.js) ──────────────
ACTION_DIMS = [
    "move_forward", "move_backward", "move_lateral", "move_vertical",
    "camera_yaw", "camera_pitch", "attack",
    "craft", "smelt", "place", "equip",
]

# Indices of "interesting" action dimensions (non-trivial behaviour)
INTERESTING_DIMS = {
    "move_forward": [1, 2],      # walk, sprint
    "move_backward": [1],         # walk backwards
    "move_lateral": [1, 2],       # left, right
    "move_vertical": [1, 2],      # jump, sneak
    "camera_yaw": [1, 2, 3, 4],   # any camera movement
    "camera_pitch": [1, 2, 3, 4],
    "craft": [1, 2, 3, 4, 5, 6],  # any crafting
    "smelt": [1],                  # smelting
    "place": [1, 2, 3],           # placing blocks
    "equip": [1, 2, 3, 4],       # equipping items
}


def action_key(action_vector):
    """Hash an action vector for deduplication."""
    return tuple(action_vector)


def is_interesting(action_vector):
    """Check if an action has any non-trivial behaviour beyond just standing/attacking."""
    for i, dim_name in enumerate(ACTION_DIMS):
        if i >= len(action_vector):
            break
        if dim_name in INTERESTING_DIMS and action_vector[i] in INTERESTING_DIMS[dim_name]:
            # Movement, camera, crafting, etc.
            if dim_name != "attack":
                return True
    return False


def has_state_change(prev_state, curr_state, pos_threshold=0.5):
    """Check if the bot's position/orientation changed significantly."""
    if prev_state is None or curr_state is None:
        return True

    # Handle both new format (vector) and legacy format (x,y,z,yaw,pitch)
    def get_pos(s):
        if "vector" in s:
            v = s["vector"]
            return (v[0], v[1], v[2], v[3], v[4])
        return (
            s.get("x", 0), s.get("y", 0), s.get("z", 0),
            s.get("yaw", 0), s.get("pitch", 0),
        )

    prev = get_pos(prev_state)
    curr = get_pos(curr_state)

    # Position delta
    pos_delta = sum((a - b) ** 2 for a, b in zip(prev[:3], curr[:3])) ** 0.5
    # Orientation delta
    orient_delta = abs(prev[3] - curr[3]) + abs(prev[4] - curr[4])

    return pos_delta > pos_threshold or orient_delta > 0.05


def balance_dataset(input_path, output_path, max_repeat=3, upsample_rare=False,
                    rare_threshold_pct=2.0, upsample_factor=3, seed=42):
    """Balance the dataset by filtering repetitive frames and optionally upsampling rare actions.

    Args:
        input_path: Path to raw train.jsonl
        output_path: Path for balanced output
        max_repeat: Max consecutive identical actions to keep
        upsample_rare: Whether to duplicate rare action samples
        rare_threshold_pct: Actions below this % are considered rare
        upsample_factor: How many times to duplicate rare samples
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # ── Pass 1: Read and classify ──────────────────────────────────────────
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print("ERROR: No rows found in input file")
        return

    print(f"Raw dataset: {len(rows)} samples")

    # ── Pass 2: Subsample repetitive runs ──────────────────────────────────
    filtered = []
    run_count = 0
    prev_action = None
    prev_state = None

    for row in rows:
        action = row.get("action", [])
        state = row.get("state", {})
        ak = action_key(action)

        if ak == prev_action:
            run_count += 1
            # Keep if: still within max_repeat, OR state changed, OR it's an interesting action
            if run_count <= max_repeat:
                filtered.append(row)
            elif has_state_change(prev_state, state):
                filtered.append(row)
            # else: skip this repetitive frame
        else:
            run_count = 1
            filtered.append(row)

        prev_action = ak
        prev_state = state

    removed = len(rows) - len(filtered)
    print(f"After dedup (max_repeat={max_repeat}): {len(filtered)} samples ({removed} removed)")

    # ── Pass 3: Analyze action distribution ────────────────────────────────
    action_counts = Counter()
    interesting_rows = []
    boring_rows = []

    for row in filtered:
        action = row.get("action", [])
        ak = action_key(action)
        action_counts[ak] += 1

        if is_interesting(action):
            interesting_rows.append(row)
        else:
            boring_rows.append(row)

    print(f"\nAction distribution:")
    print(f"  Interesting frames (movement/camera/craft): {len(interesting_rows)}")
    print(f"  Static frames (attack-only/idle):           {len(boring_rows)}")

    # Show top 10 actions
    print(f"\n  Top 10 action patterns:")
    for ak, count in action_counts.most_common(10):
        pct = 100 * count / len(filtered)
        label = "★" if is_interesting(list(ak)) else " "
        print(f"    {label} {list(ak)} — {count} ({pct:.1f}%)")

    # ── Pass 4: Upsample rare actions (optional) ──────────────────────────
    final = list(filtered)

    if upsample_rare and len(filtered) > 0:
        threshold = rare_threshold_pct / 100.0 * len(filtered)
        rare_rows = []
        for row in filtered:
            ak = action_key(row.get("action", []))
            if action_counts[ak] < threshold and is_interesting(row.get("action", [])):
                rare_rows.append(row)

        if rare_rows:
            duplicated = rare_rows * (upsample_factor - 1)
            final.extend(duplicated)
            random.shuffle(final)
            print(f"\nUpsampled {len(rare_rows)} rare samples ×{upsample_factor} → +{len(duplicated)} rows")

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in final:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nBalanced dataset: {output_path}")
    print(f"Final samples: {len(final)}")

    # Summary stats
    final_interesting = sum(1 for r in final if is_interesting(r.get("action", [])))
    print(f"Interesting ratio: {100 * final_interesting / max(len(final), 1):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance RL training dataset")
    parser.add_argument("--input", default="data/train.jsonl",
                        help="Input JSONL file (raw)")
    parser.add_argument("--output", default="data/train_balanced.jsonl",
                        help="Output JSONL file (balanced)")
    parser.add_argument("--max_repeat", type=int, default=3,
                        help="Max consecutive identical actions to keep (default: 3)")
    parser.add_argument("--upsample_rare", action="store_true",
                        help="Upsample rare interesting actions")
    parser.add_argument("--rare_threshold_pct", type=float, default=2.0,
                        help="Actions below this %% are rare (default: 2.0)")
    parser.add_argument("--upsample_factor", type=int, default=3,
                        help="Duplication factor for rare samples (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    balance_dataset(
        input_path=args.input,
        output_path=args.output,
        max_repeat=args.max_repeat,
        upsample_rare=args.upsample_rare,
        rare_threshold_pct=args.rare_threshold_pct,
        upsample_factor=args.upsample_factor,
        seed=args.seed,
    )
