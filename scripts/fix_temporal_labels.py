"""
Fix temporal label misalignment in the JSONL dataset.

The original dataset labels each frame with the action that *produced* it.
The correct labeling is: frame[t].action = action that will be taken *from* that view,
i.e. action[t+1]. This shift is critical for camera_yaw classes because two
consecutive frames look nearly identical — without the shift the model can't
distinguish "I just turned" from "I'm about to turn".

Fix applied per recording session independently:
  frame[t].action  ←  original frame[t+1].action
  last frame of each session is dropped (no next action exists)

Usage:
  python scripts/fix_temporal_labels.py --input data/train_clean.jsonl --output data/train_aligned.jsonl
  python scripts/fix_temporal_labels.py --input data/train_clean.jsonl --output data/train_aligned.jsonl --dry-run
  python scripts/fix_temporal_labels.py --input data/train_clean.jsonl --output data/train_aligned.jsonl --validate
"""

import json
import argparse
import os
import random
from collections import Counter


# ---------------------------------------------------------------------------
# Session boundary detection
# ---------------------------------------------------------------------------

def get_session_key(entry: dict) -> str:
    """
    Derive a session identifier from the image path.
    Uses the immediate parent directory name, which encodes the recording ID
    (e.g. 'Rec_10_5h8clv_screenshots').
    Falls back to the full path if the directory cannot be extracted.
    """
    image_path = entry.get("image", "")
    # Normalise separators
    normalised = image_path.replace("\\", "/")
    parts = normalised.split("/")
    # The screenshot directory is the second-to-last component
    if len(parts) >= 2:
        return parts[-2]
    return normalised


# ---------------------------------------------------------------------------
# Core fix
# ---------------------------------------------------------------------------

def split_into_sessions(entries: list) -> dict:
    """Group entries by session key, preserving insertion order."""
    sessions: dict[str, list] = {}
    for entry in entries:
        key = get_session_key(entry)
        sessions.setdefault(key, []).append(entry)
    return sessions


def shift_labels(session_entries: list) -> list:
    """
    For a single session, replace action[t] with action[t+1].
    Also computes state-delta aux features between frame t and t+1:
      - yaw_delta: yaw change in radians, wrapped to [-pi, pi]
      - speed:     horizontal displacement (blocks)
      - dy:        vertical displacement (blocks)
    Drops the last frame (no future action available).
    Returns the corrected entries (length = len(session_entries) - 1).
    """
    import math
    corrected = []
    for t in range(len(session_entries) - 1):
        entry = dict(session_entries[t])
        entry["action"] = session_entries[t + 1]["action"]

        s0 = session_entries[t].get("state") or {}
        s1 = session_entries[t + 1].get("state") or {}
        if s0 and s1:
            raw_dyaw = s1.get("yaw", 0.0) - s0.get("yaw", 0.0)
            dyaw = (raw_dyaw + math.pi) % (2 * math.pi) - math.pi
            dx = s1.get("x", 0.0) - s0.get("x", 0.0)
            dz = s1.get("z", 0.0) - s0.get("z", 0.0)
            dy = s1.get("y", 0.0) - s0.get("y", 0.0)
            entry["aux"] = {
                "yaw_delta": round(dyaw, 5),
                "speed":     round(math.sqrt(dx * dx + dz * dz), 5),
                "dy":        round(dy, 5),
            }

        corrected.append(entry)
    return corrected


def fix_dataset(entries: list) -> tuple[list, dict]:
    """
    Apply temporal label shift to all sessions.

    Returns:
        corrected_entries: list of fixed entries
        info: dict with per-session stats
    """
    sessions = split_into_sessions(entries)
    corrected_entries = []
    info = {}

    for session_key, session_entries in sessions.items():
        fixed = shift_labels(session_entries)
        corrected_entries.extend(fixed)
        info[session_key] = {
            "before": len(session_entries),
            "after": len(fixed),
            "dropped": len(session_entries) - len(fixed),
        }

    return corrected_entries, info


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i} is invalid JSON, skipping: {e}")
    return entries


def save_jsonl(path: str, entries: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def label_distribution(entries: list) -> Counter:
    return Counter(e.get("action", "idle") for e in entries)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(original: list, corrected: list, session_info: dict, dry_run: bool) -> None:
    total_before = len(original)
    total_after = len(corrected)
    total_dropped = total_before - total_after

    print("\n" + "=" * 60)
    print("  Temporal label alignment — summary")
    print("=" * 60)
    if dry_run:
        print("  *** DRY RUN — no files written ***")
    print(f"\n  Sessions found: {len(session_info)}")
    for key, s in session_info.items():
        print(f"    {key}: {s['before']} frames -> {s['after']} (dropped {s['dropped']})")

    print(f"\n  Total frames before : {total_before}")
    print(f"  Total frames dropped: {total_dropped}")
    print(f"  Total frames after  : {total_after}")

    dist_before = label_distribution(original)
    dist_after = label_distribution(corrected)
    all_labels = sorted(set(dist_before) | set(dist_after))

    print(f"\n  {'Label':<25} {'Before':>8}  {'After':>8}  {'Delta':>6}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*6}")
    for label in all_labels:
        b = dist_before.get(label, 0)
        a = dist_after.get(label, 0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<25} {b:>8}  {a:>8}  {sign}{delta:>5}")
    print()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(original_path: str, corrected_path: str, n: int = 10) -> None:
    """
    Load both files, sample n random frames (by global index in original),
    and print original_label -> corrected_label so you can verify the shift.
    """
    original = load_jsonl(original_path)
    corrected = load_jsonl(corrected_path)

    # Build a mapping from image path -> corrected action for quick lookup
    corrected_map = {e["image"]: e["action"] for e in corrected}

    # We can only validate frames that appear in both files (i.e. not the dropped last frames)
    eligible = [(i, e) for i, e in enumerate(original) if e["image"] in corrected_map]

    if not eligible:
        print("[validate] No matching frames found between original and corrected file.")
        return

    sample = random.sample(eligible, min(n, len(eligible)))
    sample.sort(key=lambda x: x[0])

    print(f"\n{'=' * 60}")
    print(f"  Validation sample ({len(sample)} frames)")
    print(f"{'=' * 60}")
    print(f"  {'[idx]':<8} {'Original label':<25} -> Corrected label")
    print(f"  {'-'*8} {'-'*25}   {'-'*20}")
    for idx, entry in sample:
        orig_label = entry["action"]
        corr_label = corrected_map[entry["image"]]
        changed = " *" if orig_label != corr_label else " (same)"
        print(f"  [{idx:<5}]  {orig_label:<25} -> {corr_label}{changed}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix temporal label misalignment in a JSONL IL dataset."
    )
    parser.add_argument("--input",    required=True,  help="Input JSONL file")
    parser.add_argument("--output",   required=True,  help="Output JSONL file (corrected)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would change without writing anything")
    parser.add_argument("--validate", action="store_true",
                        help="After writing, sample 10 frames to verify the shift")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for the validation sample (default: 42)")
    args = parser.parse_args()

    if not args.dry_run and args.input == args.output:
        parser.error("--input and --output must be different paths to avoid overwriting the original.")

    print(f"[fix_temporal_labels] Loading: {args.input}")
    original = load_jsonl(args.input)
    print(f"  Loaded {len(original)} entries")

    corrected, session_info = fix_dataset(original)

    print_summary(original, corrected, session_info, dry_run=args.dry_run)

    if not args.dry_run:
        save_jsonl(args.output, corrected)
        print(f"  Saved corrected dataset -> {args.output}")

        if args.validate:
            random.seed(args.seed)
            validate(args.input, args.output)
    elif args.validate:
        print("  [validate] Skipped (dry-run mode — no output file written)")


if __name__ == "__main__":
    main()
