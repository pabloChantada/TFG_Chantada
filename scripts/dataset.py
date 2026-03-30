import argparse
import json
import os
import glob
from collections import Counter


def build_dataset(recordings_dir, output_jsonl, filter_actions=None, input_jsonl=None):
    """Merge DatasetRecorder JSONL recordings into a single training file.

    Each recording is a *.jsonl where every line has:
      image, state, action, (optional) aux

    Only lines where the image file exists and action is a non-empty string
    are written to the output. The aux field is forwarded as-is when present.

    Args:
        recordings_dir:  Directory containing *.jsonl recording files.
        output_jsonl:    Output path for the merged dataset.
        filter_actions:  If set, only keep entries whose action is in this set.
        input_jsonl:     If set, read from this single file instead of recordings_dir.
    """
    if input_jsonl:
        jsonl_files = [input_jsonl]
    else:
        jsonl_files = sorted(glob.glob(os.path.join(recordings_dir, "*.jsonl")))

    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total       = 0
    skipped     = 0
    used_files  = 0
    action_dist = Counter()

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for jf in jsonl_files:
            file_count = 0
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
                        tree_visible = item.get("tree_visible")
                        tree_distance = item.get("tree_distance")

                        if not img_path or not isinstance(action, str) or not action:
                            skipped += 1
                            continue

                        if not os.path.exists(img_path):
                            skipped += 1
                            continue

                        if filter_actions and action not in filter_actions:
                            skipped += 1
                            continue

                        entry = {
                            "image":  img_path,
                            "state":  item.get("state", {}),
                            "action": action,
                            "tree_visible": tree_visible,
                            "tree_distance": tree_distance,
                        }

                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        action_dist[action] += 1
                        file_count += 1
                        total += 1

            except Exception as e:
                print(f"Saltando {jf}: {e}")
                continue

            if file_count > 0:
                used_files += 1

    print(f"Dataset generado : {output_jsonl}")
    print(f"Ficheros usados  : {used_files}/{len(jsonl_files)}")
    print(f"Ejemplos válidos : {total}  (saltados: {skipped})")
    print(f"Distribución de acciones:")
    for action, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        print(f"  {action:<25} {count:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge DatasetRecorder JSONL recordings into a training dataset."
    )
    parser.add_argument(
        "--recordings_dir", default="data/recordings",
        help="Directory containing *.jsonl recording files (default: data/recordings)"
    )
    parser.add_argument(
        "--output_jsonl", default="data/train.jsonl",
        help="Output path for the merged JSONL (default: data/train.jsonl)"
    )
    parser.add_argument(
        "--filter_actions", nargs="+", default=None,
        metavar="ACTION",
        help="Only keep entries with these action labels (e.g. idle attack jump move_forward_sprint)"
    )
    parser.add_argument(
        "--input_jsonl", default=None,
        help="Read from a single JSONL file instead of scanning recordings_dir"
    )
    args = parser.parse_args()

    build_dataset(
        recordings_dir=args.recordings_dir,
        output_jsonl=args.output_jsonl,
        filter_actions=set(args.filter_actions) if args.filter_actions else None,
        input_jsonl=args.input_jsonl,
    )
