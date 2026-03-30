"""
Preparación del dataset JSONL para Imitation Learning.

El objetivo es hacer el dataset más representativo para que la red aprenda
patrones de comportamiento y no quede sobreajustada a secuencias repetitivas
o a valores concretos de píxeles.

Filtros (en orden de aplicación):
  1. Elimina clases con muy pocos ejemplos  (--min_samples, default 50)
  2. Elimina frames consecutivos idénticos  (--max_repeat,  default 3)
  3. Subsampling temporal uniforme          (--frame_skip,  default 1 = desactivado)

Uso:
  python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl
  python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl \\
      --min_samples 30 --max_repeat 5 --frame_skip 2
"""

import json
import argparse
from collections import Counter


def prepare(input_file, output_file, min_samples, max_repeat, frame_skip):
    print(f"[prepare_dataset] {input_file} → {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_lines = [l for l in f if l.strip()]

    total = len(raw_lines)
    print(f"  Entradas totales: {total}")

    # Parsear JSON (con aviso en caso de error)
    entries = []
    for i, line in enumerate(raw_lines):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [WARN] Línea {i} inválida, omitida: {e}")

    def get_action(e):
        a = e.get("action", "idle")
        if isinstance(a, dict):
            return a.get("name", "idle") or "idle"
        return str(a) if a else "idle"

    # ── 1. Filtro por clase mínima ────────────────────────────────────────────
    counts = Counter(get_action(e) for e in entries)
    print("\n  Distribución original:")
    for action, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {action:25s} {n}")

    valid = {a for a, n in counts.items() if n >= min_samples}
    removed = set(counts) - valid
    if removed:
        print(f"\n  Clases eliminadas (< {min_samples}): {sorted(removed)}")

    after_class = [e for e in entries if get_action(e) in valid]
    skipped_class = len(entries) - len(after_class)

    # ── 2. Eliminar frames consecutivos idénticos ─────────────────────────────
    after_repeat = []
    last_action, streak, skipped_repeat = None, 0, 0

    for e in after_class:
        a = get_action(e)
        streak = streak + 1 if a == last_action else 1
        last_action = a
        if streak <= max_repeat:
            after_repeat.append(e)
        else:
            skipped_repeat += 1

    # ── 3. Subsampling temporal ───────────────────────────────────────────────
    if frame_skip > 1:
        final = [e for i, e in enumerate(after_repeat) if i % frame_skip == 0]
        skipped_skip = len(after_repeat) - len(final)
    else:
        final = after_repeat
        skipped_skip = 0

    # ── Guardar ───────────────────────────────────────────────────────────────
    with open(output_file, 'w', encoding='utf-8') as f:
        for e in final:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    reduction = (1 - len(final) / total) * 100 if total else 0
    print(f"\n  --- Resumen ---")
    print(f"  Entradas originales:              {total}")
    print(f"  Eliminadas por clase minoritaria: {skipped_class}  (min_samples={min_samples})")
    print(f"  Eliminadas por frames repetidos:  {skipped_repeat}  (max_repeat={max_repeat})")
    print(f"  Eliminadas por frame skip:        {skipped_skip}  (frame_skip={frame_skip})")
    print(f"  Entradas finales:                 {len(final)}  (reducción {reduction:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparación de dataset JSONL para IL")
    parser.add_argument('--input',       required=True,  help='JSONL de entrada')
    parser.add_argument('--output',      required=True,  help='JSONL de salida')
    parser.add_argument('--min_samples', type=int, default=50,
                        help='Mínimo de muestras por clase (default: 50)')
    parser.add_argument('--max_repeat',  type=int, default=3,
                        help='Máx. frames consecutivos con la misma acción (default: 3)')
    parser.add_argument('--frame_skip',  type=int, default=1,
                        help='Subsampling temporal: 1=off, 3=1 de cada 3 (default: 1)')
    args = parser.parse_args()

    prepare(args.input, args.output, args.min_samples, args.max_repeat, args.frame_skip)
