#!/usr/bin/env python3
"""
action_dist.py — Distribución de acciones en inferencia por modelo (IL y RL).

Lee uno o varios `eval*.jsonl` (esquema de episode_metrics.js / eval_policy.py, con
el campo `action_counts` por episodio) y agrega el conteo de acciones de TODOS los
episodios de cada fichero. Sirve para ver qué política ejecuta cada modelo en el
bucle real y detectar el colapso a la clase mayoritaria (p.ej. el ConvLSTM
concentrando casi todo en `attack`), algo que la accuracy por frame esconde.

Funciona igual para IL y RL: ambos escriben `action_counts` en cada registro
(el ILAgent en modo eval; eval_policy.py para SAC/PPO/DQN). En IL/RL discreto hay
exactamente una acción por paso, así que los porcentajes son sobre pasos; en el RL
`hybrid` (multi-flag) son sobre activaciones de flag.

Uso:
  python src/evaluation/action_dist.py \
      data/eval/il/eval.jsonl:GRU \
      data/eval/il/eval_conv.jsonl:ConvLSTM \
      data/eval/rl/eval.jsonl:SAC \
      --latex --out modelo-tfg-fic-v1.6_2223xun/imaxes/action_dist.png

Cada argumento es `ruta[:Etiqueta]`. Si se omite la etiqueta se usa el nombre del
fichero. --latex imprime una tabla; --out PATH guarda el histograma comparado.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
})


def load_counts(path: Path) -> tuple[Counter, int]:
    """Suma action_counts de todos los episodios. Devuelve (counter, n_episodios)."""
    total = Counter()
    n_eps = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_eps += 1
            ac = rec.get("action_counts")
            if ac:
                total.update(ac)
    return total, n_eps


def load_reference(path: Path) -> Counter:
    """Distribución de etiquetas del experto: cuenta el campo `action` de cada
    frame de un dataset de demostraciones (p.ej. data/train.jsonl). Es la forma
    correcta de situar al HTN en este gráfico, ya que en el entorno no ejecuta
    acciones discretas, sino que su comportamiento se discretizó al grabar."""
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line).get("action")
            if a:
                c[a] += 1
    return c


def parse_arg(spec: str) -> tuple[Path, str]:
    # admite rutas de Windows con ':' (C:\...): la etiqueta va tras el ÚLTIMO ':'
    # solo si lo que sigue no parece una ruta (no acaba en separador).
    if ":" in spec[2:]:  # ignora el "C:" inicial de rutas absolutas
        head, _, label = spec.rpartition(":")
        if head and not head.endswith(("\\", "/")):
            return Path(head), label
    p = Path(spec)
    return p, p.stem


def plot_histogram(results, actions, out_path: Path):
    """Histograma de barras agrupadas: una barra por modelo en cada acción (%)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(results)
    x = np.arange(len(actions))
    width = 0.8 / max(1, n_models)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(actions)), 4.2))
    for j, (label, counts, _n_eps, steps) in enumerate(results):
        pcts = [100 * counts.get(a, 0) / steps if steps else 0.0 for a in actions]
        ax.bar(x + j * width - 0.4 + width / 2, pcts, width, label=label)

    ax.set_ylabel("% de pasos")
    ax.set_xlabel("Acción")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in actions], fontsize=8)
    ax.set_title("Distribución de acciones en inferencia")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nHistograma guardado en: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="ruta[:Etiqueta] de cada eval.jsonl")
    ap.add_argument("--latex", action="store_true", help="imprime una tabla LaTeX")
    ap.add_argument("--out", type=str, default=None,
                    help="ruta del PNG del histograma comparado (opcional)")
    ap.add_argument("--reference", type=str, default=None,
                    help="dataset jsonl[:Etiqueta] cuya distribución de `action` se añade "
                         "como referencia del experto (p.ej. data/train.jsonl:HTN). Aparece "
                         "como primera columna/barra, restringida a las acciones de los modelos.")
    args = ap.parse_args()

    parsed = [parse_arg(s) for s in args.inputs]
    results = []  # (label, counter, n_eps, total_steps)
    all_actions = []
    for path, label in parsed:
        if not path.exists():
            print(f"[aviso] no existe {path} — se omite", file=sys.stderr)
            continue
        counts, n_eps = load_counts(path)
        steps = sum(counts.values())
        results.append((label, counts, n_eps, steps))
        for a in counts:
            if a not in all_actions:
                all_actions.append(a)

    # Referencia del experto (HTN): se inserta como primera serie, restringida a
    # las acciones que usan los modelos (excluye p.ej. jump, filtrado en training).
    if args.reference:
        rpath, rlabel = parse_arg(args.reference)
        if rpath.exists():
            ref_all = load_reference(rpath)
            ref = Counter({a: ref_all[a] for a in all_actions if a in ref_all})
            results.insert(0, (rlabel, ref, 0, sum(ref.values())))
        else:
            print(f"[aviso] referencia no existe: {rpath} — se omite", file=sys.stderr)

    if not results:
        print("Sin datos. ¿Re-ejecutaste la eval con la instrumentación nueva?")
        return

    # ── Texto ────────────────────────────────────────────────────────────────
    for label, counts, n_eps, steps in results:
        print(f"\n== {label}  ({n_eps} episodios, {steps} pasos) ==")
        if steps == 0:
            print("  (sin action_counts — re-ejecuta la eval con la versión instrumentada)")
            continue
        for action, c in counts.most_common():
            print(f"  {action:24s} {c:6d}  ({100*c/steps:5.1f}%)")

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    if args.latex:
        print("\n% --- tabla LaTeX (porcentajes por modelo) ---")
        cols = "l" + "|r" * len(results)
        print(r"\begin{tabular}{" + cols + "}")
        print(r"\rowcolor{udcpink!25}")
        header = r"\textbf{Acción}" + "".join(
            f" & \\textbf{{{lbl}}}" for lbl, *_ in results) + r" \\\hline"
        print(header)
        for action in all_actions:
            row = action.replace("_", r"\_")
            for _, counts, _, steps in results:
                pct = 100 * counts.get(action, 0) / steps if steps else 0.0
                row += f" & {pct:.1f}\\%"
            print(row + r" \\")
        print(r"\end{tabular}")

    # ── Histograma ───────────────────────────────────────────────────────────
    if args.out and any(steps for *_, steps in results):
        plot_histogram(results, all_actions, Path(args.out))


if __name__ == "__main__":
    main()
