#!/usr/bin/env python3
"""
dataset_dist.py — Distribución de clases de acción del dataset de demostraciones HTN.

Cuenta el campo `action` de los .jsonl grabados por dataset_recorder.js (es el mismo
que filtra `MinecraftDataset.filter_min_samples`) y produce:
  - tabla en consola (acción, muestras, %)
  - gráfico de BARRAS ordenado, con línea de umbral que resalta las clases que se
    descartan (justifica visualmente la limpieza)  → barras es lo correcto aquí,
    no un box plot (esto son conteos categóricos, no una variable continua)
  - snippet LaTeX de la tabla de conteos

Uso:
  # un fichero, varios, o un directorio (glob *.jsonl)
  python src/evaluation/dataset_dist.py data/recordings --min-samples 30
  python src/evaluation/dataset_dist.py data/recordings/Rec_1_*.jsonl data/recordings/Rec_2_*.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Carga ────────────────────────────────────────────────────────────────────

def _resolve_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif any(c in p for c in "*?["):
            files.extend(sorted(Path().glob(p)))
        else:
            files.append(path)
    return [f for f in files if f.exists()]


def count_actions(files: list[Path], field: str, require_image: bool):
    """Devuelve (conteo agregado, {run/session: Counter})."""
    counts: Counter = Counter()
    per_run: dict[str, Counter] = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                action = item.get(field)
                if not isinstance(action, str) or not action:
                    continue
                if require_image:
                    img = item.get("image")
                    if not img or not Path(img).exists():
                        continue
                counts[action] += 1
                run = item.get("session_id") or fp.stem
                per_run.setdefault(run, Counter())[action] += 1
    return counts, per_run


# ── Salidas ──────────────────────────────────────────────────────────────────

def print_table(counts: Counter, total: int, min_samples: int):
    print("\n" + "=" * 60)
    print(f"DISTRIBUCIÓN DE CLASES DE ACCIÓN  (total {total} muestras)")
    print("=" * 60)
    print(f"{'Acción':<26}{'Muestras':>10}{'%':>9}   ")
    print("-" * 60)
    for action, n in counts.most_common():
        flag = "  ← descartada" if n < min_samples else ""
        print(f"{action:<26}{n:>10}{100*n/total:>8.1f}%{flag}")
    print("-" * 60)
    dropped = [a for a, n in counts.items() if n < min_samples]
    print(f"Umbral min_samples={min_samples}  →  descartadas: {dropped or 'ninguna'}\n")


def write_latex(counts: Counter, total: int, min_samples: int, out: Path):
    lines = [
        r"\begin{table}[hp!]", r"  \centering", r"  \small",
        r"  \rowcolors{2}{white}{udcgray!25}",
        r"  \begin{tabular}{l|r|r}",
        r"  \rowcolor{udcpink!25}",
        r"  \textbf{Acción} & \textbf{Muestras} & \textbf{\%} \\\hline",
    ]
    for action, n in counts.most_common():
        name = action.replace("_", r"\_")
        mark = r"$^{\dagger}$" if n < min_samples else ""
        lines.append(f"  \\texttt{{{name}}}{mark} & {n} & {100*n/total:.1f} \\\\")
    lines += [
        r"  \end{tabular}",
        r"  \caption{Distribución de clases de acción en el dataset de demostraciones. "
        f"Las marcadas con $\\dagger$ caen por debajo del umbral ($<{min_samples}$ "
        r"muestras) y se descartan en la limpieza.}",
        r"  \label{tab:dataset-clases}", r"\end{table}", "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX → {out}")


def plot_bars(counts: Counter, total: int, min_samples: int, out: Path):
    items   = counts.most_common()
    labels  = [a for a, _ in items]
    values  = [n for _, n in items]
    colors  = ["crimson" if n < min_samples else "steelblue" for n in values]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(labels))))
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    ax.invert_yaxis()  # más frecuente arriba

    for b, n in zip(bars, values):
        ax.text(n + total * 0.005, b.get_y() + b.get_height() / 2,
                f"{n} ({100*n/total:.1f}%)", va="center", fontsize=8)

    ax.axvline(min_samples, color="red", linestyle="--", linewidth=1,
               label=f"umbral ({min_samples})")
    ax.set_xlabel("Número de muestras")
    ax.set_title("Distribución de clases de acción en el dataset HTN")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG  → {out}")


def plot_trend(counts: Counter, per_run: dict[str, Counter], out: Path):
    """Box plot del % de cada clase a lo largo de los runs (tendencia/consistencia)."""
    runs = [c for c in per_run.values() if sum(c.values()) > 0]
    if len(runs) < 2:
        print("Tendencia omitida: hacen falta ≥2 runs (session_id) para el box plot.")
        return

    classes = [a for a, _ in counts.most_common()]            # orden por frecuencia global
    data = []
    for action in classes:
        props = [100.0 * rc.get(action, 0) / sum(rc.values()) for rc in runs]
        data.append(props)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(classes))))
    ax.boxplot(data, vert=False, showmeans=True)
    ax.set_yticks(range(1, len(classes) + 1))
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel(f"% de muestras por run (n={len(runs)} runs)")
    ax.set_title("Consistencia de la distribución de acciones entre runs")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG  → {out}  ({len(runs)} runs)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distribución de clases de acción del dataset HTN.")
    p.add_argument("paths", nargs="+", help="Ficheros .jsonl, glob o directorio")
    p.add_argument("--min-samples", type=int, default=30,
                   help="Umbral de descarte (línea en el gráfico). Por defecto 30")
    p.add_argument("--field", default="action", help="Campo a contar (por defecto 'action')")
    p.add_argument("--require-image", action="store_true",
                   help="Contar solo frames cuya imagen existe (como hace el dataloader)")
    p.add_argument("--out", default="src/evaluation/plots", help="Directorio de salida")
    return p.parse_args()


def main():
    args  = parse_args()
    files = _resolve_files(args.paths)
    if not files:
        print("No se encontraron .jsonl en las rutas dadas.")
        return
    print(f"Ficheros: {len(files)}")

    counts, per_run = count_actions(files, args.field, args.require_image)
    total  = sum(counts.values())
    if total == 0:
        print("Sin muestras (¿campo correcto? ¿faltan imágenes con --require-image?).")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Runs (session_id) detectados: {len(per_run)}")
    print_table(counts, total, args.min_samples)
    write_latex(counts, total, args.min_samples, out_dir / "dataset_clases.tex")
    plot_bars(counts, total, args.min_samples, out_dir / "dataset_clases.png")
    plot_trend(counts, per_run, out_dir / "dataset_clases_tendencia.png")


if __name__ == "__main__":
    main()
