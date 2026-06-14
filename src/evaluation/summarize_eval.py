#!/usr/bin/env python3
"""
summarize_eval.py — Resume y compara la evaluación en entorno de HTN / IL / RL.

Lee los `eval.jsonl` unificados que escribe episode_metrics.js (y, en el futuro,
el evaluador de IL/RL) y aplica el MISMO criterio de éxito a las tres técnicas:

    éxito = (logs_collected >= K) Y (logs_broken >= 1)

con K configurable (--k). Así la comparativa del Cap. 8 es justa por construcción.

Uso:
  # una técnica (Iteración 1, HTN)
  python src/evaluation/summarize_eval.py data/eval/htn/eval.jsonl

  # varias técnicas a la vez (comparativa)
  python src/evaluation/summarize_eval.py data/eval/htn/eval.jsonl \
         data/eval/il/eval.jsonl data/eval/rl/eval.jsonl --k 5

Esquema de cada línea (eval.jsonl):
  {technique, ts, success_raw, logs_collected, logs_broken, target, steps, time_s}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Carga ────────────────────────────────────────────────────────────────────

def load_eval(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [aviso] no existe {path} — se omite")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _technique_of(records: list[dict], path: Path) -> str:
    if records and records[0].get("technique"):
        return str(records[0]["technique"]).upper()
    return path.parent.name.upper()


# ── Resumen ──────────────────────────────────────────────────────────────────

def _stats(a: np.ndarray) -> tuple:
    """(media, mediana, desviación típica muestral). std=0 con una sola muestra."""
    if len(a) == 0:
        return (None, None, None)
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return (float(a.mean()), float(np.median(a)), std)


def summarize(records: list[dict], label: str, k: int, require_broken: bool = False) -> dict:
    n        = len(records)
    collected = np.array([r.get("logs_collected", 0) for r in records], dtype=float)
    broken    = np.array([r.get("logs_broken", 0)    for r in records], dtype=float)
    times     = np.array([r.get("time_s")  for r in records if r.get("time_s")  is not None], dtype=float)
    steps     = np.array([r.get("steps")   for r in records if r.get("steps")   is not None], dtype=float)

    # Éxito: recoger >= K troncos. El gate "romper >= 1" (anti-trampa del RL frente
    # a drops residuales) es opcional, porque no todas las técnicas registran
    # logs_broken de forma fiable (el HTN lo cuenta por evento, que a veces no salta).
    success = collected >= k
    if require_broken:
        success = success & (broken >= 1)

    lc_mean, lc_med, lc_std = _stats(collected)
    t_mean,  t_med,  t_std  = _stats(times)
    s_mean,  s_med,  s_std  = _stats(steps)

    return {
        "label":          label,
        "episodios":      n,
        "exito":          float(success.mean()) if n else 0.0,
        "n_exitos":       int(success.sum()),
        "logs_media":     lc_mean if lc_mean is not None else 0.0,
        "logs_mediana":   lc_med,
        "logs_std":       lc_std,
        "logs_max":       float(collected.max())  if n else 0.0,
        "broken_media":   float(broken.mean())    if n else 0.0,
        "tiempo_media_s": t_mean,
        "tiempo_mediana": t_med,
        "tiempo_std":     t_std,
        "steps_media":    s_mean,
        "steps_mediana":  s_med,
        "steps_std":      s_std,
        "_success":       success,
        "_collected":     collected,
        "_times":         times,
    }


# ── Salidas ──────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], k: int, require_broken: bool = False):
    crit = f"logs_collected ≥ {k}" + (" Y logs_broken ≥ 1" if require_broken else "")
    print("\n" + "=" * 88)
    print(f"EVALUACIÓN EN ENTORNO  (éxito = {crit})")
    print("=" * 88)
    print(f"{'Técnica':<10}{'Eps':>6}{'Éxito':>10}{'Logs (μ)':>11}{'Rotos (μ)':>11}{'Tiempo(s)':>12}{'Steps(μ)':>10}")
    print("-" * 88)
    for r in rows:
        t  = f"{r['tiempo_media_s']:.1f}" if r["tiempo_media_s"] is not None else "—"
        st = f"{r['steps_media']:.0f}"    if r["steps_media"]    is not None else "—"
        print(f"{r['label']:<10}{r['episodios']:>6}"
              f"{r['exito']*100:>9.1f}%{r['logs_media']:>11.2f}{r['broken_media']:>11.2f}"
              f"{t:>12}{st:>10}")
    print("-" * 88 + "\n")


def print_detail(rows: list[dict]):
    """Detalle por técnica con media / mediana / desviación típica (rellena tab:eval-htn)."""
    def f(v):
        return f"{v:.2f}" if v is not None else "—"
    for r in rows:
        print(f"\nDETALLE {r['label']}  (n={r['episodios']}, éxito {r['exito']*100:.1f}%)")
        print(f"  {'Métrica':<24}{'Media':>10}{'Mediana':>10}{'Desv.típ.':>11}")
        print(f"  {'-'*55}")
        print(f"  {'Troncos por episodio':<24}{f(r['logs_media']):>10}{f(r['logs_mediana']):>10}{f(r['logs_std']):>11}")
        print(f"  {'Tiempo por episodio (s)':<24}{f(r['tiempo_media_s']):>10}{f(r['tiempo_mediana']):>10}{f(r['tiempo_std']):>11}")
        print(f"  {'Pasos por episodio':<24}{f(r['steps_media']):>10}{f(r['steps_mediana']):>10}{f(r['steps_std']):>11}")
    print()


def write_latex(rows: list[dict], out: Path, k: int, require_broken: bool = False):
    crit = f"recoger $\\geq {k}$ troncos" + (r" y romper $\geq 1$" if require_broken else "")

    def f(v, nd=2):
        return f"{v:.{nd}f}" if v is not None else "--"

    lines = [
        r"\begin{table}[hp!]",
        r"  \centering",
        r"  \small",
        r"  \rowcolors{3}{white}{udcgray!25}",
        r"  \begin{tabular}{l|r|r|rrr|rrr}",
        r"  \rowcolor{udcpink!25}",
        r"   & & & \multicolumn{3}{c|}{\textbf{Troncos}} & \multicolumn{3}{c}{\textbf{Tiempo (s)}} \\",
        r"  \rowcolor{udcpink!25}",
        r"  \textbf{Técnica} & \textbf{Eps.} & \textbf{Éxito (\%)} "
        r"& \textbf{$\mu$} & \textbf{med.} & \textbf{$\sigma$} "
        r"& \textbf{$\mu$} & \textbf{med.} & \textbf{$\sigma$} \\\hline",
    ]
    for r in rows:
        lab = r["label"].replace("_", r"\_")
        lines.append(
            f"  {lab} & {r['episodios']} & {r['exito']*100:.1f} "
            f"& {f(r['logs_media'])} & {f(r['logs_mediana'])} & {f(r['logs_std'])} "
            f"& {f(r['tiempo_media_s'], 1)} & {f(r['tiempo_mediana'], 1)} & {f(r['tiempo_std'], 1)} \\\\"
        )
    lines += [
        r"  \end{tabular}",
        r"  \caption{Evaluación en entorno bajo la métrica común "
        f"(éxito: {crit}). Troncos y tiempo por episodio: media ($\\mu$), mediana "
        r"y desviación típica ($\sigma$).}",
        r"  \label{tab:eval-comparativa}",
        r"\end{table}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX → {out}")


def plot_summary(rows: list[dict], out: Path, k: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [r["label"] for r in rows]

    # Barras de tasa de éxito
    succ = [r["exito"] * 100 for r in rows]
    bars = ax1.bar(labels, succ, color="teal", alpha=0.8)
    for b, s in zip(bars, succ):
        ax1.text(b.get_x() + b.get_width() / 2, s + 1, f"{s:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Tasa de éxito (%)")
    ax1.set_title(f"Tasa de éxito (≥{k} troncos)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Boxplot de troncos recogidos por episodio
    data = [r["_collected"] for r in rows]
    ax2.boxplot(data, showmeans=True)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.axhline(k, color="red", linestyle="--", linewidth=0.8, label=f"objetivo (K={k})")
    ax2.set_ylabel("Troncos recogidos por episodio")
    ax2.set_title("Distribución de troncos recogidos")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG  → {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Resume y compara la evaluación en entorno (HTN/IL/RL).")
    p.add_argument("files", nargs="+", help="Uno o más eval.jsonl")
    p.add_argument("--k", type=int, default=1,
                   help="Umbral de troncos para considerar éxito (por defecto 1)")
    p.add_argument("--require-broken", action="store_true",
                   help="Exigir además logs_broken >= 1 (anti-trampa del RL frente a "
                        "drops residuales). Off por defecto: el HTN no registra roturas de forma fiable.")
    p.add_argument("--out", default="src/evaluation/plots", help="Directorio de salida")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in args.files:
        path = Path(fp)
        recs = load_eval(path)
        if recs:
            rows.append(summarize(recs, _technique_of(recs, path), args.k, args.require_broken))

    if not rows:
        print("Sin datos de evaluación.")
        return

    print_table(rows, args.k, args.require_broken)
    print_detail(rows)
    write_latex(rows, out_dir / "eval_comparativa.tex", args.k, args.require_broken)
    plot_summary(rows, out_dir / "eval_comparativa.png", args.k)


if __name__ == "__main__":
    main()
