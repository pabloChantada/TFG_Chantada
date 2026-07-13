#!/usr/bin/env python3
"""
aggregate_runs.py — Agrega y compara varios runs de RL para los Caps. 7 y 8.

NO necesita Minecraft: lee únicamente los `metrics.jsonl` (+ `config.json`) que
los scripts de entrenamiento ya escriben en cada `run_dir`.

Produce:
  - Tabla comparativa por algoritmo (consola + CSV + snippet LaTeX listo para pegar).
  - Curvas de aprendizaje superpuestas (tasa de éxito y recompensa, media móvil).

Uso:
  # runs sueltos
  python src/evaluation/aggregate_runs.py src/rl/runs/dqn_xxx src/rl/runs/sac_yyy ...

  # todos los runs bajo un directorio
  python src/evaluation/aggregate_runs.py --root src/rl/runs

  # con etiquetas explícitas y ventana de media móvil
  python src/evaluation/aggregate_runs.py --root src/rl/runs --window 30 \
         --labels DQN PPO SAC

Definición de éxito (coherente con env.py): se usa el campo `success` del episodio
si existe; si no, se deriva de `logs_collected > 0` y, en último caso, de
`logs_broken > 0`. Se imprime qué criterio se usó por si hay runs antiguos.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Consola de Windows en UTF-8 para no romper con α, ×, acentos, etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Carga ───────────────────────────────────────────────────────────────────

def _load_episodes(run_dir: Path) -> list[dict]:
    """Lee metrics.jsonl (una línea JSON por episodio)."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        print(f"  [aviso] sin metrics.jsonl en {run_dir} — se omite")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _label_for(run_dir: Path, override: str | None) -> str:
    """Etiqueta del run: --labels > architecture.type del config.json > nombre del dir."""
    if override:
        return override
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            t = data.get("architecture", {}).get("type")
            if t:
                # "Hybrid SAC (Bernoulli...)" → "Hybrid SAC"; deja el detalle fuera.
                return t.split(" (")[0].strip()
        except Exception:
            pass
    return run_dir.name


def _success_series(episodes: list[dict]) -> tuple[np.ndarray, str]:
    """Vector 0/1 de éxito por episodio + el criterio empleado."""
    if episodes and "success" in episodes[0]:
        crit = "success"
        vals = [1.0 if e.get("success") else 0.0 for e in episodes]
    elif episodes and "logs_collected" in episodes[0]:
        crit = "logs_collected>0"
        vals = [1.0 if e.get("logs_collected", 0) > 0 else 0.0 for e in episodes]
    else:
        crit = "logs_broken>0"
        vals = [1.0 if e.get("logs_broken", 0) > 0 else 0.0 for e in episodes]
    return np.array(vals, dtype=np.float64), crit


def _ma(data: np.ndarray, w: int) -> np.ndarray:
    """Media móvil; devuelve vacío si no hay datos suficientes."""
    if len(data) < w:
        return np.array([])
    return np.convolve(data, np.ones(w) / w, mode="valid")


def _peak_index(ma_s: np.ndarray, ma_r: np.ndarray | None = None) -> int:
    """
    Índice del pico de la media móvil de éxito.

    Desempata (mismo éxito) por mayor recompensa media móvil y, si también empata,
    por episodio más tardío (ventana más avanzada del entrenamiento). Así un 40 %
    con más recompensa y más entrenado gana al primer 40 % que aparece.
    """
    if not len(ma_s):
        return 0
    best_s = float(ma_s.max())
    ties   = np.flatnonzero(np.isclose(ma_s, best_s))
    if ma_r is not None and len(ma_r) == len(ma_s) and len(ties) > 1:
        best_r = float(ma_r[ties].max())
        ties   = ties[np.isclose(ma_r[ties], best_r)]
    return int(ties.max())   # episodio más tardío entre los empatados


def _find_checkpoints(run_dir: Path) -> dict:
    """Mapa {episodio: ruta} de los checkpoints guardados (ficheros *ep<N>.pth)."""
    ckpts = {}
    for p in run_dir.glob("*ep*.pth"):
        m = re.search(r"ep(\d+)\.pth$", p.name)
        if m:
            ckpts[int(m.group(1))] = p
    return ckpts


def _peak_checkpoint(success: np.ndarray, rewards: np.ndarray, w: int, run_dir: Path):
    """
    (episodio_pico, mejor_ma, ruta_ckpt) según la media móvil de éxito.

    El episodio pico es el FINAL de la ventana de media móvil con mayor éxito
    (desempatando por recompensa y episodio, ver _peak_index); se elige el
    checkpoint guardado más cercano. Es un criterio de ENTRENAMIENTO (no mira la
    evaluación → sin fuga de información), reproducible y uniforme.
    """
    ma   = _ma(success, w)
    ma_r = _ma(rewards, w)
    if not len(ma):
        return None, None, None
    i_peak  = _peak_index(ma, ma_r)
    best_ma = float(ma[i_peak])
    best_ep = i_peak + w                      # episodio final de la ventana (1-based)
    ckpts   = _find_checkpoints(run_dir)
    if not ckpts:
        return best_ep, best_ma, None
    nearest = min(ckpts, key=lambda e: abs(e - best_ep))
    return best_ep, best_ma, ckpts[nearest]


# ── Resumen por run ──────────────────────────────────────────────────────────

def summarize(run_dir: Path, label: str, window: int) -> dict | None:
    episodes = _load_episodes(run_dir)
    if not episodes:
        return None

    success, crit = _success_series(episodes)
    rewards = np.array([e.get("total_reward", 0.0) for e in episodes], dtype=np.float64)
    steps   = np.array([e.get("steps", 0) for e in episodes], dtype=np.float64)
    n       = len(episodes)
    w       = min(window, n)

    # Tiempo total de entrenamiento si los episodios registran ep_time_s
    times   = [e.get("ep_time_s") for e in episodes if e.get("ep_time_s") is not None]
    total_h = sum(times) / 3600.0 if times else None

    success_ma = _ma(success, w)
    reward_ma  = _ma(rewards, w)
    if len(success_ma):
        i_peak     = _peak_index(success_ma, reward_ma)
        best_succ  = float(success_ma[i_peak])
        reward_peak = float(reward_ma[i_peak])   # recompensa MM en la MISMA ventana del pico
    else:
        best_succ   = float(success.mean())
        reward_peak = float(rewards.mean())

    best_ep, _best_ma, best_ckpt = _peak_checkpoint(success, rewards, w, run_dir)

    return {
        "best_ep":        best_ep,
        "best_ckpt":      str(best_ckpt) if best_ckpt else None,
        "label":          label,
        "run_dir":        str(run_dir),
        "criterio":       crit,
        "episodios":      n,
        "exito_global":   float(success.mean()),
        "exito_final":    float(success[-w:].mean()),   # últimos `w` episodios
        "exito_max_ma":   best_succ,                    # mejor media móvil (pico)
        "reward_final":   float(rewards[-w:].mean()),
        "reward_max":     float(rewards.max()),
        "reward_pico":    reward_peak,                   # recompensa MM en la ventana del pico
        "tiempo_h":       total_h,
        "_success":       success,
        "_rewards":       rewards,
        "_steps":         steps,
    }


# ── Salidas ──────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], window: int):
    print("\n" + "=" * 92)
    print(f"COMPARATIVA DE RUNS  (éxito 'final' = media de los últimos {window} episodios)")
    print("=" * 92)
    hdr = f"{'Algoritmo':<18}{'Eps':>6}{'Éxito fin':>11}{'Éxito máx':>11}{'Rec. fin':>11}{'Tiempo(h)':>11}"
    print(hdr)
    print("-" * 92)
    for r in rows:
        t = f"{r['tiempo_h']:.1f}" if r["tiempo_h"] is not None else "—"
        print(f"{r['label']:<18}{r['episodios']:>6}"
              f"{r['exito_final']*100:>10.1f}%{r['exito_max_ma']*100:>10.1f}%"
              f"{r['reward_final']:>11.2f}{t:>11}")
    print("-" * 92)
    crits = {r["criterio"] for r in rows}
    print(f"Criterio(s) de éxito: {', '.join(sorted(crits))}\n")


def print_checkpoints(rows: list[dict], window: int):
    """Checkpoint recomendado por run: pico de la media móvil de éxito (training)."""
    print("=" * 92)
    print(f"CHECKPOINT RECOMENDADO  (pico de la media móvil de éxito, ventana {window}, en ENTRENAMIENTO)")
    print("=" * 92)
    for r in rows:
        ep   = r.get("best_ep")
        ckpt = r.get("best_ckpt")
        name = Path(ckpt).name if ckpt else "— (sin checkpoints en el run)"
        ep_s = f"ep~{ep}" if ep else "—"
        print(f"  {r['label']:<22}pico {ep_s:>8}  →  {name}")
    print("  Selección por métrica de ENTRENAMIENTO (no mirar la evaluación). "
          "Anotar el elegido en la memoria.\n")


def write_csv(rows: list[dict], out: Path):
    cols = ["label", "run_dir", "criterio", "episodios", "exito_global",
            "exito_final", "exito_max_ma", "reward_final", "reward_max",
            "reward_pico", "tiempo_h"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV  → {out}")


def write_latex(rows: list[dict], out: Path, window: int):
    """Tabla en el estilo de la plantilla (rowcolors udcpink/udcgray)."""
    lines = [
        r"\begin{table}[hp!]",
        r"  \centering",
        r"  \small",
        r"  \rowcolors{2}{white}{udcgray!25}",
        r"  \begin{tabular}{l|r|r|r|r}",
        r"  \rowcolor{udcpink!25}",
        r"  \textbf{Algoritmo} & \textbf{Episodios} & \textbf{Éxito (\%)} "
        r"& \textbf{Recompensa} & \textbf{Tiempo (h)} \\\hline",
    ]
    for r in rows:
        t = f"{r['tiempo_h']:.1f}" if r["tiempo_h"] is not None else "--"
        lab = r["label"].replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        lines.append(
            f"  {lab} & {r['episodios']} & {r['exito_max_ma']*100:.1f} "
            f"& {r['reward_pico']:.2f} & {t} \\\\"
        )
    lines += [
        r"  \end{tabular}",
        r"  \caption{Comparativa de algoritmos de refuerzo. El éxito y la recompensa "
        f"se miden en el pico de la media móvil (ventana {window}), es decir, en el "
        r"checkpoint seleccionado para cada algoritmo.}",
        r"  \label{tab:rl-comparativa}",
        r"\end{table}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX → {out}")


def _plot_ma_split(ax, x, ma, peak_idx, color, label):
    """Dibuja la media móvil: sólida hasta el pico, punteada después + punto en el pico."""
    if not len(ma):
        return
    # Sólida hasta el pico (incluido), punteada desde el pico en adelante.
    ax.plot(x[:peak_idx + 1], ma[:peak_idx + 1],
            color=color, linewidth=2.2, label=label)
    ax.plot(x[peak_idx:], ma[peak_idx:],
            color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.85)
    ax.plot(x[peak_idx], ma[peak_idx], "o", color=color,
            markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=5)


def _kfmt(v, _):
    """Formatea pasos en miles: 10000 → '10k'."""
    return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def _draw_curve(ax, rows, window, colors, kind):
    """Dibuja en `ax` la curva de `kind` ('success' | 'reward') de todos los runs."""
    for i, r in enumerate(rows):
        c    = colors[i % len(colors)]
        succ = r["_success"]
        rew  = r["_rewards"]
        w    = min(window, len(succ))

        # Eje X = pasos de entorno acumulados (sample efficiency, estándar en RL).
        # Si el run no registró `steps`, se cae a numeración de episodios.
        cum  = np.cumsum(r["_steps"])
        if cum[-1] <= 0:
            cum = np.arange(1, len(succ) + 1, dtype=np.float64)
        x    = cum[w - 1:]          # un punto por valor de media móvil (alineado al fin de ventana)

        ma_s = _ma(succ, w)
        ma_r = _ma(rew, w)
        if not len(ma_s):
            continue

        # "Mejor punto del algoritmo": pico de la media móvil de éxito (checkpoint
        # recomendado), desempatando por recompensa y episodio. El mismo corte se
        # usa en ambas figuras.
        peak_idx = _peak_index(ma_s, ma_r)
        ma       = ma_s if kind == "success" else ma_r

        # Línea vertical tenue marcando el pico (checkpoint elegido), por detrás.
        ax.axvline(x[peak_idx], color=c, linewidth=1.0, alpha=0.60,
                   linestyle=(0, (2, 2)), zorder=0)
        # Media móvil: sólida hasta el pico, punteada después.
        _plot_ma_split(ax, x, ma, peak_idx, c, r["label"])


def _save_curve(rows, window, kind, title, ylabel, out):
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8, 5))

    _draw_curve(ax, rows, window, colors, kind)

    ax.set_title(title)
    ax.set_xlabel("Steps"); ax.set_ylabel(ylabel)
    if kind == "success":
        ax.set_ylim(-0.1, 0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    else:
        ax.axhline(0, color="gray", linestyle="dotted", linewidth=0.4, zorder=0)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_kfmt))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    handles, labels = ax.get_legend_handles_labels()
    peak_marker = plt.Line2D([], [], marker="o", color="#444444", linestyle="",
                             markersize=6, markeredgecolor="white",
                             label="pico (checkpoint elegido)")
    fig.legend(handles + [peak_marker], labels + ["pico (máx. exito y recompensa)"],
               loc="lower center", ncol=min(len(labels) + 1, 5),
               bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG  -> {out}")


def plot_curves(rows: list[dict], out: Path, window: int):
    """Genera DOS figuras separadas (éxito y recompensa) a partir de `out`."""
    # Estilo limpio para impresión + paleta colorblind-friendly (tab10).
    plt.rcParams.update({
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  11,
        "axes.edgecolor":   "#444444",
        "axes.linewidth":   0.8,
        "legend.frameon":   False,
        "figure.dpi":       150,
    })
    base = out.with_suffix("")
    ext  = out.suffix or ".png"
    _save_curve(rows, window, "success",
                "Tasa de éxito", "Éxito (media móvil)",
                base.with_name(base.name + "_exito" + ext))
    _save_curve(rows, window, "reward",
                "Recompensa por step", "Recompensa total",
                base.with_name(base.name + "_recompensa" + ext))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Agrega y compara runs de RL (sin Minecraft).")
    p.add_argument("runs", nargs="*", help="Directorios de run (cada uno con metrics.jsonl)")
    p.add_argument("--root", help="Directorio padre: usa todos sus subdirectorios con metrics.jsonl")
    p.add_argument("--labels", nargs="*", help="Etiquetas explícitas, en el mismo orden que los runs")
    p.add_argument("--window", type=int, default=20, help="Ventana de la media móvil (por defecto 20)")
    p.add_argument("--out", default="src/evaluation/plots/cap8_rl", help="Directorio de salida")
    return p.parse_args()


def main():
    args = parse_args()

    if args.root:
        root     = Path(args.root)
        run_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and (d / "metrics.jsonl").exists())
    else:
        run_dirs = [Path(r) for r in args.runs]

    if not run_dirs:
        print("No se encontraron runs. Pasa directorios o --root <dir>.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rd in enumerate(run_dirs):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        s = summarize(rd, _label_for(rd, label), args.window)
        if s:
            rows.append(s)

    if not rows:
        print("Ningún run con datos válidos.")
        return

    print_table(rows, args.window)
    print_checkpoints(rows, args.window)
    write_csv(rows,   out_dir / "comparativa_rl.csv")
    write_latex(rows, out_dir / "comparativa_rl.tex", args.window)
    plot_curves(rows, out_dir / "comparativa_rl.png", args.window)


if __name__ == "__main__":
    main()
