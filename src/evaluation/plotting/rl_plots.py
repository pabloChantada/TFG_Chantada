#!/usr/bin/env python3
"""
rl_plots.py — Figuras extra para la comparativa de refuerzo (Caps. 8-9).

No necesita Minecraft: lee los `metrics.jsonl` (+ `config.json`) de los runs y los
`eval_*.jsonl` de la evaluación en entorno.

Genera:
  A) Distribución de acciones en evaluación, por algoritmo  (desde eval_*.jsonl)
     → <out>/cap8_rl/accion_dist_rl.png
  B) Diagnóstico de estabilidad: SAC (entropía + α), DQN (ε)  (desde metrics.jsonl)
     → <out>/cap8_rl/estabilidad_rl.png
  C) Coste vs rendimiento: horas de entrenamiento vs éxito en eval
     → <out>/cap9_comparativa/coste_vs_rendimiento.png

Uso:
  python src/evaluation/rl_plots.py \
      --runs  src/rl/visual/runs/random_... src/rl/visual/runs/<dqn> \
              src/rl/visual/runs/ppo_..._good src/rl/visual/runs/sac_... \
      --eval  data/eval/rl/eval_random.jsonl data/eval/rl/eval_dqn.jsonl \
              data/eval/rl/eval_ppo.jsonl data/eval/rl/eval_sac.jsonl \
              data/eval/htn/eval.jsonl data/eval/il/eval_gru.jsonl \
      --k 5 --il-hours 0.5
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

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "legend.title_fontsize": 12,
})

# Orden y nombres de las 7 acciones discretas (coherente con shared/constants.py).
ACTIONS = ["attack", "move_forward_sprint", "move_forward_jump",
           "camera_right", "camera_left", "camera_up", "camera_down"]

# Acciones comunes con el agente de imitación (sus 3 clases discretas; la cámara
# en IL era continua, no acción discreta). Permite comparar RL vs IL en igualdad.
COMMON_ACTIONS = ["attack", "move_forward_sprint", "move_forward_jump"]

# Orden de presentación de los algoritmos.
ALGO_ORDER = ["random", "DQN", "PPO", "SAC", "HTN", "IL"]


def canon(name: str) -> str:
    """Etiqueta canónica de un run o fichero de eval."""
    n = name.lower()
    if "random" in n or "ninguna" in n:                 return "random"
    if "ppo" in n:                                       return "PPO"
    if "dagger" in n:                                    return "SAC-DAgger"
    if "sac" in n:                                       return "SAC"
    if "dqn" in n or "double" in n:                      return "DQN"
    if "htn" in n:                                       return "HTN"
    if any(k in n for k in ("gru", "convlstm", "vit", "state", "/il", "il_", "eval_il")):
        return "IL"
    return name


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _run_label(run_dir: Path) -> str:
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            t = json.loads(cfg.read_text(encoding="utf-8")).get("architecture", {}).get("type", "")
            if t:
                return canon(t)
        except Exception:
            pass
    return canon(run_dir.name)


def _eval_label(path: Path, episodes: list[dict]) -> str:
    # Prioriza el nombre del fichero (eval_dqn → DQN); cae al campo technique.
    lab = canon(path.stem)
    if lab in ("DQN", "PPO", "SAC", "random", "HTN", "IL", "SAC-DAgger"):
        return lab
    tech = episodes[0].get("technique", "") if episodes else ""
    return canon(tech) or lab


def _order_key(label: str) -> int:
    return ALGO_ORDER.index(label) if label in ALGO_ORDER else len(ALGO_ORDER)


# ── A) Distribución de acciones en evaluación ─────────────────────────────────

def plot_action_dist(eval_files: list[Path], out: Path, actions=ACTIONS,
                     title="Distribución de acciones en evaluación por algoritmo",
                     only_labels=None):
    """
    Barras horizontales apiladas: fracción de cada acción por técnica.

    `actions` define el subconjunto a mostrar (renormalizado sobre él): ACTIONS para
    las 7 del RL, o COMMON_ACTIONS para comparar RL con IL en las acciones comunes.
    `only_labels` filtra qué técnicas incluir.
    """
    dist = {}   # label -> vector de fracciones (orden `actions`)
    for fp in eval_files:
        eps = _load_jsonl(fp)
        if not eps or "action_counts" not in eps[0]:
            continue
        lab = _eval_label(fp, eps)
        if only_labels and lab not in only_labels:
            continue
        totals = np.zeros(len(actions), dtype=np.float64)
        for e in eps:
            ac = e.get("action_counts") or {}
            for i, a in enumerate(actions):
                totals[i] += ac.get(a, 0)
        s = totals.sum()
        if s == 0:
            continue
        dist[lab] = totals / s   # renormaliza sobre el subconjunto `actions`

    if not dist:
        print(f"  [A] sin action_counts para {out.name} — se omite")
        return

    labels = sorted(dist.keys(), key=_order_key)
    data   = np.array([dist[l] for l in labels])     # (n_tec, len(actions))

    fig, ax = plt.subplots(figsize=(10, 0.9 * len(labels) + 1.5))
    cmap   = plt.get_cmap("tab10")
    left   = np.zeros(len(labels))
    for i, a in enumerate(actions):
        ax.barh(labels, data[:, i], left=left, color=cmap(i % 10), label=a)
        left += data[:, i]
    ax.set_xlabel("Fracción de acciones")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(ncol=min(4, len(actions)), fontsize=11, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [A] PNG → {out}")


# ── B) Diagnóstico de estabilidad ─────────────────────────────────────────────

def plot_stability(run_dirs: list[Path], out: Path):
    """SAC: entropía + α (eje doble). DQN: ε. Una columna por algoritmo disponible."""
    runs = {}
    for rd in run_dirs:
        mp = rd / "metrics.jsonl"
        if not mp.exists():
            continue
        runs[_run_label(rd)] = _load_jsonl(mp)

    sac = runs.get("SAC")
    dqn = runs.get("DQN")
    panels = sum(x is not None for x in (sac, dqn))
    if panels == 0:
        print("  [B] sin runs SAC/DQN con metrics — se omite")
        return

    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 4.2))
    if panels == 1:
        axes = [axes]
    i = 0
    tags = ["(a)", "(b)"]  # etiqueta de panel, asignada por orden de aparición

    def _cum_steps(episodes):
        # Eje en pasos acumulados (mismo criterio que aggregate_runs.py).
        return np.cumsum([e.get("steps", 0) for e in episodes])

    if sac is not None:
        ax = axes[i]; tag = tags[i]; i += 1
        steps = _cum_steps(sac)
        ent = [e.get("entropy", 0.0) for e in sac]
        al  = [e.get("alpha", 0.0) for e in sac]
        ax.plot(steps, ent, color="tab:blue", label="entropía")
        ax.set_xlabel("Steps"); ax.set_ylabel("Entropía", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(steps, al, color="tab:red", label="α")
        ax2.set_ylabel("Temperatura α", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"{tag} SAC: entropía saturada y explosión de α")

    if dqn is not None:
        ax = axes[i]; tag = tags[i]; i += 1
        steps = _cum_steps(dqn)
        eps = [e.get("epsilon", 0.0) for e in dqn]
        ax.plot(steps, eps, color="tab:green")
        ax.set_xlabel("Steps"); ax.set_ylabel("ε (exploración)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{tag} DQN: decaimiento de ε")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [B] PNG → {out}")


# ── C) Coste vs rendimiento ───────────────────────────────────────────────────

def _train_hours(run_dir: Path) -> float | None:
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return None
    eps = _load_jsonl(mp)
    times = [e.get("ep_time_s") for e in eps if e.get("ep_time_s") is not None]
    return sum(times) / 3600.0 if times else None


def _eval_success(path: Path, k: int) -> float:
    # Criterio común (coherente con summarize_eval, --require-broken OFF): solo
    # logs_collected >= k. HTN/IL no registran logs_broken de forma fiable, así que
    # NO se exige aquí; al evaluar con world_reset por episodio no hay drops residuales.
    eps = _load_jsonl(path)
    if not eps:
        return 0.0
    ok = sum(1 for e in eps if e.get("logs_collected", 0) >= k)
    return 100.0 * ok / len(eps)


def plot_cost_vs_perf(run_dirs: list[Path], eval_files: list[Path],
                      out: Path, k: int, il_hours: float | None):
    """Dispersión: horas de entrenamiento (cómputo) vs éxito en evaluación."""
    # Horas por algoritmo (RL). HTN y random: 0 (no entrenan una red).
    hours = {"HTN": 0.0, "random": 0.0}
    for rd in run_dirs:
        lab = _run_label(rd)
        h = _train_hours(rd)
        if lab in ("DQN", "PPO", "SAC", "SAC-DAgger") and h is not None:
            hours[lab] = h
    if il_hours is not None:
        hours["IL"] = il_hours

    # Éxito por algoritmo (eval).
    succ = {}
    for fp in eval_files:
        eps = _load_jsonl(fp)
        succ[_eval_label(fp, eps)] = _eval_success(fp, k)

    labels = [l for l in ALGO_ORDER if l in hours and l in succ]
    if not labels:
        print("  [C] no hay (horas, éxito) emparejables — se omite")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    for j, l in enumerate(labels):
        ax.scatter(hours[l], succ[l], s=120, color=cmap(j % 10), zorder=3)
        ax.annotate(l, (hours[l], succ[l]), textcoords="offset points",
                    xytext=(8, 6), fontsize=13, fontweight="bold")
    ax.set_xlabel("Coste de entrenamiento (horas de cómputo)")
    ax.set_ylabel(f"Tasa de éxito en evaluación (≥{k} troncos) [%]")
    ax.set_title("Coste de entrenamiento frente a rendimiento")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [C] PNG → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Figuras extra de la comparativa de refuerzo.")
    p.add_argument("--runs",  nargs="*", default=[], help="Directorios de run (metrics.jsonl)")
    p.add_argument("--eval",  nargs="*", default=[], help="Ficheros eval_*.jsonl")
    p.add_argument("--k", type=int, default=5, help="Troncos para considerar éxito (def. 5)")
    p.add_argument("--il-hours", type=float, default=None,
                   help="Horas de entrenamiento del IL (no se registran en sus runs)")
    p.add_argument("--out", default="src/evaluation/plots", help="Directorio base de salida")
    return p.parse_args()


def main():
    args   = parse_args()
    out    = Path(args.out)
    runs   = [Path(r) for r in args.runs]
    evals  = [Path(e) for e in args.eval]
    cap8   = out / "cap8_rl"
    cap9   = out / "cap9_comparativa"

    if evals:
        # Completa (7 acciones), solo algoritmos RL.
        plot_action_dist(evals, cap8 / "accion_dist_rl.png", ACTIONS,
                         "Distribución de acciones en evaluación por algoritmo (RL)",
                         only_labels={"random", "DQN", "PPO", "SAC", "SAC-DAgger"})
        # Restringida a las acciones comunes con IL (RL + IL, renormalizada).
        plot_action_dist(evals, cap8 / "accion_dist_comun.png", COMMON_ACTIONS,
                         "Acciones comunes con IL (atacar / avanzar) por técnica",
                         only_labels={"DQN", "PPO", "SAC", "IL"})
    if runs:
        plot_stability(runs, cap8 / "estabilidad_rl.png")
    if runs and evals:
        plot_cost_vs_perf(runs, evals, cap9 / "coste_vs_rendimiento.png",
                          args.k, args.il_hours)


if __name__ == "__main__":
    main()
