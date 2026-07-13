#!/usr/bin/env python3
"""
progression.py — Tabla de "historia de diseño" de una iteración.

Compara las VARIANTES de una misma iteración en el orden en que se fueron
añadiendo (p.ej. IL: solo-estado → visual → doble-cabeza) sobre la métrica
ESPECÍFICA de esa iteración. No mezcla técnicas distintas (eso es el Cap. 8;
para ello usar summarize_eval.py).

Detecta el tipo de run automáticamente:
  - IL  → lee plots/history.json  (accuracy de validación, loss de acción, MSE de cámara)
  - RL  → lee metrics.jsonl       (episodios, tasa de éxito final, recompensa final)

La etiqueta de cada variante se resuelve: --labels > config["variant"] > inferida.
El ORDEN de la tabla = el orden en que pasas los runs (= orden de diseño).

Uso:
  # IL: progresión de la iteración 2
  python src/evaluation/progression.py \
      src/il/runs/<solo_estado> src/il/runs/<visual> src/il/runs/<doble_cabeza> \
      --labels "Solo estado" "Visual" "Doble cabeza"

  # RL: progresión de la iteración 3
  python src/evaluation/progression.py <run_estado> <run_visual> --labels "Estado" "Visual"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Detección de tipo y carga ────────────────────────────────────────────────

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _config(run_dir: Path) -> dict:
    return _read_json(run_dir / "config.json") or {}


def _variant_label(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    cfg = _config(run_dir)
    # campo explícito (IL: top-level; RL: dentro de "train")
    v = cfg.get("variant") or cfg.get("train", {}).get("variant")
    if v:
        return str(v)
    # inferencia
    if cfg.get("model_type"):                       # IL
        return {"rnn": "GRU", "convlstm": "ConvLSTM", "vit": "ViT"}.get(
            cfg["model_type"], cfg["model_type"])
    t = cfg.get("architecture", {}).get("type")     # RL
    if t:
        return t.split(" (")[0].strip()
    return run_dir.name


def _detect_type(run_dir: Path) -> str:
    if (run_dir / "plots" / "history.json").exists() or (run_dir / "history.json").exists():
        return "il"
    if (run_dir / "metrics.jsonl").exists():
        return "rl"
    return "unknown"


# ── Resúmenes por tipo ───────────────────────────────────────────────────────

def _summ_il(run_dir: Path, label: str) -> dict:
    hp = run_dir / "plots" / "history.json"
    if not hp.exists():
        hp = run_dir / "history.json"
    h = _read_json(hp) or {}
    val_acc = h.get("val_acc", [])
    return {
        "label":      label,
        "val_acc":    max(val_acc) if val_acc else None,
        "act_loss":   h.get("val_action_loss", [None])[-1] if h.get("val_action_loss") else None,
        "cam_mse":    h.get("val_camera_loss", [None])[-1] if h.get("val_camera_loss") else None,
        "epocas":     len(val_acc) if val_acc else None,
    }


def _summ_rl(run_dir: Path, label: str, window: int) -> dict:
    eps = []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        eps = [json.loads(l) for l in f if l.strip()]
    n = len(eps)
    w = min(window, n) if n else 0
    if eps and "success" in eps[0]:
        succ = np.array([1.0 if e.get("success") else 0.0 for e in eps])
    else:
        succ = np.array([1.0 if e.get("logs_collected", 0) > 0 else 0.0 for e in eps])
    rew = np.array([e.get("total_reward", 0.0) for e in eps])
    return {
        "label":        label,
        "episodios":    n,
        "exito_final":  float(succ[-w:].mean()) if w else 0.0,
        "reward_final": float(rew[-w:].mean())  if w else 0.0,
    }


# ── Salidas ──────────────────────────────────────────────────────────────────

def render_il(rows, out: Path):
    print("\n" + "=" * 74)
    print("PROGRESIÓN DE DISEÑO (IL) — métricas de validación")
    print("=" * 74)
    print(f"{'Variante':<20}{'Val. acc':>11}{'Loss acción':>14}{'MSE cámara':>13}{'Épocas':>9}")
    print("-" * 74)
    for r in rows:
        acc = f"{r['val_acc']*100:.1f}%" if r['val_acc'] is not None else "—"
        al  = f"{r['act_loss']:.3f}"     if r['act_loss'] is not None else "—"
        cm  = f"{r['cam_mse']:.4f}"      if r['cam_mse']  is not None else "—"
        ep  = r['epocas'] if r['epocas'] is not None else "—"
        print(f"{r['label']:<20}{acc:>11}{al:>14}{cm:>13}{str(ep):>9}")
    print("-" * 74 + "\n")

    lines = [
        r"\begin{table}[hp!]", r"  \centering", r"  \small",
        r"  \rowcolors{2}{white}{udcgray!25}",
        r"  \begin{tabular}{l|r|r|r}",
        r"  \rowcolor{udcpink!25}",
        r"  \textbf{Variante} & \textbf{Acc. validación (\%)} "
        r"& \textbf{Loss acción} & \textbf{MSE cámara} \\\hline",
    ]
    for r in rows:
        acc = f"{r['val_acc']*100:.1f}" if r['val_acc'] is not None else "--"
        al  = f"{r['act_loss']:.3f}"    if r['act_loss'] is not None else "--"
        cm  = f"{r['cam_mse']:.4f}"     if r['cam_mse']  is not None else "--"
        lab = r['label'].replace('_', r'\_')
        lines.append(f"  {lab} & {acc} & {al} & {cm} \\\\")
    lines += [r"  \end{tabular}",
              r"  \caption{Progresión de diseño de la iteración (métricas de validación).}",
              r"  \label{tab:progresion-il}", r"\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX → {out}")


def render_rl(rows, out: Path, window: int):
    print("\n" + "=" * 64)
    print(f"PROGRESIÓN DE DISEÑO (RL) — éxito final (últimos {window} eps)")
    print("=" * 64)
    print(f"{'Variante':<20}{'Episodios':>11}{'Éxito final':>14}{'Rec. final':>13}")
    print("-" * 64)
    for r in rows:
        print(f"{r['label']:<20}{r['episodios']:>11}{r['exito_final']*100:>13.1f}%{r['reward_final']:>13.2f}")
    print("-" * 64 + "\n")

    lines = [
        r"\begin{table}[hp!]", r"  \centering", r"  \small",
        r"  \rowcolors{2}{white}{udcgray!25}",
        r"  \begin{tabular}{l|r|r}",
        r"  \rowcolor{udcpink!25}",
        r"  \textbf{Variante} & \textbf{Episodios} & \textbf{Tasa de éxito (\%)} \\\hline",
    ]
    for r in rows:
        lab = r['label'].replace('_', r'\_')
        lines.append(f"  {lab} & {r['episodios']} & {r['exito_final']*100:.1f} \\\\")
    lines += [r"  \end{tabular}",
              r"  \caption{Progresión de diseño de la iteración (tasa de éxito en entorno).}",
              r"  \label{tab:progresion-rl}", r"\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX → {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tabla de progresión de diseño de una iteración (IL o RL).")
    p.add_argument("runs", nargs="+", help="Directorios de run EN ORDEN DE DISEÑO")
    p.add_argument("--labels", nargs="*", help="Etiquetas de variante (mismo orden que los runs)")
    p.add_argument("--window", type=int, default=20, help="Ventana de éxito final (solo RL)")
    p.add_argument("--out", default="src/evaluation/plots/progresion.tex")
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = [Path(r) for r in args.runs]

    types = {_detect_type(d) for d in run_dirs}
    if "unknown" in types:
        bad = [str(d) for d in run_dirs if _detect_type(d) == "unknown"]
        print(f"No reconozco el tipo de: {bad} (falta history.json o metrics.jsonl)")
        return
    if len(types) > 1:
        print(f"No mezcles tipos en una misma progresión: {types}. "
              f"La comparación entre técnicas es del Cap. 8 (summarize_eval.py).")
        return

    kind = types.pop()
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(run_dirs):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        lab = _variant_label(d, label)
        rows.append(_summ_il(d, lab) if kind == "il" else _summ_rl(d, lab, args.window))

    if kind == "il":
        render_il(rows, out)
    else:
        render_rl(rows, out, args.window)


if __name__ == "__main__":
    main()
