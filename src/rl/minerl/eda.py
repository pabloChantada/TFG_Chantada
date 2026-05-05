"""
EDA del dataset MineRL Treechop-v0.

Objetivo: entender la distribución del dataset humano antes de diseñar el
mapping de acciones a nuestro espacio discreto de 7 acciones (Mineflayer).

Cubre:
  1. Estructura: nº episodios, frames, duraciones, success rate.
  2. Reward: distribución de reward total y de logs por episodio.
  3. Acciones binarias: frecuencia individual y co-ocurrencias.
  4. Cámara: distribución de magnitudes de dyaw / dpitch.
  5. Mapping simulado a nuestras 7 acciones: pérdida de información,
     distribución resultante vs nuestro dataset IL (data/train.jsonl).

Salida:
  - Stats por consola.
  - PNGs en data/MineRLTreechop-v0/eda/.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATASET_ROOT = Path("data/MineRLTreechop-v0/MineRLTreechop-v0")
OUT_DIR      = Path("data/MineRLTreechop-v0/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Acciones binarias presentes en MineRL Treechop-v0
BINARY_ACTIONS = ["forward", "left", "back", "right",
                  "jump", "sneak", "sprint", "attack"]

# Umbral (en grados) para considerar la cámara "activa" en un step
CAMERA_THRESH_DEG = 0.05

# Mapping a nuestras 7 acciones discretas (RL Mineflayer):
#   ["attack", "move_forward_sprint", "move_forward_jump",
#    "camera_right", "camera_left", "camera_up", "camera_down"]
OUR_ACTIONS = ["attack", "move_forward_sprint", "move_forward_jump",
               "camera_right", "camera_left", "camera_up", "camera_down"]


def list_episodes() -> list[Path]:
    return sorted(p for p in DATASET_ROOT.iterdir()
                  if p.is_dir() and (p / "rendered.npz").exists())


def load_episode(ep_dir: Path) -> dict:
    npz  = np.load(ep_dir / "rendered.npz")
    meta = json.loads((ep_dir / "metadata.json").read_text())
    return {
        "name":     ep_dir.name,
        "meta":     meta,
        "reward":   npz["reward"].astype(np.float32),
        "camera":   npz["action$camera"].astype(np.float32),       # (T, 2): pitch, yaw (grados)
        "binary":   {a: npz[f"action$"+a].astype(np.int8) for a in BINARY_ACTIONS},
        "n_steps":  int(npz["reward"].shape[0]),
    }


def map_to_our_action(forward, jump, sprint, attack, dpitch, dyaw):
    """Mapping prioritario MineRL→nuestras 7. Devuelve idx o None si no encaja."""
    abs_yaw, abs_pitch = abs(dyaw), abs(dpitch)
    cam_active = max(abs_yaw, abs_pitch) > CAMERA_THRESH_DEG

    # Prioridad: attack > camera > forward+sprint/jump
    # (attack se elige primero porque es la única señal de "tala" — sin ella, el
    # mapping no aprende a romper troncos. Cámara antes que forward porque las
    # acciones forward incluyen movimiento implícito durante toda la trayectoria.)
    if attack:
        return 0  # attack
    if cam_active:
        if abs_yaw >= abs_pitch:
            return 3 if dyaw > 0 else 4  # camera_right / camera_left
        return 5 if dpitch < 0 else 6   # camera_up (pitch negativo = mirar arriba) / camera_down
    if forward and sprint:
        return 1  # move_forward_sprint
    if forward and jump:
        return 2  # move_forward_jump
    if forward:
        return 1  # forward sin modificadores → sprint por defecto
    return None  # back/left/right/sneak puro → sin acción equivalente


def main():
    ep_paths = list_episodes()
    print(f"Encontrados {len(ep_paths)} episodios en {DATASET_ROOT}\n")
    if not ep_paths:
        return

    total_steps    = 0
    total_reward   = 0.0
    success_count  = 0
    ep_lengths     = []
    ep_rewards     = []
    ep_durations_s = []
    binary_counts  = Counter()
    cooc_counts    = Counter()  # co-ocurrencias de pares de flags
    cam_yaw_abs    = []
    cam_pitch_abs  = []
    cam_active_steps   = 0
    cam_inactive_steps = 0

    our_action_counts = Counter()
    unmapped_steps     = 0
    multi_flag_steps   = 0  # steps con >1 flag binario activo (info perdida en mapping)
    rewards_in_attack  = 0  # rewards que ocurren en steps con attack=1 (sanity check)

    for i, ep in enumerate(ep_paths):
        d = load_episode(ep)
        T = d["n_steps"]
        total_steps  += T
        total_reward += float(d["reward"].sum())
        ep_lengths.append(T)
        ep_rewards.append(float(d["reward"].sum()))
        ep_durations_s.append(d["meta"].get("duration_ms", 0) / 1000.0)
        if str(d["meta"].get("success", "")).lower() == "true":
            success_count += 1

        for a in BINARY_ACTIONS:
            binary_counts[a] += int(d["binary"][a].sum())

        # Co-ocurrencias entre pares
        for i_a, a in enumerate(BINARY_ACTIONS):
            for b in BINARY_ACTIONS[i_a+1:]:
                cooc_counts[(a, b)] += int((d["binary"][a] & d["binary"][b]).sum())

        # Cámara
        dpitch = d["camera"][:, 0]
        dyaw   = d["camera"][:, 1]
        active_mask = (np.abs(dyaw) > CAMERA_THRESH_DEG) | (np.abs(dpitch) > CAMERA_THRESH_DEG)
        cam_active_steps   += int(active_mask.sum())
        cam_inactive_steps += int((~active_mask).sum())
        cam_yaw_abs.append(np.abs(dyaw[active_mask]))
        cam_pitch_abs.append(np.abs(dpitch[active_mask]))

        # Mapping
        flags_per_step = sum(d["binary"][a] for a in BINARY_ACTIONS)
        multi_flag_steps += int((flags_per_step > 1).sum())
        for t in range(T):
            idx = map_to_our_action(
                forward=d["binary"]["forward"][t],
                jump   =d["binary"]["jump"][t],
                sprint =d["binary"]["sprint"][t],
                attack =d["binary"]["attack"][t],
                dpitch =dpitch[t],
                dyaw   =dyaw[t],
            )
            if idx is None:
                unmapped_steps += 1
            else:
                our_action_counts[OUR_ACTIONS[idx]] += 1
            if d["binary"]["attack"][t] and d["reward"][t] > 0:
                rewards_in_attack += 1

        if (i + 1) % 50 == 0:
            print(f"  procesados {i+1}/{len(ep_paths)}...")

    cam_yaw_abs   = np.concatenate(cam_yaw_abs)   if cam_yaw_abs   else np.array([])
    cam_pitch_abs = np.concatenate(cam_pitch_abs) if cam_pitch_abs else np.array([])

    # ─── 1. Estructura ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. ESTRUCTURA DEL DATASET")
    print("="*70)
    print(f"Episodios:         {len(ep_paths)}")
    print(f"Steps totales:     {total_steps:,}  (~{total_steps*0.05/3600:.1f}h a 50ms/step)")
    print(f"Success rate:      {success_count}/{len(ep_paths)}  ({100*success_count/len(ep_paths):.1f}%)")
    print(f"Steps por episodio: media={np.mean(ep_lengths):.0f}  "
          f"mediana={np.median(ep_lengths):.0f}  min={min(ep_lengths)}  max={max(ep_lengths)}")
    print(f"Duración real:     media={np.mean(ep_durations_s):.1f}s  "
          f"min={min(ep_durations_s):.1f}s  max={max(ep_durations_s):.1f}s")

    # ─── 2. Reward ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("2. REWARD (= nº de logs en Treechop-v0)")
    print("="*70)
    rewards_arr = np.array(ep_rewards)
    print(f"Reward total acumulado: {total_reward:.0f} logs")
    print(f"Reward por episodio: media={rewards_arr.mean():.1f}  "
          f"mediana={np.median(rewards_arr):.0f}  min={rewards_arr.min():.0f}  "
          f"max={rewards_arr.max():.0f}")
    print(f"Logs/step (densidad): {total_reward/total_steps:.4f}  "
          f"(~1 log cada {total_steps/max(1,total_reward):.0f} steps)")
    print(f"Sanity: {rewards_in_attack}/{int(total_reward)} rewards ocurren con attack=1  "
          f"({100*rewards_in_attack/max(1,total_reward):.0f}%)")

    # ─── 3. Acciones binarias ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("3. ACCIONES BINARIAS (frecuencia por step)")
    print("="*70)
    for a in BINARY_ACTIONS:
        c = binary_counts[a]
        print(f"  {a:10s}  {c:>10,}  ({100*c/total_steps:5.1f}% de los steps)")

    print("\nCO-OCURRENCIAS (% de steps con AMBOS flags activos):")
    interesting = sorted(cooc_counts.items(), key=lambda x: -x[1])[:10]
    for (a, b), c in interesting:
        if c > 0:
            print(f"  {a}+{b:<10s}  {c:>10,}  ({100*c/total_steps:5.1f}%)")

    print(f"\nSteps con >1 flag binario activo: {multi_flag_steps:,}  "
          f"({100*multi_flag_steps/total_steps:.1f}%) — INFO PERDIDA al serializar")

    # ─── 4. Cámara ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("4. CÁMARA (deltas en grados, |·| > 0.05°)")
    print("="*70)
    print(f"Steps con cámara activa: {cam_active_steps:,} ({100*cam_active_steps/total_steps:.1f}%)")
    print(f"Steps con cámara inactiva: {cam_inactive_steps:,} ({100*cam_inactive_steps/total_steps:.1f}%)")
    if cam_yaw_abs.size:
        print(f"|dyaw|   (activos): mean={cam_yaw_abs.mean():.2f}°  "
              f"p50={np.median(cam_yaw_abs):.2f}°  p90={np.percentile(cam_yaw_abs,90):.2f}°  "
              f"max={cam_yaw_abs.max():.2f}°")
        print(f"|dpitch| (activos): mean={cam_pitch_abs.mean():.2f}°  "
              f"p50={np.median(cam_pitch_abs):.2f}°  p90={np.percentile(cam_pitch_abs,90):.2f}°  "
              f"max={cam_pitch_abs.max():.2f}°")
    our_step_deg = math.degrees(0.15)
    print(f"\nNuestro env usa CAMERA_TURN_RAD=0.15rad = {our_step_deg:.2f}° por step.")
    print(f"  → {100*(cam_yaw_abs > our_step_deg).mean():.1f}% de los giros humanos "
          f"superan un solo step nuestro de yaw.")

    # ─── 5. Mapping a nuestras 7 acciones ──────────────────────────────────────
    print("\n" + "="*70)
    print("5. MAPPING SIMULADO → 7 acciones discretas")
    print("="*70)
    mapped_total = sum(our_action_counts.values())
    print(f"Steps mapeados:    {mapped_total:,}  ({100*mapped_total/total_steps:.1f}%)")
    print(f"Steps DESCARTADOS: {unmapped_steps:,}  ({100*unmapped_steps/total_steps:.1f}%)")
    print(f"  (back/left/right/sneak puros, o no-op)\n")
    print("Distribución resultante:")
    for a in OUR_ACTIONS:
        c = our_action_counts[a]
        print(f"  {a:22s}  {c:>10,}  ({100*c/max(1,mapped_total):5.1f}%)")

    # ─── 6. Comparar con dataset IL local ──────────────────────────────────────
    il_path = Path("data/train.jsonl")
    if il_path.exists():
        print("\n" + "="*70)
        print("6. COMPARACIÓN con dataset IL local (data/train.jsonl)")
        print("="*70)
        il_actions = Counter()
        il_total = 0
        with il_path.open() as f:
            for line in f:
                try:
                    item = json.loads(line)
                    a = item.get("action")
                    if a:
                        il_actions[a] += 1
                        il_total += 1
                except Exception:
                    continue
        print(f"Total samples IL: {il_total:,}")
        print("Distribución IL:")
        for a, c in il_actions.most_common():
            print(f"  {a:22s}  {c:>6,}  ({100*c/il_total:5.1f}%)")

    # ─── Plots ──────────────────────────────────────────────────────────────────
    _, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(ep_lengths, bins=40)
    axes[0, 0].set_title("Steps por episodio")
    axes[0, 0].set_xlabel("steps"); axes[0, 0].set_ylabel("episodios")

    axes[0, 1].hist(ep_rewards, bins=40)
    axes[0, 1].set_title("Reward (nº logs) por episodio")
    axes[0, 1].set_xlabel("logs"); axes[0, 1].set_ylabel("episodios")

    if cam_yaw_abs.size:
        axes[1, 0].hist(np.clip(cam_yaw_abs, 0, 30), bins=60)
        axes[1, 0].axvline(our_step_deg, color="r", linestyle="--",
                           label=f"nuestro step = {our_step_deg:.2f}°")
        axes[1, 0].set_title("|dyaw| en steps con cámara activa (clip 30°)")
        axes[1, 0].set_xlabel("grados"); axes[1, 0].legend()

    bar_labels = list(our_action_counts.keys())
    bar_values = [our_action_counts[k] for k in bar_labels]
    axes[1, 1].barh(bar_labels, bar_values)
    axes[1, 1].set_title("Mapping → 7 acciones (steps)")

    plt.tight_layout()
    out = OUT_DIR / "eda_summary.png"
    plt.savefig(out, dpi=110)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
