"""
Evaluación greedy/estocástica para checkpoints PPO, SAC discreto o Hybrid SAC.

Uso:
    python src/rl/visual/eval_policy.py --algo ppo    --checkpoint <ruta>.pth --episodes 15
    python src/rl/visual/eval_policy.py --algo sac    --checkpoint <ruta>.pth --episodes 15
    python src/rl/visual/eval_policy.py --algo hybrid --checkpoint <ruta>.pth --episodes 15

Detecta automáticamente el formato del checkpoint si no se especifica `--algo`
inspeccionando las claves del state_dict.
"""

import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RL_DIR / "shared"))
for _sub in ("visual/algorithms", "visual/models", "visual/train", "visual/eval"):
    sys.path.insert(0, str(_RL_DIR / _sub))

import numpy as np
import torch

from env import MinecraftRLEnv
from constants import (ACTIONS, IMG_SIZE, RL_BRIDGE_PORT, MAX_STEPS, LOGS_TO_SUCCESS,
                       HYBRID_FLAGS, N_HYBRID_FLAGS)

N_ACTIONS = len(ACTIONS)


def detect_algo(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kind = ckpt.get("kind")
    if kind in ("hybrid_sac", "hybrid_bc"):
        return "hybrid"
    if "actor" in ckpt and "q1" in ckpt:
        # Distinguir hybrid (tiene action_proj en q1) de sac discreto (no la tiene)
        q1_keys = ckpt["q1"].keys() if isinstance(ckpt["q1"], dict) else []
        if any("action_proj" in k for k in q1_keys):
            return "hybrid"
        return "sac"
    if "actor" in ckpt and "q1" not in ckpt:
        # Checkpoint BC puro de hybrid (solo actor)
        actor_keys = ckpt["actor"].keys() if isinstance(ckpt["actor"], dict) else []
        if any("flag_head" in k or "cam_mu_head" in k for k in actor_keys):
            return "hybrid"
    if "q_net" in ckpt:
        return "dqn"
    if "net" in ckpt:
        return "ppo"
    raise ValueError(f"No reconozco el formato del checkpoint: claves={list(ckpt.keys())}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Ruta del checkpoint. No es necesario con --algo random.")
    p.add_argument("--algo",       type=str, default=None,
                   choices=[None, "ppo", "sac", "hybrid", "dqn", "random"])
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--port",       type=int, default=RL_BRIDGE_PORT)
    p.add_argument("--feat-dim",   type=int, default=256)
    p.add_argument("--hidden",     type=int, default=256)
    p.add_argument("--no-state",   action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="Argmax de la política. Por defecto muestrea (recomendado para PPO/SAC).")
    p.add_argument("--logs-to-success", type=int, default=None)
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--eval-jsonl", type=str, default="data/eval/rl/eval.jsonl",
                   help="Ruta del eval.jsonl unificado (para summarize_eval.py / Cap. 8)")
    p.add_argument("--no-save",    action="store_true", help="No escribir el eval.jsonl")
    p.add_argument("--no-early-stop",  action="store_true",
                   help="Desactiva la parada por recompensa acumulada negativa (para eval con más pasos).")
    p.add_argument("--world-reset-every", type=int, default=0,
                   help="Regenerar el mundo cada N episodios (POST /world_reset). Para la "
                        "comparativa justa del Cap. 8 usar 1: cada episodio parte del mismo "
                        "mundo (seed fija) que HTN/IL, evitando la deforestación acumulada.")
    return p.parse_args()


def _world_reset(port: int):
    """Regenera el mundo del servidor gestionado por el bridge (mismo seed)."""
    import requests
    try:
        r = requests.post(f"http://localhost:{port}/world_reset", json={}, timeout=180)
        data = r.json()
        if data.get("managed"):
            print(f"  [world_reset] OK — seed={data.get('seed', '?')}")
        else:
            print("  [world_reset] aviso: bridge sin servidor gestionado (--server-dir), ignorado")
    except Exception as exc:
        print(f"  [world_reset] ERROR: {exc}")


def _append_eval(path: str, record: dict):
    """Añade un registro al eval.jsonl unificado (mismo esquema que HTN/IL)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_agent(args):
    if args.algo != "random" and not args.checkpoint:
        raise SystemExit("Se requiere --checkpoint (salvo con --algo random).")

    algo = args.algo or detect_algo(args.checkpoint)

    # Baseline aleatorio: no necesita checkpoint ni modelo.
    if algo == "random":
        img_channels = 12  # 4 frames apilados (irrelevante: random no usa la imagen)
        print(f"Algo: RANDOM  (sin checkpoint)  img_channels={img_channels}")
        return algo, None, img_channels

    use_state = not args.no_state
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    img_channels = ckpt.get("img_channels", 12)

    if algo == "ppo":
        from ppo import PPOAgent
        agent = PPOAgent(feat_dim=args.feat_dim, hidden=args.hidden,
                         use_state=use_state, img_channels=img_channels)
        agent.load(args.checkpoint)
    elif algo == "hybrid":
        from hybrid_sac import HybridSACAgent
        agent = HybridSACAgent(feat_dim=args.feat_dim, hidden=args.hidden,
                               use_state=use_state, img_channels=img_channels)
        # Para BC-only checkpoints (sin q1) cargar solo el actor
        only_actor = "q1" not in ckpt
        agent.load(args.checkpoint, only_actor=only_actor)
    elif algo == "dqn":
        from dqn import VisualDQNAgent
        agent = VisualDQNAgent(feat_dim=args.feat_dim, hidden=args.hidden,
                               use_state=use_state, img_channels=img_channels)
        agent.load(args.checkpoint)
        agent.eps_start = agent.eps_end = 0.0   # política greedy pura (argmax Q) en eval
    else:  # sac
        from sac import DiscreteSACAgent
        agent = DiscreteSACAgent(feat_dim=args.feat_dim, hidden=args.hidden,
                                 use_state=use_state, img_channels=img_channels)
        agent.load(args.checkpoint)
    print(f"Algo: {algo.upper()}  ckpt: {args.checkpoint}  step={getattr(agent, '_step', 0)}  img_channels={img_channels}")
    return algo, agent, img_channels


def run_eval(args):
    algo, agent, img_channels = load_agent(args)
    logs_target = args.logs_to_success if args.logs_to_success is not None else LOGS_TO_SUCCESS
    max_steps   = args.max_steps       if args.max_steps       is not None else MAX_STEPS
    env = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                         img_frame_stack=img_channels // 3,
                         max_steps=max_steps,
                         logs_to_success=logs_target,
                         hybrid=(algo == "hybrid"),
                         cumulative_reward_threshold=None if args.no_early_stop else -10.0)
    print(f"Eval: episodes={args.episodes}  deterministic={args.deterministic}  "
          f"logs_to_success={logs_target}  max_steps={max_steps}")

    rewards, logs, steps, successes = [], [], [], 0
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    # Para hybrid usamos contadores por flag (no índice discreto)
    flag_counts = {f: 0 for f in (HYBRID_FLAGS if algo == "hybrid" else [])}
    cam_yaw_abs_sum = cam_pitch_abs_sum = 0.0
    hybrid_total_steps = 0

    for ep in range(1, args.episodes + 1):
        # Mundo idéntico por episodio (seed fija), igual que mass_record en HTN/IL.
        # Se omite en el ep. 1 porque el bridge ya arrancó con el mundo fresco.
        if args.world_reset_every and ep > 1 and (ep - 1) % args.world_reset_every == 0:
            _world_reset(args.port)

        obs, _ = env.reset()
        ep_t0 = time.time()
        ep_reward = 0.0
        ep_logs = ep_steps = 0
        terminated = truncated = False
        ep_action_counts = {}   # histograma de acciones del episodio (mismo esquema que IL)
        print(f"\n  Episodio {ep}/{args.episodes}")

        while not (terminated or truncated):
            if algo == "ppo":
                action, _, _ = agent.select_action(obs, deterministic=args.deterministic)
            elif algo == "dqn":
                action = agent.select_action(obs)          # eps=0 → greedy (argmax Q)
            elif algo == "random":
                action = random.randrange(N_ACTIONS)
            else:
                action = agent.select_action(obs, deterministic=args.deterministic)

            if algo == "hybrid":
                for i, f in enumerate(HYBRID_FLAGS):
                    if action[i] > 0.5:
                        flag_counts[f] += 1
                        ep_action_counts[f] = ep_action_counts.get(f, 0) + 1
                cam_yaw_abs_sum   += abs(float(action[N_HYBRID_FLAGS]))
                cam_pitch_abs_sum += abs(float(action[N_HYBRID_FLAGS + 1]))
                hybrid_total_steps += 1
            else:
                action_counts[action] += 1
                name = ACTIONS[action]
                ep_action_counts[name] = ep_action_counts.get(name, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            logs_step  = sum(1 for b in info["blocks_broken"] if "log" in b)
            ep_logs   += logs_step
            if args.verbose and (logs_step or info.get("is_attacking_tree")):
                action_label = (info.get("action_name") if algo == "hybrid"
                                else ACTIONS[action])
                print(f"    [step {ep_steps:3d}] action={action_label} reward={reward:+.2f}")

        success = bool(info.get("success", False))
        if success:
            successes += 1
        print(f"    fin: success={success}  reward={ep_reward:+.2f}  "
              f"steps={ep_steps}  logs={ep_logs}")
        rewards.append(ep_reward)
        logs.append(ep_logs)
        steps.append(ep_steps)

        if not args.no_save:
            _append_eval(args.eval_jsonl, {
                "technique":      "rl",
                "ts":             datetime.now().isoformat(timespec="seconds"),
                "success_raw":    success,
                "logs_collected": int(info.get("logs_collected_total", ep_logs)),
                "logs_broken":    int(info.get("logs_broken_total", 0)),
                "target":         logs_target,
                "steps":          ep_steps,
                "time_s":         round(time.time() - ep_t0, 1),
                "action_counts":  ep_action_counts or None,
            })

    env.close()

    print(f"\n{'='*60}")
    print(f"  RESUMEN ({algo.upper()}, {args.episodes} eps, "
          f"{'greedy' if args.deterministic else 'estocastico'})")
    print(f"  success rate = {successes}/{args.episodes} ({100.0*successes/args.episodes:.1f}%)")
    print(f"  reward  media={np.mean(rewards):+.2f}  std={np.std(rewards):.2f}")
    print(f"  logs    media={np.mean(logs):.2f}  total={sum(logs)}  max={max(logs)}")
    print(f"  steps   media={np.mean(steps):.1f}  min={min(steps)}  max={max(steps)}")
    if algo == "hybrid":
        n = max(1, hybrid_total_steps)
        print(f"  flags activos:")
        for f, c in flag_counts.items():
            pct = 100.0 * c / n
            bar = "#" * int(pct / 2)
            print(f"    {f:<10} {c:5d} ({pct:5.1f}%) {bar}")
        print(f"  camera |dyaw|_avg ={cam_yaw_abs_sum / n:.4f} rad")
        print(f"         |dpitch|_avg={cam_pitch_abs_sum / n:.4f} rad")
    else:
        total = int(action_counts.sum())
        print(f"  acciones:")
        for i, c in enumerate(action_counts):
            pct = 100.0 * c / total if total else 0.0
            bar = "#" * int(pct / 2)
            print(f"    {ACTIONS[i]:<22} {c:5d} ({pct:5.1f}%) {bar}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_eval(parse_args())
