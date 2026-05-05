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
import argparse
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    if "net" in ckpt:
        return "ppo"
    raise ValueError(f"No reconozco el formato del checkpoint: claves={list(ckpt.keys())}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--algo",       type=str, default=None, choices=[None, "ppo", "sac", "hybrid"])
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
    return p.parse_args()


def load_agent(args):
    algo = args.algo or detect_algo(args.checkpoint)
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
    else:
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
                         hybrid=(algo == "hybrid"))
    print(f"Eval: episodes={args.episodes}  deterministic={args.deterministic}  "
          f"logs_to_success={logs_target}  max_steps={max_steps}")

    rewards, logs, steps, successes = [], [], [], 0
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    # Para hybrid usamos contadores por flag (no índice discreto)
    flag_counts = {f: 0 for f in (HYBRID_FLAGS if algo == "hybrid" else [])}
    cam_yaw_abs_sum = cam_pitch_abs_sum = 0.0
    hybrid_total_steps = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_logs = ep_steps = 0
        terminated = truncated = False
        print(f"\n  Episodio {ep}/{args.episodes}")

        while not (terminated or truncated):
            if algo == "ppo":
                action, _, _ = agent.select_action(obs, deterministic=args.deterministic)
            else:
                action = agent.select_action(obs, deterministic=args.deterministic)

            if algo == "hybrid":
                for i, f in enumerate(HYBRID_FLAGS):
                    if action[i] > 0.5:
                        flag_counts[f] += 1
                cam_yaw_abs_sum   += abs(float(action[N_HYBRID_FLAGS]))
                cam_pitch_abs_sum += abs(float(action[N_HYBRID_FLAGS + 1]))
                hybrid_total_steps += 1
            else:
                action_counts[action] += 1
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
