"""
Evaluación greedy/estocástica para checkpoints PPO o SAC discreto.

Uso:
    python src/rl/visual/eval_policy.py --algo ppo --checkpoint <ruta>.pth --episodes 15
    python src/rl/visual/eval_policy.py --algo sac --checkpoint <ruta>.pth --episodes 15

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
from constants import ACTIONS, IMG_SIZE, RL_BRIDGE_PORT, MAX_STEPS, LOGS_TO_SUCCESS

N_ACTIONS = len(ACTIONS)


def detect_algo(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "actor" in ckpt and "q1" in ckpt:
        return "sac"
    if "net" in ckpt:
        return "ppo"
    raise ValueError(f"No reconozco el formato del checkpoint: claves={list(ckpt.keys())}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--algo",       type=str, default=None, choices=[None, "ppo", "sac"])
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
    else:
        from sac import DiscreteSACAgent
        agent = DiscreteSACAgent(feat_dim=args.feat_dim, hidden=args.hidden,
                                 use_state=use_state, img_channels=img_channels)
    agent.load(args.checkpoint)
    print(f"Algo: {algo.upper()}  ckpt: {args.checkpoint}  step={agent._step}  img_channels={img_channels}")
    return algo, agent, img_channels


def run_eval(args):
    algo, agent, img_channels = load_agent(args)
    logs_target = args.logs_to_success if args.logs_to_success is not None else LOGS_TO_SUCCESS
    max_steps   = args.max_steps       if args.max_steps       is not None else MAX_STEPS
    env = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                         img_frame_stack=img_channels // 3,
                         max_steps=max_steps,
                         logs_to_success=logs_target)
    print(f"Eval: episodes={args.episodes}  deterministic={args.deterministic}  "
          f"logs_to_success={logs_target}  max_steps={max_steps}")

    rewards, logs, steps, successes = [], [], [], 0
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

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
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            logs_step  = sum(1 for b in info["blocks_broken"] if "log" in b)
            ep_logs   += logs_step
            if args.verbose and (logs_step or info.get("is_attacking_tree")):
                print(f"    [step {ep_steps:3d}] action={ACTIONS[action]} reward={reward:+.2f}")

        success = bool(info.get("success", False))
        if success:
            successes += 1
        print(f"    fin: success={success}  reward={ep_reward:+.2f}  "
              f"steps={ep_steps}  logs={ep_logs}")
        rewards.append(ep_reward)
        logs.append(ep_logs)
        steps.append(ep_steps)

    env.close()

    total = int(action_counts.sum())
    print(f"\n{'═'*60}")
    print(f"  RESUMEN ({algo.upper()}, {args.episodes} eps, "
          f"{'greedy' if args.deterministic else 'estocástico'})")
    print(f"  success rate = {successes}/{args.episodes} ({100.0*successes/args.episodes:.1f}%)")
    print(f"  reward  media={np.mean(rewards):+.2f}  std={np.std(rewards):.2f}")
    print(f"  logs    media={np.mean(logs):.2f}  total={sum(logs)}  max={max(logs)}")
    print(f"  steps   media={np.mean(steps):.1f}  min={min(steps)}  max={max(steps)}")
    print(f"  acciones:")
    for i, n in enumerate(action_counts):
        pct = 100.0 * n / total if total else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {ACTIONS[i]:<22} {n:5d} ({pct:5.1f}%) {bar}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    run_eval(parse_args())
