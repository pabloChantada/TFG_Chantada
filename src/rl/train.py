"""
Script de entrenamiento RL — DQN — Minecraft woodcutting.

Uso:
    python src/rl/train.py [opciones]

Opciones:
    --episodes N        Nº de episodios (defecto: 500)
    --port PORT         Puerto del bridge Node.js (defecto: 8766)
    --run-dir DIR       Directorio de salida (defecto: src/rl/runs/<timestamp>)
    --eval-every N      Guardar métricas cada N episodios (defecto: 50)
    --seed N            Semilla aleatoria (defecto: 42)
    --hidden N          Neuronas por capa oculta del DQN (defecto: 128)
    --lr LR             Learning rate (defecto: 1e-3)
    --gamma G           Factor de descuento (defecto: 0.99)
    --eps-decay N       Steps hasta epsilon mínimo (defecto: 5000)
    --target-update N   Steps entre sync target network (defecto: 200)
    --batch-size N      Batch size (defecto: 64)
    --resume PATH       Cargar checkpoint previo y continuar
"""

import argparse
import signal
import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'il'))

from env import MinecraftRLEnv
from metrics import RLMetrics
from dqn import DQNAgent
from constants import MAX_STEPS, RL_BRIDGE_PORT


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DQN Minecraft woodcutting")
    p.add_argument("--episodes",      type=int,   default=120,
                   help="Nº de episodios (~1h a 300ms/step con MAX_STEPS=100)")
    p.add_argument("--port",          type=int,   default=RL_BRIDGE_PORT)
    p.add_argument("--run-dir",       type=str,   default=None)
    p.add_argument("--eval-every",    type=int,   default=50)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--eps-decay",     type=int,   default=5_000)
    p.add_argument("--target-update", type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--frame-stack",   type=int,   default=1,
                   help="Nº de frames apilados en la observación (defecto: 1 = sin stacking)")
    p.add_argument("--resume",        type=str,   default=None)
    return p.parse_args()


def make_run_dir(base: str = "src/rl/runs") -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = Path(base) / ts
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


# ── Loop de entrenamiento ─────────────────────────────────────────────────────

def train(args):
    run_dir = args.run_dir or make_run_dir()
    from constants import STATE_DIM
    obs_dim = STATE_DIM * args.frame_stack

    print(f"Run dir:  {run_dir}")
    print(f"Episodes: {args.episodes}  |  bridge: localhost:{args.port}")
    print(f"DQN: hidden={args.hidden}  lr={args.lr}  gamma={args.gamma}  "
          f"eps_decay={args.eps_decay}  target_update={args.target_update}")
    print(f"Obs: state_dim={STATE_DIM}  frame_stack={args.frame_stack}  "
          f"→ input={obs_dim}\n")

    env     = MinecraftRLEnv(bridge_port=args.port, frame_stack=args.frame_stack)
    metrics = RLMetrics(run_dir)

    agent = DQNAgent(
        state_dim     = obs_dim,
        hidden        = args.hidden,
        lr            = args.lr,
        gamma         = args.gamma,
        eps_decay     = args.eps_decay,
        target_update = args.target_update,
        batch_size    = args.batch_size,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "dqn_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido — checkpoint guardado en: {ckpt}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    from constants import ACTIONS

    train_start = time.time()

    for ep in range(1, args.episodes + 1):
        ep_start = time.time()
        elapsed_total = ep_start - train_start

        print(f"\n{'─'*60}")
        print(f"  Episodio {ep}/{args.episodes}  |  eps={agent.epsilon:.3f}  "
              f"buffer={len(agent.buffer)}  "
              f"elapsed={elapsed_total/60:.1f}min")
        print(f"{'─'*60}")

        obs, _     = env.reset()
        ep_reward  = 0.0
        ep_steps   = 0
        ep_logs    = 0
        ep_losses  = []
        action_counts = {a: 0 for a in ACTIONS}
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            loss = agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                ep_losses.append(loss)

            action_counts[info["action_name"]] += 1
            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1
            logs_this_step = sum(1 for b in info["blocks_broken"] if "log" in b)
            ep_logs   += logs_this_step

            # Verbose por step: solo eventos interesantes
            if logs_this_step:
                print(f"  [step {ep_steps:3d}]  *** LOG ROTO ***  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif info.get("is_attacking_tree"):
                print(f"  [step {ep_steps:3d}]  atacando árbol  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")

        # ── Resumen del episodio ──────────────────────────────────────────────
        ep_time    = time.time() - ep_start
        avg_loss   = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
        end_reason = "terminado" if terminated else "truncado (timeout)"
        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs,
                            extra={"avg_loss":  round(avg_loss, 6),
                                   "epsilon":   round(agent.epsilon, 4),
                                   "ep_time_s": round(ep_time, 2)})

        print(f"\n  Fin: {end_reason}")
        print(f"  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs={ep_logs}  loss={avg_loss:.4f}  tiempo={ep_time:.1f}s")
        print(f"  Acciones: " +
              "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        if ep % args.eval_every == 0:
            ckpt = Path(run_dir) / f"dqn_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  → checkpoint guardado: {ckpt}")

    # Checkpoint y métricas finales
    agent.save(str(Path(run_dir) / "dqn_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
