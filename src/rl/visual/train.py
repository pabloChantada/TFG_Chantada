"""
Entrenamiento DQN visual — Minecraft woodcutting.

Uso:
    python src/rl/visual/train.py [opciones]

Opciones:
    --episodes N        Nº de episodios (defecto: 120)
    --port PORT         Puerto del bridge Node.js (defecto: 8766)
    --run-dir DIR       Directorio de salida (defecto: src/rl/visual/runs/<timestamp>)
    --feat-dim N        Dimensión de features CNN (defecto: 256)
    --hidden N          Neuronas MLP del Q-network (defecto: 256)
    --lr LR             Learning rate (defecto: 1e-4)
    --gamma G           Factor de descuento (defecto: 0.99)
    --eps-decay N       Steps hasta epsilon mínimo (defecto: 150000)
    --target-update N   Steps entre sync target network (defecto: 2500)
    --batch-size N      Batch size (defecto: 64)
    --buffer-size N     Tamaño del replay buffer (defecto: 100000)
    --warmup N          Steps mínimos antes de empezar a entrenar (defecto: 5000)
    --reward-clip V     Clip de reward a [-V, +V] para estabilidad (defecto: 1.0)
    --no-reward-clip    Desactivar reward clipping
    --img-frame-stack K Apilar los últimos K frames en el eje de canales (defecto: 4)
    --resume PATH       Cargar checkpoint previo y continuar
"""

import json
import sys
import signal
import time
import argparse
from datetime import datetime
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests as _requests

from env import MinecraftRLEnv
from metrics import RLMetrics
from dqn import VisualDQNAgent
from constants import MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS, ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="DQN visual — Minecraft woodcutting")
    p.add_argument("--episodes",      type=int,   default=120)
    p.add_argument("--max-steps",     type=int,   default=MAX_STEPS)
    p.add_argument("--port",          type=int,   default=RL_BRIDGE_PORT)
    p.add_argument("--run-dir",       type=str,   default=None)
    p.add_argument("--eval-every",    type=int,   default=10)
    p.add_argument("--feat-dim",      type=int,   default=256)
    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--eps-decay",     type=int,   default=100_000)
    p.add_argument("--target-update", type=int,   default=2_500)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--buffer-size",   type=int,   default=100_000)
    p.add_argument("--warmup",        type=int,   default=3_000)
    p.add_argument("--reward-clip",   type=float, default=1.0,
                   help="Clip de reward a [-V, +V] para estabilidad (0 o --no-reward-clip lo desactiva)")
    p.add_argument("--no-reward-clip", dest="reward_clip", action="store_const", const=None,
                   help="Desactivar reward clipping")
    p.add_argument("--img-frame-stack", type=int, default=4,
                   help="Apilar los últimos K frames en el eje de canales (defecto: 4)")
    p.add_argument("--no-state",           action="store_true",
                   help="Usar solo imagen, sin fusionar vector de estado")
    p.add_argument("--resume",             type=str, default=None)
    p.add_argument("--reset-world-every",  type=int, default=0)
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / ts
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def train(args):
    run_dir = args.run_dir or make_run_dir()

    import subprocess
    import torch as _torch
    from constants import (
        ACTIONS, STATE_KEYS, STATE_DIM, IMG_SIZE,
        CAMERA_TURN_RAD, CAMERA_PITCH_RAD,
        REWARD_BREAK_LOG, REWARD_COLLECT_LOG, REWARD_HIT_TREE,
        REWARD_LOOK_AT_LOG, REWARD_APPROACH, REWARD_STEP,
        REWARD_WRONG_BLOCK,
        REWARD_DONE_PENALTY, REWARD_SUCCESS, LOGS_TO_SUCCESS,
        MAX_STEPS, CUMULATIVE_REWARD_THRESHOLD,
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    cfg = {
        "meta": {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit,
            "torch":      _torch.__version__,
            "device":     "cuda" if _torch.cuda.is_available() else "cpu",
        },
        "train": vars(args),
        "architecture": {
            "type":             "Double-DQN",
            "img_size":         IMG_SIZE,
            "img_frame_stack":  args.img_frame_stack,
            "img_channels":     3 * args.img_frame_stack,
            "feat_dim":         args.feat_dim,
            "hidden":           args.hidden,
            "use_state":        not args.no_state,
            "state_dim":        STATE_DIM,
            "state_keys":       STATE_KEYS,
            "n_actions":        len(ACTIONS),
            "actions":          ACTIONS,
            "cnn_channels":     [3 * args.img_frame_stack, 32, 64, 128],
            "norm":             "GroupNorm(8)",
            "pooling":          "MaxPool2d(2)x3",
            "fusion":           "sum(img_feat, state_proj)" if not args.no_state else "img_feat only",
            "grad_clip":        10.0,
            "optimizer":        "Adam",
        },
        "env": {
            "max_steps":                  MAX_STEPS,
            "cumulative_reward_threshold": CUMULATIVE_REWARD_THRESHOLD,
            "logs_to_success":            LOGS_TO_SUCCESS,
            "camera_turn_rad":            CAMERA_TURN_RAD,
            "camera_pitch_rad":           CAMERA_PITCH_RAD,
        },
        "rewards": {
            "break_log":    REWARD_BREAK_LOG,
            "collect_log":  REWARD_COLLECT_LOG,
            "hit_tree":     REWARD_HIT_TREE,
            "look_at_log":  REWARD_LOOK_AT_LOG,
            "approach":     REWARD_APPROACH,
            "step":         REWARD_STEP,
            "wrong_block":  REWARD_WRONG_BLOCK,
            "done_penalty": REWARD_DONE_PENALTY,
            "success":      REWARD_SUCCESS,
        },
    }
    Path(run_dir, "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"Run dir:  {run_dir}")
    print(f"Episodes: {args.episodes}  |  bridge: localhost:{args.port}")
    print(f"DQN visual: feat_dim={args.feat_dim}  hidden={args.hidden}  "
          f"lr={args.lr}  gamma={args.gamma}  "
          f"eps_decay={args.eps_decay}  target_update={args.target_update}  "
          f"batch={args.batch_size}  buffer={args.buffer_size}  warmup={args.warmup}  "
          f"reward_clip={args.reward_clip}  img_frame_stack={args.img_frame_stack}\n")

    env     = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                             img_frame_stack=args.img_frame_stack,
                             max_steps=args.max_steps)
    metrics = RLMetrics(run_dir)
    use_state = not args.no_state
    print(f"Modo: {'imagen + estado' if use_state else 'solo imagen'}\n")

    agent   = VisualDQNAgent(
        feat_dim      = args.feat_dim,
        hidden        = args.hidden,
        use_state     = use_state,
        lr            = args.lr,
        gamma         = args.gamma,
        eps_decay     = args.eps_decay,
        target_update = args.target_update,
        batch_size    = args.batch_size,
        buffer_size   = args.buffer_size,
        warmup_steps  = args.warmup,
        reward_clip   = args.reward_clip,
        img_channels  = 3 * args.img_frame_stack,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "dqn_visual_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido — checkpoint guardado en: {ckpt}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    train_start = time.time()
    global_step = agent._step  # continuar desde el step del checkpoint si aplica

    for ep in range(1, args.episodes + 1):
        ep_start      = time.time()
        elapsed_total = ep_start - train_start

        print(f"\n{'─'*60}")
        print(f"  Episodio {ep}/{args.episodes}  |  eps={agent.epsilon:.3f}  "
              f"buffer={len(agent.buffer)}  elapsed={elapsed_total/60:.1f}min")
        print(f"{'─'*60}")

        obs, _            = env.reset()
        ep_reward         = 0.0
        ep_steps          = 0
        ep_logs_broken    = 0
        ep_logs_collected = 0
        ep_losses         = []
        action_counts     = {a: 0 for a in ACTIONS}
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            loss = agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                ep_losses.append(loss)

            action_counts[info["action_name"]] += 1
            obs          = next_obs
            ep_reward   += reward
            ep_steps    += 1
            global_step += 1

            broken_this_step    = sum(1 for b in info["blocks_broken"] if "log" in b)
            collected_this_step = info.get("logs_collected", 0)
            ep_logs_broken     += broken_this_step
            ep_logs_collected  += collected_this_step

            metrics.log_step(
                global_step = global_step,
                episode     = ep,
                reward      = reward,
                loss        = loss,
                epsilon     = agent.epsilon,
                action      = info["action_name"],
                extra       = {"logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step},
            )

            # State en cada step
            cur_state = obs["state"][-STATE_DIM:]
            state_str = "  ".join(f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, cur_state))
            print(f"  [step {ep_steps:3d}]  state: {state_str}  action={info['action_name']}")

            if broken_this_step:
                print(f"  [step {ep_steps:3d}]  *** LOG ROTO ***  "
                      f"bloques={info['blocks_broken']}  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif collected_this_step:
                print(f"  [step {ep_steps:3d}]  LOG RECOGIDO (+{collected_this_step})  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif info.get("is_attacking_tree"):
                print(f"  [step {ep_steps:3d}]  HIT_TREE ({info.get('attacked_block', '?')})  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif info.get("attacked_block"):
                print(f"  [step {ep_steps:3d}]  attack→{info['attacked_block']} (no log)  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")

        ep_time    = time.time() - ep_start
        avg_loss   = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
        success    = bool(info.get("success", False))
        if success:
            end_reason = "ÉXITO (tronco recogido)"
        elif terminated:
            end_reason = "terminado (muerte / reward threshold)"
        else:
            end_reason = "truncado (timeout)"
        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken,
                            extra={"avg_loss":       round(avg_loss, 6),
                                   "epsilon":        round(agent.epsilon, 4),
                                   "ep_time_s":      round(ep_time, 2),
                                   "logs_collected": ep_logs_collected,
                                   "success":        success})

        print(f"\n  Fin: {end_reason}")
        print(f"  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}  "
              f"loss={avg_loss:.4f}  tiempo={ep_time:.1f}s")
        print(f"  Acciones: " + "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        reset_n = args.reset_world_every or args.eval_every
        if ep % reset_n == 0:
            ckpt = Path(run_dir) / f"dqn_visual_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  → checkpoint guardado: {ckpt}")

        if args.reset_world_every and ep % args.reset_world_every == 0 and ep < args.episodes:
            print(f"\n  [world_reset] Reiniciando mundo tras episodio {ep}...")
            try:
                r    = _requests.post(f"http://localhost:{args.port}/world_reset",
                                      json={}, timeout=180)
                data = r.json()
                if data.get("managed"):
                    print(f"  [world_reset] OK — seed={data.get('seed', '?')}")
                else:
                    print(f"  [world_reset] Agente sin servidor gestionado, ignorado.")
            except Exception as exc:
                print(f"  [world_reset] ERROR: {exc}  (continúa sin reset)")

    agent.save(str(Path(run_dir) / "dqn_visual_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
