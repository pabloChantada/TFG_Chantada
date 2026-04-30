"""
Entrenamiento SAC discreto visual — Minecraft woodcutting.

Uso:
    python src/rl/visual/train_sac.py [opciones]

SAC discreto es off-policy (replay buffer + target networks como DQN) pero
añade entropía regularizada en el objetivo: maximiza retorno + α · H(π).
La temperatura α se ajusta automáticamente para alcanzar una entropía objetivo
(por defecto 98% del máximo posible para un Categórica de 7 acciones).

Ventajas frente a DQN:
- No colapsa a una política determinista trivial (entropía explícita en el loss).
- Twin Q + soft target updates → más estable, sin divergencia tipo "Q→10^33".

Métricas registradas:
    loss_q1, loss_q2, loss_actor, loss_alpha, alpha, entropy, weight_max
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
import torch as _torch

from env import MinecraftRLEnv
from metrics import RLMetrics
from sac import DiscreteSACAgent
from constants import MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS, ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="SAC discreto visual — Minecraft woodcutting")
    p.add_argument("--episodes",       type=int,   default=600)
    p.add_argument("--max-steps",      type=int,   default=MAX_STEPS)
    p.add_argument("--port",           type=int,   default=RL_BRIDGE_PORT)
    p.add_argument("--run-dir",        type=str,   default=None)
    p.add_argument("--feat-dim",       type=int,   default=256)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--lr-actor",       type=float, default=3e-4)
    p.add_argument("--lr-critic",      type=float, default=3e-4)
    p.add_argument("--lr-alpha",       type=float, default=3e-4)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--tau",            type=float, default=0.005)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--buffer-size",    type=int,   default=100_000)
    p.add_argument("--warmup",         type=int,   default=1_000)
    p.add_argument("--target-entropy-pct", type=float, default=0.98,
                   help="Entropía objetivo como fracción de log(n_actions). "
                        "0.98 = exploración alta; 0.5 = explotación temprana.")
    p.add_argument("--reward-clip",    type=float, default=1.0)
    p.add_argument("--no-reward-clip", dest="reward_clip", action="store_const", const=None)
    p.add_argument("--img-frame-stack",type=int,   default=4)
    p.add_argument("--no-state",       action="store_true")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--reset-world-every", type=int, default=20)
    p.add_argument("--checkpoint-every",  type=int, default=10)
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / f"sac_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def write_config(run_dir, args):
    import subprocess
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
            "algorithm":  "SAC-discrete",
        },
        "train": vars(args),
        "architecture": {
            "type":            "SAC discrete (twin Q + soft target + auto α)",
            "img_size":        IMG_SIZE,
            "img_frame_stack": args.img_frame_stack,
            "img_channels":    3 * args.img_frame_stack,
            "feat_dim":        args.feat_dim,
            "hidden":          args.hidden,
            "use_state":       not args.no_state,
            "state_dim":       STATE_DIM,
            "state_keys":      STATE_KEYS,
            "n_actions":       len(ACTIONS),
            "actions":         ACTIONS,
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


def maybe_world_reset(port: int, ep: int, period: int):
    if not period or ep % period != 0:
        return
    print(f"\n  [world_reset] Reiniciando mundo tras episodio {ep}...")
    try:
        r    = _requests.post(f"http://localhost:{port}/world_reset", json={}, timeout=180)
        data = r.json()
        if data.get("managed"):
            print(f"  [world_reset] OK — seed={data.get('seed', '?')}")
        else:
            print(f"  [world_reset] Agente sin servidor gestionado, ignorado.")
    except Exception as exc:
        print(f"  [world_reset] ERROR: {exc}")


def train(args):
    run_dir = args.run_dir or make_run_dir()
    write_config(run_dir, args)

    print(f"Run dir:  {run_dir}")
    print(f"SAC: lr_actor={args.lr_actor}  lr_critic={args.lr_critic}  "
          f"gamma={args.gamma}  tau={args.tau}  batch={args.batch_size}  "
          f"warmup={args.warmup}  target_ent_pct={args.target_entropy_pct}\n")

    env     = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                             img_frame_stack=args.img_frame_stack,
                             max_steps=args.max_steps)
    metrics = RLMetrics(run_dir)
    use_state = not args.no_state

    agent = DiscreteSACAgent(
        feat_dim     = args.feat_dim,
        hidden       = args.hidden,
        use_state    = use_state,
        img_channels = 3 * args.img_frame_stack,
        lr_actor     = args.lr_actor,
        lr_critic    = args.lr_critic,
        lr_alpha     = args.lr_alpha,
        gamma        = args.gamma,
        tau          = args.tau,
        batch_size   = args.batch_size,
        buffer_size  = args.buffer_size,
        warmup_steps = args.warmup,
        target_entropy_pct = args.target_entropy_pct,
        reward_clip  = args.reward_clip,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "sac_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido — checkpoint en: {ckpt}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    train_start = time.time()
    global_step = agent._step

    for ep in range(1, args.episodes + 1):
        ep_start          = time.time()
        ep_reward         = 0.0
        ep_steps          = 0
        ep_logs_broken    = 0
        ep_logs_collected = 0
        ep_losses_q       = []
        ep_losses_actor   = []
        ep_alphas         = []
        ep_entropies      = []
        action_counts     = {a: 0 for a in ACTIONS}
        terminated = truncated = False

        elapsed = (time.time() - train_start) / 60
        print(f"\n{'─'*60}")
        print(f"  SAC Episodio {ep}/{args.episodes}  |  α={agent.alpha:.4f}  "
              f"buffer={len(agent.buffer)}  elapsed={elapsed:.1f}min")
        print(f"{'─'*60}")

        obs, _ = env.reset()

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            stats = agent.step(obs, action, reward, next_obs, done)

            if stats:
                ep_losses_q.append((stats["loss_q1"] + stats["loss_q2"]) / 2)
                ep_losses_actor.append(stats["loss_actor"])
                ep_alphas.append(stats["alpha"])
                ep_entropies.append(stats["entropy"])

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
                loss        = (stats["loss_q1"] + stats["loss_q2"]) / 2 if stats else None,
                epsilon     = 0.0,  # SAC no usa epsilon-greedy
                action      = info["action_name"],
                extra       = {"alpha": round(agent.alpha, 4),
                               "logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step},
            )

            # Verbose por step: state + acción + α + entropía instantánea
            cur_state = obs["state"][-STATE_DIM:]
            state_str = "  ".join(f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, cur_state))
            ent_str  = f"H={stats['entropy']:.3f}"  if stats else "H=---"
            loss_str = f"lossQ={stats['loss_q1']:.3f}" if stats else "lossQ=---"
            print(f"  [step {ep_steps:3d}]  state: {state_str}  "
                  f"action={info['action_name']}  α={agent.alpha:.3f}  {ent_str}  {loss_str}")

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

        ep_time   = time.time() - ep_start
        avg_q     = sum(ep_losses_q)     / len(ep_losses_q)     if ep_losses_q     else 0.0
        avg_actor = sum(ep_losses_actor) / len(ep_losses_actor) if ep_losses_actor else 0.0
        avg_alpha = sum(ep_alphas)       / len(ep_alphas)       if ep_alphas       else agent.alpha
        avg_ent   = sum(ep_entropies)    / len(ep_entropies)    if ep_entropies    else 0.0
        success   = bool(info.get("success", False))
        end_reason = ("ÉXITO" if success else
                      "terminado" if terminated else "truncado (timeout)")

        w_norms = agent.weight_max_abs()
        w_max   = max(w_norms.values()) if w_norms else 0.0

        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken,
                            extra={"avg_loss":       round(avg_q, 6),
                                   "avg_loss_actor": round(avg_actor, 6),
                                   "alpha":          round(avg_alpha, 4),
                                   "entropy":        round(avg_ent, 4),
                                   "epsilon":        0.0,
                                   "ep_time_s":      round(ep_time, 2),
                                   "logs_collected": ep_logs_collected,
                                   "success":        success,
                                   "weight_max":     round(w_max, 4)})

        print(f"\n  Fin: {end_reason}  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}")
        print(f"  loss_Q={avg_q:.4f}  loss_actor={avg_actor:+.4f}  "
              f"α={avg_alpha:.4f}  H={avg_ent:.3f}  ||w||_max={w_max:.2f}  "
              f"tiempo={ep_time:.1f}s")
        print(f"  Acciones: " + "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        if w_max > 50.0:
            print(f"  ⚠ DIVERGENCIA: ||weight||_max={w_max:.2f} (umbral=50).")

        if ep % args.checkpoint_every == 0:
            ckpt = Path(run_dir) / f"sac_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  → checkpoint guardado: {ckpt}")

        maybe_world_reset(args.port, ep, args.reset_world_every)

    agent.save(str(Path(run_dir) / "sac_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento SAC finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
