"""
Entrenamiento PPO visual — Minecraft woodcutting.

Uso:
    python src/rl/visual/train_ppo.py [opciones]

PPO es on-policy: cada update consume el rollout (rollout_steps transiciones)
y se vacía. Con steps lentos (Mineflayer ~0.5-3s/step) tiene menor eficiencia
muestral que DQN, pero suele ser más estable y sin divergencias de Q.

Métricas adicionales registradas:
    policy_loss, value_loss, entropy, approx_kl, clip_frac, weight_max
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
from ppo import PPOAgent
from constants import MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS, ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="PPO visual — Minecraft woodcutting")
    p.add_argument("--episodes",       type=int,   default=600)
    p.add_argument("--max-steps",      type=int,   default=MAX_STEPS)
    p.add_argument("--port",           type=int,   default=RL_BRIDGE_PORT)
    p.add_argument("--run-dir",        type=str,   default=None)
    p.add_argument("--feat-dim",       type=int,   default=256)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--gae-lambda",     type=float, default=0.95)
    p.add_argument("--clip-eps",       type=float, default=0.2)
    p.add_argument("--ent-coef",       type=float, default=0.01)
    p.add_argument("--vf-coef",        type=float, default=0.5)
    p.add_argument("--max-grad",       type=float, default=0.5)
    p.add_argument("--epochs",         type=int,   default=4)
    p.add_argument("--minibatch",      type=int,   default=64)
    p.add_argument("--rollout-steps",  type=int,   default=1024,
                   help="Transiciones por update PPO. Más alto = más estable, "
                        "más lento de actualizar.")
    p.add_argument("--img-frame-stack",type=int,   default=4)
    p.add_argument("--no-state",       action="store_true")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--reset-world-every", type=int, default=20)
    p.add_argument("--checkpoint-every",  type=int, default=10)
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / f"ppo_{ts}"
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
            "algorithm":  "PPO",
        },
        "train": vars(args),
        "architecture": {
            "type":            "PPO (actor-critic, clipped objective + GAE)",
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
    print(f"PPO: lr={args.lr}  gamma={args.gamma}  gae_lambda={args.gae_lambda}  "
          f"clip_eps={args.clip_eps}  rollout={args.rollout_steps}  "
          f"epochs={args.epochs}  minibatch={args.minibatch}\n")

    env     = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                             img_frame_stack=args.img_frame_stack,
                             max_steps=args.max_steps)
    metrics = RLMetrics(run_dir)
    use_state = not args.no_state

    agent = PPOAgent(
        feat_dim     = args.feat_dim,
        hidden       = args.hidden,
        use_state    = use_state,
        img_channels = 3 * args.img_frame_stack,
        lr           = args.lr,
        gamma        = args.gamma,
        gae_lambda   = args.gae_lambda,
        clip_eps     = args.clip_eps,
        ent_coef     = args.ent_coef,
        vf_coef      = args.vf_coef,
        max_grad     = args.max_grad,
        epochs       = args.epochs,
        minibatch    = args.minibatch,
        rollout_steps= args.rollout_steps,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "ppo_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido — checkpoint en: {ckpt}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    train_start = time.time()
    global_step = agent._step

    obs, _      = env.reset()
    last_stats  = None

    for ep in range(1, args.episodes + 1):
        ep_start          = time.time()
        ep_reward         = 0.0
        ep_steps          = 0
        ep_logs_broken    = 0
        ep_logs_collected = 0
        action_counts     = {a: 0 for a in ACTIONS}
        terminated = truncated = False

        elapsed = (time.time() - train_start) / 60
        print(f"\n{'─'*60}")
        print(f"  PPO Episodio {ep}/{args.episodes}  |  buffer={agent.buffer.idx}/{agent.rollout_steps}  "
              f"elapsed={elapsed:.1f}min")
        print(f"{'─'*60}")

        while not (terminated or truncated):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # NOTA: a diferencia de los agentes off-policy (DQN/SAC), aquí se usa
            # done = terminated OR truncated a propósito. El buffer del PPO abarca
            # varios episodios y GAE usa este flag para CORTAR la cadena de ventajas
            # en los límites de episodio; ponerlo a 'terminated' filtraría la
            # ventaja al episodio siguiente. El bootstrap correcto en truncamiento
            # exigiría guardar V(s) del estado truncado aparte (pendiente).
            done = terminated or truncated

            agent.store(obs, action, log_prob, value, reward, done)

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
                loss        = None,
                epsilon     = 0.0,  # PPO no usa epsilon-greedy; se mantiene el campo por compat
                action      = info["action_name"],
                extra       = {"value": round(value, 4),
                               "log_prob": round(log_prob, 4),
                               "logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step},
            )

            # Verbose por step: state + acción + value + log_prob
            cur_state = obs["state"][-STATE_DIM:]
            state_str = "  ".join(f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, cur_state))
            print(f"  [step {ep_steps:3d}]  state: {state_str}  "
                  f"action={info['action_name']}  V={value:+.3f}  logπ={log_prob:+.3f}")

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

            # Update si el rollout está lleno (puede ocurrir a mitad de episodio).
            if agent.buffer.full():
                stats = agent.update(next_obs, done)
                last_stats = stats
                print(f"  [update]  policy_loss={stats['policy_loss']:+.4f}  "
                      f"value_loss={stats['value_loss']:.4f}  "
                      f"entropy={stats['entropy']:.3f}  "
                      f"kl={stats['approx_kl']:.4f}  clip={stats['clip_frac']:.3f}")

        ep_time = time.time() - ep_start
        success = bool(info.get("success", False))
        end_reason = ("ÉXITO" if success else
                      "terminado" if terminated else "truncado (timeout)")

        w_norms = agent.weight_max_abs()
        w_max   = max(w_norms.values()) if w_norms else 0.0

        ep_extra = {
            "ep_time_s":      round(ep_time, 2),
            "logs_collected": ep_logs_collected,
            "success":        success,
            "weight_max":     round(w_max, 4),
            "epsilon":        0.0,
            "avg_loss":       round(last_stats["value_loss"], 6) if last_stats else 0.0,
        }
        if last_stats:
            ep_extra["policy_loss"] = round(last_stats["policy_loss"], 6)
            ep_extra["value_loss"]  = round(last_stats["value_loss"],  6)
            ep_extra["entropy"]     = round(last_stats["entropy"],     4)
            ep_extra["approx_kl"]   = round(last_stats["approx_kl"],   6)
            ep_extra["clip_frac"]   = round(last_stats["clip_frac"],   4)

        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken, extra=ep_extra)

        print(f"\n  Fin: {end_reason}  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}  "
              f"||w||_max={w_max:.2f}  tiempo={ep_time:.1f}s")
        print(f"  Acciones: " + "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        if w_max > 50.0:
            print(f"  ⚠ DIVERGENCIA: ||weight||_max={w_max:.2f} (umbral=50).")

        if ep % args.checkpoint_every == 0:
            ckpt = Path(run_dir) / f"ppo_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  → checkpoint guardado: {ckpt}")

        maybe_world_reset(args.port, ep, args.reset_world_every)
        if ep < args.episodes:
            obs, _ = env.reset()  # nuevo episodio: reset del bot

    agent.save(str(Path(run_dir) / "ppo_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento PPO finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
