"""
Entrenamiento Hybrid SAC visual — espacio de acciones tipo MineRL.

Acción híbrida = flags binarios concurrentes (forward, jump, sprint, attack)
                  + cámara continua 2D (dyaw, dpitch en radianes).

Pipeline soportado:
    1. RL desde cero:                python train_hybrid_sac.py
    2. RL warm-started con BC:       python train_hybrid_sac.py --bc-ckpt PATH
    3. RL + buffer pre-cargado:      añade --preload-demos DATASET_ROOT
                                     (carga (img, state=0, action_vec) de las demos
                                      con reward del dataset)

Métricas registradas (compatibles con metrics.py):
    loss_q1, loss_q2, loss_actor, loss_alpha, alpha, entropy, weight_max
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests as _requests
import numpy as np
import torch as _torch

from env import MinecraftRLEnv
from metrics import RLMetrics
from hybrid_sac import HybridSACAgent
from constants import (
    MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS,
    HYBRID_FLAGS, N_HYBRID_FLAGS, CAMERA_MAX_RAD,
)


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid SAC visual — Minecraft woodcutting")
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
    p.add_argument("--buffer-size",    type=int,   default=200_000)
    p.add_argument("--warmup",         type=int,   default=1_000)
    p.add_argument("--target-entropy", type=float, default=None,
                   help="Si None, usa el default (0.7*log2*N_FLAGS + H_gauss(sigma=0.3*MAX)).")
    p.add_argument("--reward-clip",    type=float, default=1.0)
    p.add_argument("--no-reward-clip", dest="reward_clip", action="store_const", const=None)
    p.add_argument("--img-frame-stack",type=int,   default=4)
    p.add_argument("--no-state",       action="store_true")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--bc-ckpt",        type=str,   default=None,
                   help="Checkpoint BC del actor (cargado solo en self.actor).")
    p.add_argument("--preload-demos",  type=str,   default=None,
                   help="Pre-carga el replay buffer con demos del dataset MineRL "
                        "indicado (ruta a carpeta con subdirs *.npz + recording.mp4).")
    p.add_argument("--preload-max",    type=int,   default=50_000,
                   help="Máximo nº de transiciones de demo a pre-cargar.")
    p.add_argument("--reset-world-every", type=int, default=20)
    p.add_argument("--checkpoint-every",  type=int, default=10)
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / f"hybrid_sac_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def write_config(run_dir, args):
    import subprocess
    from constants import (
        IMG_SIZE,
        REWARD_BREAK_LOG, REWARD_COLLECT_LOG, REWARD_HIT_TREE,
        REWARD_LOOK_AT_LOG, REWARD_APPROACH, REWARD_STEP,
        REWARD_WRONG_BLOCK, REWARD_DONE_PENALTY, REWARD_SUCCESS,
        LOGS_TO_SUCCESS, MAX_STEPS, CUMULATIVE_REWARD_THRESHOLD,
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
            "algorithm":  "Hybrid-SAC",
        },
        "train": vars(args),
        "architecture": {
            "type":            "Hybrid SAC (Bernoulli×Gaussiana squashed, twin Q + soft target + auto α)",
            "img_size":        IMG_SIZE,
            "img_frame_stack": args.img_frame_stack,
            "img_channels":    3 * args.img_frame_stack,
            "feat_dim":        args.feat_dim,
            "hidden":          args.hidden,
            "use_state":       not args.no_state,
            "state_dim":       STATE_DIM,
            "state_keys":      STATE_KEYS,
            "hybrid_flags":    HYBRID_FLAGS,
            "camera_max_rad":  CAMERA_MAX_RAD,
            "action_vec_dim":  N_HYBRID_FLAGS + 2,
        },
        "env": {
            "max_steps":                  MAX_STEPS,
            "cumulative_reward_threshold": CUMULATIVE_REWARD_THRESHOLD,
            "logs_to_success":            LOGS_TO_SUCCESS,
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
            print(f"  [world_reset] OK -- seed={data.get('seed', '?')}")
        else:
            print(f"  [world_reset] Agente sin servidor gestionado, ignorado.")
    except Exception as exc:
        print(f"  [world_reset] ERROR: {exc}")


def preload_demo_buffer(agent: HybridSACAgent, dataset_root: str, max_steps: int,
                        img_frame_stack: int):
    """Pre-carga el replay buffer con transiciones del dataset MineRL.

    Usa el mismo pipeline que bc_pretrain (cv2 + frame stack 4 + ImageNet norm)
    pero sin loss/gradiente: solo empuja transiciones al buffer.
    State se rellena con ceros (MineRL no expone tree_visible/etc., el SAC
    online aprenderá a aprovechar el state real más adelante).
    """
    minerl_dir = _RL_DIR / "minerl"
    sys.path.insert(0, str(minerl_dir))
    from map_actions import list_episodes, load_episode_actions, load_episode_meta
    from bc_pretrain import _preprocess_frame
    import cv2

    eps = list_episodes(dataset_root)
    print(f"\n[demos] Pre-cargando hasta {max_steps:,} transiciones de {len(eps)} episodios...")
    pushed = 0
    state_zero = np.zeros(STATE_DIM, dtype=np.float32)

    for ep in eps:
        if pushed >= max_steps:
            break
        actions = load_episode_actions(ep / "rendered.npz")     # (T, 6)
        meta    = load_episode_meta(ep / "rendered.npz")
        rewards = meta["reward"]
        dones   = meta["done"]
        T = actions.shape[0]

        cap = cv2.VideoCapture(str(ep / "recording.mp4"))
        try:
            stack = []          # frames CHW
            prev_obs_img = None
            for t in range(T):
                if pushed >= max_steps:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                proc = _preprocess_frame(frame)
                if not stack:
                    stack = [proc] * img_frame_stack
                else:
                    stack.append(proc)
                    if len(stack) > img_frame_stack:
                        stack.pop(0)
                stacked = np.concatenate(stack, axis=0)         # (3*FS, H, W)

                if prev_obs_img is not None:
                    # Empujar (s_{t-1}, a_{t-1}, r_{t-1}, s_t, done_{t-1})
                    agent.buffer.push(
                        prev_obs_img, state_zero, actions[t-1], rewards[t-1],
                        stacked, state_zero, bool(dones[t-1]),
                    )
                    pushed += 1
                prev_obs_img = stacked.copy()
        finally:
            cap.release()

        if pushed % 5000 < (T or 1):
            print(f"  [demos] {pushed:,} / {max_steps:,}")
    print(f"[demos] OK: {pushed:,} transiciones precargadas.\n")


def train(args):
    run_dir = args.run_dir or make_run_dir()
    write_config(run_dir, args)

    print(f"Run dir:  {run_dir}")
    print(f"HybridSAC: lr_actor={args.lr_actor}  lr_critic={args.lr_critic}  "
          f"gamma={args.gamma}  tau={args.tau}  batch={args.batch_size}  "
          f"warmup={args.warmup}\n")

    env     = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                             img_frame_stack=args.img_frame_stack,
                             max_steps=args.max_steps, hybrid=True)
    metrics = RLMetrics(run_dir)
    use_state = not args.no_state

    agent = HybridSACAgent(
        feat_dim     = args.feat_dim,
        hidden       = args.hidden,
        use_state    = use_state,
        img_channels = 3 * args.img_frame_stack,
        n_flags      = N_HYBRID_FLAGS,
        camera_max_rad = CAMERA_MAX_RAD,
        lr_actor     = args.lr_actor,
        lr_critic    = args.lr_critic,
        lr_alpha     = args.lr_alpha,
        gamma        = args.gamma,
        tau          = args.tau,
        batch_size   = args.batch_size,
        buffer_size  = args.buffer_size,
        warmup_steps = args.warmup,
        target_entropy = args.target_entropy,
        reward_clip  = args.reward_clip,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")
    elif args.bc_ckpt:
        agent.load(args.bc_ckpt, only_actor=True)
        print(f"BC checkpoint cargado solo en actor: {args.bc_ckpt}\n")

    if args.preload_demos:
        preload_demo_buffer(agent, args.preload_demos, args.preload_max,
                            img_frame_stack=args.img_frame_stack)

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "hybrid_sac_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido -- checkpoint en: {ckpt}")
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
        flag_counts       = {f: 0 for f in HYBRID_FLAGS}
        cam_yaw_abs_sum   = 0.0
        cam_pitch_abs_sum = 0.0
        terminated = truncated = False

        elapsed = (time.time() - train_start) / 60
        print(f"\n{'-'*60}")
        print(f"  HybridSAC Ep {ep}/{args.episodes}  |  alpha={agent.alpha:.4f}  "
              f"buffer={len(agent.buffer)}  elapsed={elapsed:.1f}min")
        print(f"{'-'*60}")

        obs, _ = env.reset()

        while not (terminated or truncated):
            action_vec = agent.select_action(obs)        # (n_flags + 2,)
            next_obs, reward, terminated, truncated, info = env.step(action_vec)
            done = terminated or truncated
            # Bootstrap solo se anula en terminación real, no en truncamiento por
            # límite de pasos (ahí el estado final no es terminal).
            stats = agent.step(obs, action_vec, reward, next_obs, terminated)

            if stats:
                ep_losses_q.append((stats["loss_q1"] + stats["loss_q2"]) / 2)
                ep_losses_actor.append(stats["loss_actor"])
                ep_alphas.append(stats["alpha"])
                ep_entropies.append(stats["entropy"])

            for i, f in enumerate(HYBRID_FLAGS):
                if action_vec[i] > 0.5:
                    flag_counts[f] += 1
            cam_yaw_abs_sum   += abs(float(action_vec[N_HYBRID_FLAGS]))
            cam_pitch_abs_sum += abs(float(action_vec[N_HYBRID_FLAGS + 1]))

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
                epsilon     = 0.0,
                action      = info["action_name"],
                extra       = {"alpha": round(agent.alpha, 4),
                               "logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step},
            )

            cur_state = obs["state"][-STATE_DIM:]
            state_str = "  ".join(f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, cur_state))
            ent_str  = f"H={stats['entropy']:+.3f}"  if stats else "H=---"
            loss_str = f"lossQ={stats['loss_q1']:.3f}" if stats else "lossQ=---"
            print(f"  [step {ep_steps:3d}]  {info['action_name'][:50]:50s}  "
                  f"alpha={agent.alpha:.3f}  {ent_str}  {loss_str}")

            if broken_this_step:
                print(f"  [step {ep_steps:3d}]  *** LOG ROTO ***  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif collected_this_step:
                print(f"  [step {ep_steps:3d}]  LOG RECOGIDO (+{collected_this_step})  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")

        ep_time   = time.time() - ep_start
        avg_q     = sum(ep_losses_q)     / len(ep_losses_q)     if ep_losses_q     else 0.0
        avg_actor = sum(ep_losses_actor) / len(ep_losses_actor) if ep_losses_actor else 0.0
        avg_alpha = sum(ep_alphas)       / len(ep_alphas)       if ep_alphas       else agent.alpha
        avg_ent   = sum(ep_entropies)    / len(ep_entropies)    if ep_entropies    else 0.0
        success   = bool(info.get("success", False))
        end_reason = ("EXITO" if success else
                      "terminado" if terminated else "truncado (timeout)")

        w_norms = agent.weight_max_abs()
        w_max   = max(w_norms.values()) if w_norms else 0.0

        cam_y_mean = cam_yaw_abs_sum   / max(1, ep_steps)
        cam_p_mean = cam_pitch_abs_sum / max(1, ep_steps)

        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken,
                            extra={"avg_loss":       round(avg_q, 6),
                                   "avg_loss_actor": round(avg_actor, 6),
                                   "alpha":          round(avg_alpha, 4),
                                   "entropy":        round(avg_ent, 4),
                                   "epsilon":        0.0,
                                   "ep_time_s":      round(ep_time, 2),
                                   "logs_collected": ep_logs_collected,
                                   "success":        success,
                                   "weight_max":     round(w_max, 4),
                                   "cam_yaw_abs":    round(cam_y_mean, 4),
                                   "cam_pitch_abs":  round(cam_p_mean, 4),
                                   "flag_pct":       {f: round(100*c/max(1,ep_steps), 1)
                                                       for f, c in flag_counts.items()}})

        print(f"\n  Fin: {end_reason}  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}")
        print(f"  loss_Q={avg_q:.4f}  loss_actor={avg_actor:+.4f}  "
              f"alpha={avg_alpha:.4f}  H={avg_ent:+.3f}  ||w||_max={w_max:.2f}  "
              f"tiempo={ep_time:.1f}s")
        print(f"  Flags: " + "  ".join(f"{f}={100*c/max(1,ep_steps):.0f}%"
                                       for f, c in flag_counts.items()))
        print(f"  Cam |dyaw|_avg={cam_y_mean:.3f}rad  |dpitch|_avg={cam_p_mean:.3f}rad")

        if w_max > 50.0:
            print(f"  WARN DIVERGENCIA: ||weight||_max={w_max:.2f} (umbral=50).")

        if ep % args.checkpoint_every == 0:
            ckpt = Path(run_dir) / f"hybrid_sac_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  -> checkpoint guardado: {ckpt}")

        maybe_world_reset(args.port, ep, args.reset_world_every)

    agent.save(str(Path(run_dir) / "hybrid_sac_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento Hybrid SAC finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
