"""
Entrenamiento SAC discreto visual + tutor simbólico — Minecraft.

Idea: en cada step pedimos al actor SAC dos cosas — la acción greedy
(argmax de logits) y la acción muestreada de la categórica.
    - Si muestreada == greedy → SAC actuó "confiado": usamos esa acción.
    - Si muestreada != greedy → SAC quería explorar (acción "random"):
      pedimos al tutor simbólico que elija en su lugar.

Así el tutor sustituye a TODA la exploración estocástica de SAC durante todo
el entrenamiento, sin probabilidad ni decay. Al principio el actor está sin
entrenar (logits ≈ uniformes) y el tutor decide casi siempre; según el actor
aprende, el softmax se concentra en la greedy y el tutor se llama cada vez
menos — el decaimiento emerge solo. Las transiciones del tutor entran al
replay buffer como cualquier otra, así SAC aprende del mix on/off-policy.

Cuando interviene el tutor, se anota (imagen, acción, regla) en disco —
útil para inspección manual y como dataset IL reutilizable.

Anotaciones en `<run_dir>/dagger/`:
    - frames/step_XXXXXXXX.png   (frame RGB 128×128 del momento)
    - dagger.jsonl               (uno por step con tutor: action, rule, step…)

Uso:
    python src/rl/visual/train_sac_dagger.py [opciones SAC] [--no-annotate]
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

import numpy as np
import requests as _requests
import torch as _torch
from PIL import Image

from env import MinecraftRLEnv
from metrics import RLMetrics
from sac import DiscreteSACAgent
from symbolic import SymbolicPolicy
from constants import MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS, ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="SAC discreto + tutor simbólico")
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
    p.add_argument("--target-entropy-pct", type=float, default=0.98)
    p.add_argument("--reward-clip",    type=float, default=1.0)
    p.add_argument("--no-reward-clip", dest="reward_clip", action="store_const", const=None)
    p.add_argument("--img-frame-stack",type=int,   default=4)
    p.add_argument("--no-state",       action="store_true")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--reset-world-every", type=int, default=20)
    p.add_argument("--checkpoint-every",  type=int, default=10)

    # ── Tutor simbólico ──
    p.add_argument("--sym-seed",    type=int, default=None,
                   help="Semilla para el RNG del tutor (reproducibilidad).")
    p.add_argument("--no-annotate", action="store_true",
                   help="No guardar PNGs ni dagger.jsonl (solo entrenar).")
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / f"sac_dagger_{ts}"
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
            "algorithm":  "SAC-discrete + tutor simbólico (sustituye exploración)",
        },
        "train": vars(args),
        "symbolic": {
            "mode":     "replace_non_greedy",  # tutor solo cuando sampled != argmax
            "annotate": not args.no_annotate,
        },
        "architecture": {
            "type":            "SAC discrete + symbolic tutor",
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


@_torch.no_grad()
def sac_greedy_and_sample(agent: DiscreteSACAgent, obs: dict) -> tuple[int, int]:
    """
    Devuelve (sampled, greedy) del actor SAC sobre el mismo forward pass.

    `sampled` viene de la categórica softmax(logits); `greedy` es argmax.
    Si son distintos, el step de SAC era "exploración" y lo delegamos al tutor.
    """
    img   = obs["image"].astype(np.float32) / 255.0 if obs["image"].max() > 1.0 \
            else np.asarray(obs["image"], dtype=np.float32)
    img_t = _torch.as_tensor(img, dtype=_torch.float32,
                             device=agent.device).unsqueeze(0)
    state_t = (_torch.as_tensor(obs["state"], dtype=_torch.float32,
                                device=agent.device).unsqueeze(0)
               if agent.use_state else None)
    logits = agent.actor(img_t, state_t)
    greedy = int(logits.argmax(dim=-1).item())
    probs  = _torch.softmax(logits, dim=-1)
    sampled = int(_torch.distributions.Categorical(probs=probs).sample().item())
    return sampled, greedy


class DaggerAnnotator:
    """Guarda (frame, acción, regla) cada vez que el tutor decide."""

    def __init__(self, run_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.dir     = Path(run_dir) / "dagger"
        self.frames  = self.dir / "frames"
        if enabled:
            self.frames.mkdir(parents=True, exist_ok=True)
            self._jsonl = self.dir / "dagger.jsonl"
            self._jsonl.touch(exist_ok=True)
        self.count = 0

    def annotate(self, global_step: int, episode: int, ep_step: int,
                 obs: dict, action_idx: int, action_name: str, rule: str,
                 sac_greedy_idx: int):
        if not self.enabled:
            return
        # obs["image"] es CHW con canales = 3 * frame_stack. Tomamos el último
        # frame (los 3 canales más recientes) y lo escribimos como PNG RGB.
        img  = obs["image"]
        last = img[-3:, :, :].transpose(1, 2, 0)  # HWC
        last = np.ascontiguousarray(last, dtype=np.uint8)
        png_path = self.frames / f"step_{global_step:08d}.png"
        Image.fromarray(last, mode="RGB").save(png_path, optimize=False)

        entry = {
            "global_step":    global_step,
            "episode":        episode,
            "ep_step":        ep_step,
            "action_idx":     int(action_idx),
            "action":         action_name,
            "rule":           rule,
            "sac_greedy_idx": int(sac_greedy_idx),
            "sac_greedy":     ACTIONS[sac_greedy_idx],
            "frame":          png_path.name,
        }
        with open(self._jsonl, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.count += 1


def train(args):
    run_dir = args.run_dir or make_run_dir()
    write_config(run_dir, args)

    print(f"Run dir:  {run_dir}")
    print(f"SAC + tutor: lr_actor={args.lr_actor}  lr_critic={args.lr_critic}  "
          f"gamma={args.gamma}  tau={args.tau}  batch={args.batch_size}  "
          f"warmup={args.warmup}")
    print(f"Tutor activo cuando sampled != argmax  |  "
          f"annotate={'no' if args.no_annotate else 'sí'}\n")

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

    symbolic  = SymbolicPolicy(seed=args.sym_seed)
    annotator = DaggerAnnotator(run_dir, enabled=not args.no_annotate)

    if args.resume:
        agent.load(args.resume)
        print(f"Checkpoint cargado: {args.resume}  (step={agent._step})\n")

    def _save_and_exit(_sig, _frame):
        ckpt = Path(run_dir) / "sac_dagger_interrupted.pth"
        agent.save(str(ckpt))
        metrics.plot(save=True)
        print(f"\nInterrumpido — checkpoint en: {ckpt}")
        print(f"Anotaciones tutor: {annotator.count}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    train_start = time.time()
    global_step = agent._step

    for ep in range(1, args.episodes + 1):
        ep_start          = time.time()
        ep_reward         = 0.0
        ep_steps          = 0
        ep_sym_steps      = 0
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
        print(f"  SAC+Tutor Ep {ep}/{args.episodes}  |  α={agent.alpha:.4f}  "
              f"buffer={len(agent.buffer)}  elapsed={elapsed:.1f}min")
        print(f"{'─'*60}")

        obs, _ = env.reset()

        while not (terminated or truncated):
            sampled_idx, greedy_idx = sac_greedy_and_sample(agent, obs)

            if sampled_idx == greedy_idx:
                # SAC fue greedy: acción confiada del policy
                action        = sampled_idx
                rule          = None
                action_src    = "sac"
                from_symbolic = False
            else:
                # SAC exploró: pasamos el turno al tutor
                action, rule  = symbolic.act(obs)
                action_src    = f"sym:{rule}"
                from_symbolic = True

            next_obs, reward, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            # Bootstrap solo se anula en terminación real, no en truncamiento por
            # límite de pasos (ahí el estado final no es terminal).
            stats = agent.step(obs, action, reward, next_obs, terminated)

            if from_symbolic:
                annotator.annotate(
                    global_step    = global_step,
                    episode        = ep,
                    ep_step        = ep_steps + 1,
                    obs            = obs,
                    action_idx     = action,
                    action_name    = info["action_name"],
                    rule           = rule,
                    sac_greedy_idx = greedy_idx,
                )
                ep_sym_steps += 1

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
                epsilon     = 0.0,
                action      = info["action_name"],
                extra       = {"alpha":          round(agent.alpha, 4),
                               "logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step,
                               "action_src":     action_src,
                               "sac_greedy":     ACTIONS[greedy_idx]},
            )

            cur_state = obs["state"][-STATE_DIM:]
            state_str = "  ".join(f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, cur_state))
            ent_str   = f"H={stats['entropy']:.3f}"  if stats else "H=---"
            loss_str  = f"lossQ={stats['loss_q1']:.3f}" if stats else "lossQ=---"
            tag       = "SYM" if from_symbolic else "SAC"
            print(f"  [step {ep_steps:3d} {tag}]  state: {state_str}  "
                  f"action={info['action_name']}  "
                  f"α={agent.alpha:.3f}  {ent_str}  {loss_str}")

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
        avg_q      = sum(ep_losses_q)     / len(ep_losses_q)     if ep_losses_q     else 0.0
        avg_actor  = sum(ep_losses_actor) / len(ep_losses_actor) if ep_losses_actor else 0.0
        avg_alpha  = sum(ep_alphas)       / len(ep_alphas)       if ep_alphas       else agent.alpha
        avg_ent    = sum(ep_entropies)    / len(ep_entropies)    if ep_entropies    else 0.0
        success    = bool(info.get("success", False))
        end_reason = ("ÉXITO" if success else
                      "terminado" if terminated else "truncado (timeout)")

        w_norms = agent.weight_max_abs()
        w_max   = max(w_norms.values()) if w_norms else 0.0
        sym_frac = ep_sym_steps / ep_steps if ep_steps else 0.0

        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken,
                            extra={"avg_loss":       round(avg_q, 6),
                                   "avg_loss_actor": round(avg_actor, 6),
                                   "alpha":          round(avg_alpha, 4),
                                   "entropy":        round(avg_ent, 4),
                                   "epsilon":        0.0,
                                   "sym_steps":      ep_sym_steps,
                                   "sym_fraction":   round(sym_frac, 4),
                                   "ep_time_s":      round(ep_time, 2),
                                   "logs_collected": ep_logs_collected,
                                   "success":        success,
                                   "weight_max":     round(w_max, 4)})

        print(f"\n  Fin: {end_reason}  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}")
        print(f"  tutor: {ep_sym_steps}/{ep_steps} steps ({sym_frac*100:.1f}%)  "
              f"anotaciones acumuladas={annotator.count}")
        print(f"  loss_Q={avg_q:.4f}  loss_actor={avg_actor:+.4f}  "
              f"α={avg_alpha:.4f}  H={avg_ent:.3f}  ||w||_max={w_max:.2f}  "
              f"tiempo={ep_time:.1f}s")
        print(f"  Acciones: " + "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        if w_max > 50.0:
            print(f"  ⚠ DIVERGENCIA: ||weight||_max={w_max:.2f} (umbral=50).")

        if ep % args.checkpoint_every == 0:
            ckpt = Path(run_dir) / f"sac_dagger_ep{ep}.pth"
            agent.save(str(ckpt))
            metrics.plot(save=True)
            print(f"  → checkpoint guardado: {ckpt}")

        maybe_world_reset(args.port, ep, args.reset_world_every)

    agent.save(str(Path(run_dir) / "sac_dagger_final.pth"))
    metrics.plot(save=True)
    env.close()
    print(f"\nEntrenamiento SAC+Tutor finalizado. Run en: {run_dir}")
    print(f"Anotaciones tutor totales: {annotator.count}")


if __name__ == "__main__":
    train(parse_args())
