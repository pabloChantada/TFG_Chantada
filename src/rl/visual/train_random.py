"""
Baseline random — Minecraft woodcutting.

Política uniforme sobre el espacio discreto de acciones (equivalente a SAC/DQN
con epsilon fijado a 1.0). Sirve como suelo cero para comparar los métodos
entrenados: cualquier algoritmo aprendido debe superar este baseline en
success rate, troncos recogidos y reward acumulado.

No hay red neuronal, replay buffer ni gradientes. Solo `env.action_space.sample()`
y registro de métricas idéntico al de SAC para hacer la comparación directa.

Uso:
    python src/rl/visual/train_random.py [--episodes N] [--port P]
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
from constants import MAX_STEPS, RL_BRIDGE_PORT, STATE_DIM, STATE_KEYS, ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="Random baseline (eps=1) — Minecraft woodcutting")
    p.add_argument("--episodes",          type=int, default=100)
    p.add_argument("--max-steps",         type=int, default=MAX_STEPS)
    p.add_argument("--port",              type=int, default=RL_BRIDGE_PORT)
    p.add_argument("--run-dir",           type=str, default=None)
    p.add_argument("--img-frame-stack",   type=int, default=4)
    p.add_argument("--reset-world-every", type=int, default=20)
    p.add_argument("--seed",              type=int, default=None,
                   help="Semilla para el RNG del action_space (reproducibilidad).")
    return p.parse_args()


def make_run_dir() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    d  = _RL_DIR / "visual" / "runs" / f"random_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def write_config(run_dir, args):
    import subprocess
    from constants import (
        ACTIONS, STATE_KEYS, STATE_DIM, IMG_SIZE,
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
            "algorithm":  "random-uniform (eps=1)",
        },
        "train": vars(args),
        "architecture": {
            "type":         "ninguna (sampling uniforme)",
            "n_actions":    len(ACTIONS),
            "actions":      ACTIONS,
            "state_dim":    STATE_DIM,
            "state_keys":   STATE_KEYS,
            "img_size":     IMG_SIZE,
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
            print(f"  [world_reset] OK — seed={data.get('seed', '?')}")
        else:
            print(f"  [world_reset] Agente sin servidor gestionado, ignorado.")
    except Exception as exc:
        print(f"  [world_reset] ERROR: {exc}")


def run(args):
    run_dir = args.run_dir or make_run_dir()
    write_config(run_dir, args)

    print(f"Run dir:  {run_dir}")
    print(f"Random baseline: uniform over {len(ACTIONS)} acciones — eps=1.0\n")

    env     = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                             img_frame_stack=args.img_frame_stack,
                             max_steps=args.max_steps)
    metrics = RLMetrics(run_dir)

    if args.seed is not None:
        env.action_space.seed(args.seed)

    def _save_and_exit(_sig, _frame):
        metrics.plot(save=True)
        print(f"\nInterrumpido — métricas en: {run_dir}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    train_start = time.time()
    global_step = 0

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
        print(f"  RANDOM Episodio {ep}/{args.episodes}  |  elapsed={elapsed:.1f}min")
        print(f"{'─'*60}")

        obs, _ = env.reset()

        while not (terminated or truncated):
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, info = env.step(action)

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
                epsilon     = 1.0,
                action      = info["action_name"],
                extra       = {"logs_collected": collected_this_step,
                               "logs_broken":    broken_this_step},
            )

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

        ep_time   = time.time() - ep_start
        success   = bool(info.get("success", False))
        end_reason = ("ÉXITO" if success else
                      "terminado" if terminated else "truncado (timeout)")

        metrics.log_episode(ep, ep_reward, ep_steps, ep_logs_broken,
                            extra={"epsilon":        1.0,
                                   "ep_time_s":      round(ep_time, 2),
                                   "logs_collected": ep_logs_collected,
                                   "success":        success})

        print(f"\n  Fin: {end_reason}  reward={ep_reward:+.4f}  steps={ep_steps}  "
              f"logs_rotos={ep_logs_broken}  logs_recogidos={ep_logs_collected}  "
              f"tiempo={ep_time:.1f}s")
        print(f"  Acciones: " + "  ".join(f"{a}={n}" for a, n in action_counts.items() if n))

        if ep % 10 == 0:
            metrics.plot(save=True)

        maybe_world_reset(args.port, ep, args.reset_world_every)

    metrics.plot(save=True)
    env.close()
    print(f"\nBaseline random finalizado. Run en: {run_dir}")


if __name__ == "__main__":
    run(parse_args())
