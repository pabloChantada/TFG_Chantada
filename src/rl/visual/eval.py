"""
Evaluación de un modelo DQN visual entrenado.

Uso:
    python src/rl/visual/eval.py --checkpoint src/rl/visual/runs/<run>/dqn_visual_final.pth
"""

import sys
import argparse
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np

from env import MinecraftRLEnv
from model import VisualQNetwork
from constants import ACTIONS, STATE_DIM, IMG_SIZE, RL_BRIDGE_PORT, MAX_STEPS, LOGS_TO_SUCCESS


def parse_args():
    p = argparse.ArgumentParser(description="Evaluación DQN visual")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes",   type=int, default=5)
    p.add_argument("--port",       type=int, default=RL_BRIDGE_PORT)
    p.add_argument("--feat-dim",   type=int, default=256)
    p.add_argument("--hidden",     type=int, default=256)
    p.add_argument("--no-state",   action="store_true",
                   help="Modo solo imagen (debe coincidir con el checkpoint)")
    p.add_argument("--render",     action="store_true")
    p.add_argument("--logs-to-success", type=int, default=None,
                   help="Override del nº de troncos para terminar el episodio. "
                        "Por defecto usa el de constants.py. Útil para evaluar "
                        "modelos entrenados con régimen multi-log (e.g. --logs-to-success 99)")
    p.add_argument("--max-steps",  type=int, default=None,
                   help="Override del límite de steps por episodio")
    p.add_argument("--epsilon",    type=float, default=0.0,
                   help="Probabilidad de acción aleatoria (default 0=greedy puro). "
                        "Útil para diagnosticar policy collapse: si el bot se queda "
                        "girando con eps=0 pero rinde con eps=0.05, hay colapso.")
    p.add_argument("--verbose-q",  action="store_true",
                   help="Imprime las Q-values en cada step")
    return p.parse_args()


def load_model(checkpoint_path: str, feat_dim: int, hidden: int, use_state: bool):
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt         = torch.load(checkpoint_path, map_location=device)
    img_channels = ckpt.get("img_channels", 3)
    net          = VisualQNetwork(feat_dim=feat_dim, hidden=hidden,
                                  use_state=use_state, in_channels=img_channels)
    net.load_state_dict(ckpt["q_net"])
    net.to(device).eval()
    print(f"Checkpoint cargado: {checkpoint_path}")
    print(f"  step={ckpt.get('step', '?')}  epsilon_entrenamiento={ckpt.get('epsilon', '?'):.4f}"
          f"  img_channels={img_channels}")
    return net, device, img_channels


@torch.no_grad()
def select_action(net: VisualQNetwork, obs: dict, device,
                  epsilon: float = 0.0, verbose: bool = False) -> tuple[int, np.ndarray]:
    img   = torch.tensor(obs["image"], dtype=torch.float32, device=device)
    state = torch.tensor(obs["state"], dtype=torch.float32, device=device)
    if img.max() > 1.0:
        img = img / 255.0
    s = state.unsqueeze(0) if net.use_state else None
    q = net(img.unsqueeze(0), s)
    q_np = q.squeeze(0).cpu().numpy()
    if epsilon > 0.0 and np.random.rand() < epsilon:
        action = int(np.random.randint(len(ACTIONS)))
    else:
        action = int(q.argmax(dim=1).item())
    if verbose:
        qstr = "  ".join(f"{ACTIONS[i][:6]}={q_np[i]:+.2f}" for i in range(len(ACTIONS)))
        print(f"    Q: {qstr}  ->  {ACTIONS[action]}")
    return action, q_np


def run_eval(args):
    use_state   = not args.no_state
    net, device, img_channels = load_model(args.checkpoint, args.feat_dim, args.hidden, use_state)
    logs_target = args.logs_to_success if args.logs_to_success is not None else LOGS_TO_SUCCESS
    max_steps   = args.max_steps       if args.max_steps       is not None else MAX_STEPS
    env         = MinecraftRLEnv(bridge_port=args.port, use_visual=True,
                                 img_frame_stack=img_channels // 3,
                                 max_steps=max_steps,
                                 logs_to_success=logs_target)
    print(f"Eval config: logs_to_success={logs_target}  max_steps={max_steps}")

    total_rewards, total_logs, total_successes, total_steps = [], [], 0, []
    action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
    print(f"Política: epsilon={args.epsilon}  verbose_q={args.verbose_q}")

    for ep in range(1, args.episodes + 1):
        obs, _    = env.reset()
        ep_reward = 0.0
        ep_logs   = 0
        ep_steps  = 0
        terminated = truncated = False

        print(f"\n{'─'*55}")
        print(f"  Episodio {ep}/{args.episodes}  [greedy, epsilon=0]")
        print(f"{'─'*55}")

        while not (terminated or truncated):
            action, _   = select_action(net, obs, device,
                                        epsilon=args.epsilon, verbose=args.verbose_q)
            action_name = ACTIONS[action]
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            logs_step  = sum(1 for b in info["blocks_broken"] if "log" in b)
            ep_logs   += logs_step

            if args.render:
                print(f"  step={ep_steps:3d}  accion={action_name:<22}  "
                      f"reward={reward:+.3f}  total={ep_reward:+.3f}")
            elif logs_step:
                print(f"  [step {ep_steps:3d}]  *** LOG ROTO ***  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")
            elif info.get("is_attacking_tree"):
                print(f"  [step {ep_steps:3d}]  atacando árbol  "
                      f"reward={reward:+.2f}  total={ep_reward:+.2f}")

        end_reason = "terminado" if terminated else "timeout"
        success    = bool(info.get("success", False))
        if success:
            total_successes += 1
        print(f"\n  Fin: {end_reason}  |  success={success}  reward={ep_reward:+.4f}  "
              f"steps={ep_steps}  logs={ep_logs}")

        total_rewards.append(ep_reward)
        total_logs.append(ep_logs)
        total_steps.append(ep_steps)

    env.close()

    print(f"\n{'═'*55}")
    print(f"  RESUMEN ({args.episodes} episodios, política greedy, logs_to_success={logs_target})")
    print(f"  success rate = {total_successes}/{args.episodes} "
          f"({100.0 * total_successes / args.episodes:.1f}%)")
    print(f"  reward  media={np.mean(total_rewards):+.2f}  "
          f"std={np.std(total_rewards):.2f}  "
          f"min={np.min(total_rewards):+.2f}  max={np.max(total_rewards):+.2f}")
    print(f"  logs    media={np.mean(total_logs):.2f}  total={sum(total_logs)}  "
          f"max={max(total_logs)}")
    print(f"  steps   media={np.mean(total_steps):.1f}  "
          f"min={min(total_steps)}  max={max(total_steps)}")
    total_actions = int(action_counts.sum())
    print(f"  acciones (distribución sobre {total_actions} steps):")
    for i, n in enumerate(action_counts):
        pct = 100.0 * n / total_actions if total_actions else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {ACTIONS[i]:<22} {n:5d} ({pct:5.1f}%) {bar}")
    top_action_pct = 100.0 * action_counts.max() / total_actions if total_actions else 0.0
    if top_action_pct > 70.0:
        print(f"  ⚠ Policy collapse: {ACTIONS[int(action_counts.argmax())]} "
              f"domina con {top_action_pct:.1f}% — prueba --epsilon 0.05")
    print(f"{'═'*55}")


if __name__ == "__main__":
    run_eval(parse_args())
