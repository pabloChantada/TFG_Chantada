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
from constants import ACTIONS, STATE_DIM, IMG_SIZE, RL_BRIDGE_PORT


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
    return p.parse_args()


def load_model(checkpoint_path: str, feat_dim: int, hidden: int, use_state: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = VisualQNetwork(feat_dim=feat_dim, hidden=hidden, use_state=use_state)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt["q_net"])
    net.to(device).eval()
    print(f"Checkpoint cargado: {checkpoint_path}")
    print(f"  step={ckpt.get('step', '?')}  epsilon_entrenamiento={ckpt.get('epsilon', '?'):.4f}")
    return net, device


@torch.no_grad()
def select_action(net: VisualQNetwork, obs: dict, device) -> int:
    img   = torch.tensor(obs["image"], dtype=torch.float32, device=device)
    state = torch.tensor(obs["state"], dtype=torch.float32, device=device)
    if img.max() > 1.0:
        img = img / 255.0
    s = state.unsqueeze(0) if net.use_state else None
    q = net(img.unsqueeze(0), s)
    return int(q.argmax(dim=1).item())


def run_eval(args):
    use_state    = not args.no_state
    net, device  = load_model(args.checkpoint, args.feat_dim, args.hidden, use_state)
    env          = MinecraftRLEnv(bridge_port=args.port, use_visual=True)

    total_rewards, total_logs = [], []

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
            action      = select_action(net, obs, device)
            action_name = ACTIONS[action]
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
        print(f"\n  Fin: {end_reason}  |  reward={ep_reward:+.4f}  "
              f"steps={ep_steps}  logs={ep_logs}")

        total_rewards.append(ep_reward)
        total_logs.append(ep_logs)

    env.close()

    print(f"\n{'═'*55}")
    print(f"  RESUMEN ({args.episodes} episodios, política greedy)")
    print(f"  reward  media={np.mean(total_rewards):+.2f}  "
          f"std={np.std(total_rewards):.2f}  "
          f"min={np.min(total_rewards):+.2f}  max={np.max(total_rewards):+.2f}")
    print(f"  logs    media={np.mean(total_logs):.1f}  total={sum(total_logs)}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    run_eval(parse_args())
