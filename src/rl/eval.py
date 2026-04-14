"""
Inferencia / evaluación de un modelo DQN entrenado.

Carga un checkpoint, ejecuta N episodios con epsilon=0 (política greedy pura)
y muestra métricas por episodio.

Uso:
    python src/rl/eval.py --checkpoint src/rl/runs/<run>/dqn_final.pth
    python src/rl/eval.py --checkpoint dqn_final.pth --episodes 10 --render
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from env import MinecraftRLEnv
from dqn import QNetwork
from constants import ACTIONS, STATE_DIM, RL_BRIDGE_PORT


def parse_args():
    p = argparse.ArgumentParser(description="Evaluación DQN Minecraft")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Ruta al .pth guardado por train.py")
    p.add_argument("--episodes",   type=int, default=5,
                   help="Nº de episodios de evaluación (defecto: 5)")
    p.add_argument("--port",       type=int, default=RL_BRIDGE_PORT)
    p.add_argument("--hidden",     type=int, default=128,
                   help="Neuronas ocultas — debe coincidir con el modelo guardado")
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--render",     action="store_true",
                   help="Imprimir acción elegida en cada step")
    return p.parse_args()


def load_model(checkpoint_path: str, state_dim: int, hidden: int) -> QNetwork:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(state_dim=state_dim, n_actions=len(ACTIONS), hidden=hidden)
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt["q_net"])
    net.to(device)
    net.eval()
    print(f"Checkpoint cargado: {checkpoint_path}")
    print(f"  step={ckpt.get('step', '?')}  epsilon_entrenamiento={ckpt.get('epsilon', '?'):.4f}")
    return net, device


@torch.no_grad()
def select_action(net: QNetwork, obs: np.ndarray, device) -> int:
    t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = net(t)
    return int(q_values.argmax(dim=1).item())


def run_eval(args):
    obs_dim = STATE_DIM * args.frame_stack
    net, device = load_model(args.checkpoint, obs_dim, args.hidden)

    env = MinecraftRLEnv(bridge_port=args.port, frame_stack=args.frame_stack)

    total_rewards = []
    total_logs    = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_logs   = 0
        ep_steps  = 0
        terminated = truncated = False

        print(f"\n{'─'*55}")
        print(f"  Episodio {ep}/{args.episodes}  [greedy, epsilon=0]")
        print(f"{'─'*55}")

        while not (terminated or truncated):
            action = select_action(net, obs, device)
            action_name = ACTIONS[action]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1

            logs_step = sum(1 for b in info["blocks_broken"] if "log" in b)
            ep_logs  += logs_step

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
    print(f"  logs    media={np.mean(total_logs):.1f}  "
          f"total={sum(total_logs)}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    run_eval(parse_args())
