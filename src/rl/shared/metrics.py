"""
Métricas de entrenamiento RL: guardado incremental en JSONL y visualización.

Uso:
    metrics = RLMetrics("src/rl/runs/20260412_120000")
    metrics.log_episode(ep=1, total_reward=-3.2, steps=120, logs_broken=0)
    metrics.plot(save=True)

    # Cargar run previo:
    metrics = RLMetrics.load("src/rl/runs/20260412_120000")
    metrics.plot()
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class RLMetrics:
    """Acumula métricas episódicas y las persiste de forma incremental en JSONL."""

    def __init__(self, run_dir: str):
        self.run_dir   = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "metrics.jsonl"
        self._episodes: list[dict] = []

    # ── Registro ──────────────────────────────────────────────────────────────

    def log_episode(
        self,
        episode:      int,
        total_reward: float,
        steps:        int,
        logs_broken:  int,
        extra:        dict | None = None,
    ):
        """Registra las métricas de un episodio y las escribe en disco."""
        entry = {
            "episode":      episode,
            "total_reward": round(float(total_reward), 4),
            "steps":        steps,
            "logs_broken":  logs_broken,
        }
        if extra:
            entry.update(extra)

        self._episodes.append(entry)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Visualización ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True, window: int = 20):
        """Genera gráficas de reward, duración de episodios y troncos rotos."""
        if not self._episodes:
            print("Sin episodios que graficar.")
            return

        eps     = [e["episode"]      for e in self._episodes]
        rewards = [e["total_reward"] for e in self._episodes]
        steps   = [e["steps"]        for e in self._episodes]
        logs    = [e["logs_broken"]  for e in self._episodes]

        def ma(data: list, w: int) -> np.ndarray:
            return np.convolve(data, np.ones(w) / w, mode="valid")

        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        # ── Reward ────────────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(eps, rewards, alpha=0.35, color="steelblue", label="reward")
        if len(rewards) >= window:
            ax.plot(eps[window - 1:], ma(rewards, window),
                    color="steelblue", linewidth=2, label=f"MA({window})")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Reward total")
        ax.set_title("Reward por episodio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Steps ─────────────────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(eps, steps, alpha=0.4, color="darkorange", label="steps")
        if len(steps) >= window:
            ax.plot(eps[window - 1:], ma(steps, window),
                    color="darkorange", linewidth=2)
        ax.set_ylabel("Steps")
        ax.set_title("Duración de episodios")
        ax.grid(True, alpha=0.3)

        # ── Troncos rotos ─────────────────────────────────────────────────────
        ax = axes[2]
        ax.bar(eps, logs, color="forestgreen", alpha=0.65)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Troncos rotos")
        ax.set_title("Troncos rotos por episodio")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            out = self.run_dir / "metrics.png"
            plt.savefig(out, dpi=120)
            print(f"Métricas guardadas en {out}")
        else:
            plt.show()

        plt.close()

    # ── Carga ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, run_dir: str) -> "RLMetrics":
        """Carga las métricas de un run previo desde disco."""
        m = cls(run_dir)
        if m._log_path.exists():
            with open(m._log_path) as f:
                m._episodes = [json.loads(line) for line in f if line.strip()]
            print(f"Cargados {len(m._episodes)} episodios desde {m._log_path}")
        return m
