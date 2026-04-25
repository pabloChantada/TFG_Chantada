"""
Métricas de entrenamiento RL: guardado incremental en JSONL y visualización.

Dos niveles de granularidad:
- `log_step()`  → una línea por step en `steps.jsonl` (loss, reward, ε, acción).
- `log_episode()` → una línea por episodio en `metrics.jsonl` (resumen).

Uso:
    metrics = RLMetrics("src/rl/runs/20260412_120000")
    metrics.log_step(global_step=1, episode=1, reward=0.1, loss=0.42,
                     epsilon=0.98, action="attack")
    metrics.log_episode(episode=1, total_reward=-3.2, steps=120, logs_broken=0)
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
    """Acumula métricas por step y por episodio, y las persiste en JSONL."""

    def __init__(self, run_dir: str):
        self.run_dir      = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ep_path     = self.run_dir / "metrics.jsonl"
        self._step_path   = self.run_dir / "steps.jsonl"
        self._episodes: list[dict] = []
        self._steps:    list[dict] = []

    # ── Registro por step ─────────────────────────────────────────────────────

    def log_step(
        self,
        global_step: int,
        episode:     int,
        reward:      float,
        loss:        float | None,
        epsilon:     float,
        action:      str,
        extra:       dict | None = None,
    ):
        """Registra métricas de un step individual y las escribe en disco."""
        entry = {
            "global_step": global_step,
            "episode":     episode,
            "reward":      round(float(reward), 4),
            "loss":        round(float(loss), 6) if loss is not None else None,
            "epsilon":     round(float(epsilon), 4),
            "action":      action,
        }
        if extra:
            entry.update(extra)

        self._steps.append(entry)
        with open(self._step_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Registro por episodio ─────────────────────────────────────────────────

    def log_episode(
        self,
        episode:      int,
        total_reward: float,
        steps:        int,
        logs_broken:  int,
        extra:        dict | None = None,
    ):
        """Registra el resumen de un episodio y lo escribe en disco."""
        entry = {
            "episode":      episode,
            "total_reward": round(float(total_reward), 4),
            "steps":        steps,
            "logs_broken":  logs_broken,
        }
        if extra:
            entry.update(extra)

        self._episodes.append(entry)
        with open(self._ep_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Visualización ─────────────────────────────────────────────────────────

    def plot(self, save: bool = True, ep_window: int = 20, step_window: int = 200):
        """Genera panel completo de métricas por step y por episodio."""
        if not self._episodes and not self._steps:
            print("Sin datos que graficar.")
            return

        fig = plt.figure(figsize=(16, 12))
        gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

        # ── Fila 1: métricas por step (loss y reward) ─────────────────────────
        self._plot_loss_per_step(fig.add_subplot(gs[0, 0]), step_window)
        self._plot_reward_per_step(fig.add_subplot(gs[0, 1]), step_window)

        # ── Fila 2: exploración + distribución de acciones ────────────────────
        self._plot_epsilon(fig.add_subplot(gs[1, 0]))
        self._plot_action_distribution(fig.add_subplot(gs[1, 1]))

        # ── Fila 3: reward y duración por episodio ────────────────────────────
        self._plot_reward_per_episode(fig.add_subplot(gs[2, 0]), ep_window)
        self._plot_episode_duration(fig.add_subplot(gs[2, 1]), ep_window)

        # ── Fila 4: troncos por episodio + success rate ───────────────────────
        self._plot_logs_per_episode(fig.add_subplot(gs[3, 0]))
        self._plot_success_rate(fig.add_subplot(gs[3, 1]), ep_window)

        fig.suptitle(f"Métricas RL — {self.run_dir.name}", fontsize=14, y=0.995)

        if save:
            out = self.run_dir / "metrics.png"
            plt.savefig(out, dpi=110, bbox_inches="tight")
            print(f"Métricas guardadas en {out}")
        else:
            plt.show()

        plt.close(fig)

    # ── Helpers de plot ───────────────────────────────────────────────────────

    @staticmethod
    def _ma(data: np.ndarray, w: int) -> np.ndarray:
        if len(data) < w:
            return np.array([])
        return np.convolve(data, np.ones(w) / w, mode="valid")

    def _plot_loss_per_step(self, ax, window: int):
        """
        Escala adaptativa:
          - Si la loss varía > 2 órdenes de magnitud (divergencia), escala log
            para ver picos y valle a la vez.
          - Si es estable (rango < 2 órdenes), escala lineal con ylim al p99
            para no aplastar la señal por outliers.
        """
        losses = [s["loss"] for s in self._steps if s.get("loss") is not None]
        if not losses:
            ax.text(0.5, 0.5, "sin loss registrada (warmup o sin steps)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title("Loss por step")
            return

        losses = np.array(losses, dtype=np.float64)
        x      = np.arange(len(losses))
        p99    = float(np.percentile(losses, 99))
        p50    = float(np.percentile(losses, 50))
        lmax   = float(losses.max())

        diverging = lmax / max(p50, 1e-9) > 100.0

        ax.plot(x, losses, alpha=0.25, color="crimson", linewidth=0.6, label="loss")
        ma = self._ma(losses, window)
        if len(ma):
            ax.plot(np.arange(window - 1, len(losses)), ma,
                    color="crimson", linewidth=1.8, label=f"MA({window})")

        if diverging:
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log)")
            ax.set_title(f"Loss del DQN por step  —  max={lmax:.2g} (divergencia)")
        else:
            ax.set_ylim(0, p99 * 1.2)
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss del DQN por step  —  p50={p50:.3g}  p99={p99:.3g}")
            if lmax > p99 * 1.2:
                ax.text(0.98, 0.95,
                        f"max={lmax:.2g} fuera de rango",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="gray",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.set_xlabel("Step de entrenamiento")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    def _plot_reward_per_step(self, ax, window: int):
        """
        Reward por step es muy sparse (≥99% son penalty base).
        En vez de una línea, graficamos:
          - línea fina: reward acumulado dentro del episodio actual
            (se resetea a 0 en cada nuevo episodio)
          - scatter: eventos con reward > +0.1 (break, collect, success, hit_tree),
            coloreados según la magnitud.
        """
        del window  # no usado
        if not self._steps:
            ax.set_title("Reward por step (eventos)")
            return

        rewards  = np.array([s["reward"]   for s in self._steps], dtype=np.float64)
        episodes = np.array([s["episode"]  for s in self._steps], dtype=np.int64)
        x        = np.arange(len(rewards))

        # Reward acumulado intra-episodio: reset al cambio de episodio.
        cum   = np.zeros_like(rewards)
        running = 0.0
        prev_ep = episodes[0] if len(episodes) else -1
        for i, (r, ep) in enumerate(zip(rewards, episodes)):
            if ep != prev_ep:
                running = 0.0
                prev_ep = ep
            running += r
            cum[i] = running
        ax.plot(x, cum, color="steelblue", linewidth=0.8,
                alpha=0.7, label="reward acumulado (ep)")

        # Eventos positivos (picos dispersos).
        mask = rewards > 0.1
        if mask.any():
            ax.scatter(x[mask], rewards[mask], c=rewards[mask],
                       cmap="plasma", s=14, alpha=0.85, label="evento (+reward)",
                       zorder=3)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Eventos de reward + acumulado intra-episodio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_epsilon(self, ax):
        if not self._steps:
            ax.set_title("Epsilon")
            return
        eps = [s["epsilon"] for s in self._steps]
        ax.plot(np.arange(len(eps)), eps, color="goldenrod", linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("ε")
        ax.set_title("Decaimiento de ε (exploración)")
        ax.grid(True, alpha=0.3)

    def _plot_action_distribution(self, ax):
        if not self._steps:
            ax.set_title("Distribución de acciones")
            return
        counts: dict[str, int] = {}
        for s in self._steps:
            a = s.get("action", "?")
            counts[a] = counts.get(a, 0) + 1
        if not counts:
            ax.set_title("Distribución de acciones")
            return
        labels = list(counts.keys())
        values = [counts[k] for k in labels]
        total  = sum(values)
        pcts   = [100 * v / total for v in values]
        ax.barh(labels, pcts, color="mediumpurple", alpha=0.8)
        for i, (v, p) in enumerate(zip(values, pcts)):
            ax.text(p + 0.3, i, f"{v}  ({p:.1f}%)",
                    va="center", fontsize=8)
        ax.set_xlabel("% de steps")
        ax.set_title("Distribución global de acciones")
        ax.grid(True, alpha=0.3, axis="x")

    def _plot_reward_per_episode(self, ax, window: int):
        if not self._episodes:
            ax.set_title("Reward por episodio")
            return
        eps     = [e["episode"]      for e in self._episodes]
        rewards = np.array([e["total_reward"] for e in self._episodes], dtype=np.float64)
        ax.plot(eps, rewards, alpha=0.35, color="steelblue", label="reward")
        ma = self._ma(rewards, window)
        if len(ma):
            ax.plot(eps[window - 1:], ma,
                    color="steelblue", linewidth=2, label=f"MA({window})")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Reward total")
        ax.set_title("Reward por episodio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_episode_duration(self, ax, window: int):
        if not self._episodes:
            ax.set_title("Duración de episodios")
            return
        eps   = [e["episode"] for e in self._episodes]
        steps = np.array([e["steps"] for e in self._episodes], dtype=np.float64)
        ax.plot(eps, steps, alpha=0.4, color="darkorange")
        ma = self._ma(steps, window)
        if len(ma):
            ax.plot(eps[window - 1:], ma,
                    color="darkorange", linewidth=2, label=f"MA({window})")
            ax.legend(fontsize=8)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Steps")
        ax.set_title("Duración de episodios")
        ax.grid(True, alpha=0.3)

    def _plot_logs_per_episode(self, ax):
        if not self._episodes:
            ax.set_title("Troncos por episodio")
            return
        eps     = [e["episode"]     for e in self._episodes]
        broken  = [e.get("logs_broken", 0)    for e in self._episodes]
        collect = [e.get("logs_collected", 0) for e in self._episodes]
        w = 0.4
        x = np.array(eps, dtype=np.float64)
        ax.bar(x - w/2, broken,  width=w, color="forestgreen", alpha=0.75, label="rotos")
        ax.bar(x + w/2, collect, width=w, color="saddlebrown", alpha=0.75, label="recogidos")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Troncos")
        ax.set_title("Troncos rotos y recogidos por episodio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_success_rate(self, ax, window: int):
        """Tasa de éxito móvil: fracción de episodios con ≥1 tronco recogido."""
        if not self._episodes:
            ax.set_title("Tasa de éxito")
            return
        eps     = [e["episode"] for e in self._episodes]
        success = np.array(
            [1.0 if e.get("logs_collected", 0) > 0 else 0.0 for e in self._episodes],
            dtype=np.float64,
        )
        ax.plot(eps, success, alpha=0.2, color="teal", linewidth=0.8)
        ma = self._ma(success, window)
        if len(ma):
            ax.plot(eps[window - 1:], ma,
                    color="teal", linewidth=2, label=f"MA({window})")
            ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Episodio")
        ax.set_ylabel("P(éxito)")
        ax.set_title("Tasa de éxito (episodios con ≥1 tronco recogido)")
        ax.grid(True, alpha=0.3)

    # ── Carga ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, run_dir: str) -> "RLMetrics":
        """Carga métricas de un run previo desde disco (episodios y steps)."""
        m = cls(run_dir)
        if m._ep_path.exists():
            with open(m._ep_path) as f:
                m._episodes = [json.loads(line) for line in f if line.strip()]
            print(f"Cargados {len(m._episodes)} episodios desde {m._ep_path}")
        if m._step_path.exists():
            with open(m._step_path) as f:
                m._steps = [json.loads(line) for line in f if line.strip()]
            print(f"Cargados {len(m._steps)} steps desde {m._step_path}")
        return m
