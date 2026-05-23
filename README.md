# AI Agents in Minecraft — Comparing HTN, Imitation Learning & Reinforcement Learning

> Bachelor's thesis exploring classical and modern AI approaches to autonomous agent control in a complex 3D environment.

![Node.js](https://img.shields.io/badge/Node.js-ES2022-339933?logo=node.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![Minecraft](https://img.shields.io/badge/Minecraft-Java%201.21-62B47A?logo=minecraft&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

This project implements and compares four fundamentally different approaches for controlling an autonomous Minecraft bot. The HTN planner solves the full wood → stone → iron progression; the learned agents (IL, RL) focus on the **woodcutting** sub-task — a non-trivial benchmark that already requires visual perception, navigation and inventory awareness.

| Approach | Description |
|---|---|
| **HTN** | Hierarchical Task Network — hand-crafted rule-based planner (full progression) |
| **IL**  | Imitation Learning — CNN/ViT + recurrent head trained on HTN expert demonstrations (dual-head: discrete action + continuous camera) |
| **RL**  | Reinforcement Learning — custom PyTorch implementations of DQN, PPO, SAC (discrete) and Hybrid SAC (MineRL-style hybrid action space), plus a SAC + DAgger variant guided by a symbolic tutor |
| **LLM** *(experimental)* | LLM-driven agent built on top of [mindcraft](https://github.com/mindcraft-bots/mindcraft), vendored under `src/llm/` |

The core question: *can learned agents match or exceed a hand-crafted planner in a non-deterministic, partially observable 3D world?*

---

## Architecture

```
Minecraft Java Server
        │
        ├── HTN Agent (Node.js)     ← rule-based task decomposition (full progression)
        ├── IL Agent  (Node.js)     ← queries FastAPI inference server each frame
        └── RL Agent  (Node.js)     ← exposes HTTP bridge to Python Gymnasium env

                    │
            Mineflayer Bot API
         (actions, world state, events)

Python Layer
        ├── IL  : dual-head model  →  discrete action (9 classes) + camera regression (dyaw, dpitch)
        │         backbones: GRU + LSTM | ConvLSTM | ViT (timm) + LSTM
        │         └── FastAPI inference server (port 8765)
        │
        └── RL  : Gymnasium env (HTTP bridge, port 8766)
                  ├── state-only   : DQN over normalised state vector
                  └── visual       : DQN | PPO | SAC discrete | Hybrid SAC | SAC + DAgger | random baseline
                                     + reactive symbolic policy (tutor for DAgger)
```

**Data pipeline (IL):**
```
HTN gameplay  →  dataset_recorder.js  →  prepare_dataset.py  →  main.py (training)
(expert demos)    (screenshot + state      (balance, mirror,       (CNN/ViT + recurrent head,
                   + action + camera)       filter outliers)        CE on actions + MSE on camera)
```

**Optional MineRL warm-start (Hybrid SAC):**
```
MineRL Treechop-v0  →  bc_pretrain.py  →  hybrid_actor.pth  →  train_hybrid_sac.py --bc-ckpt
(human demonstrations) (BCE flags + MSE camera)               (online fine-tune on Mineflayer)
```

---

## Tech Stack

**Bot & game interface**
- [Mineflayer](https://github.com/PrismarineJS/mineflayer) — Minecraft bot API
- `mineflayer-pathfinder`, `mineflayer-pvp`, `mineflayer-collectblock`, `mineflayer-auto-eat`, `mineflayer-armor-manager` — bot capabilities
- [Prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) — 3D world viewer (Three.js)

**Machine Learning**
- PyTorch 2.5 + CUDA 12.1 — all models trained from scratch (no Stable-Baselines3)
- TorchVision — ResNet backbones (ImageNet pretrained, partial fine-tuning)
- [timm](https://github.com/huggingface/pytorch-image-models) — ViT backbone (`vit_small_patch16_224`, adapted to 128×128)
- Gymnasium 1.2 — RL environment interface
- FastAPI + Uvicorn — real-time IL inference server
- GradCAM — model interpretability

**Data collection**
- Puppeteer — automated screenshot capture from Prismarine-viewer
- Custom JSONL pipeline for (screenshot, state, action, camera_delta) tuples
- MineRL Treechop-v0 (optional) — ~453k human transitions used for Hybrid SAC BC pretrain

---

## Action & state spaces

**IL action space** — 9 discrete actions (`move_forward_jump`, `move_forward_sprint`, `move_backward_walk`, `move_left`, `move_right`, `jump`, `sneak`, `attack`, `equip_wooden_axe`) **+ continuous camera regression** `(dyaw, dpitch) ∈ [-1, 1]²`.

**RL discrete action space (visual / state-only)** — 7 actions: `attack`, `move_forward_sprint`, `move_forward_jump`, `camera_{left, right, up, down}`. Camera steps: 0.15 rad horizontal, 0.10 rad vertical.

**RL hybrid action space (Hybrid SAC, MineRL-style)** — 4 concurrent Bernoulli flags `{forward, jump, sprint, attack}` × 2D Gaussian camera `(dyaw, dpitch) ∈ [±0.5 rad]` (≈ p99 of the human MineRL dataset).

**State vector** — 8 normalised components (RL): `yaw, pitch, dx, dz, tree_visible, tree_distance, log_count, is_looking_at_log`. IL uses 9 components (adds absolute `x, y, z`).

**Reward shaping (woodcutting)** — `break_log = +20`, `collect_log = +10`, `hit_tree = +0.5`, per-step penalty `-0.01`, episode-success bonus `+30`, early-termination penalty `-5`. Episode ends on 1 log collected, 300 steps, or cumulative reward below −20.

---

## Results *(preliminary)*

> Full evaluation in progress. Metrics will be updated as experiments complete.

- **HTN**: consistent task completion across the full progression, zero learning overhead, brittle to edge cases.
- **IL**: dual-head model generalises visual patterns from expert demos; ViT backbone outperforms ResNet on the held-out split.
- **RL**: SAC discrete is the most stable on the woodcutting sub-task; the symbolic-tutor DAgger variant accelerates early exploration. Hybrid SAC + MineRL BC pretrain is the closest match to a human-like control scheme.

---

## Project Structure

```
src/
├── agents/                       # Node.js agent wrappers (HTN, IL, RL) + logging
├── htn/                          # HTN planner
│   ├── primitives/               #   blocks, mining, movement, wood, inventory, structures
│   ├── progression/              #   phase orchestrator (chop / iron progression)
│   └── tasks/                    #   crafting, smelting, block placement
├── il/                           # Imitation Learning
│   ├── dataset_recorder.js       #   HTN-time recorder (screenshot + state + action + camera)
│   ├── load_dataset.py           #   dataset + mirror augmentation
│   ├── model.py                  #   RNNExtractor (CNN+GRU+LSTM), ViTExtractor (timm)
│   ├── model_convlstm.py         #   ConvLSTM backbone variant
│   ├── main.py                   #   training (CE on actions + MSE on camera)
│   ├── inference_server.py       #   FastAPI server (port 8765)
│   └── plots.py                  #   training curves, GradCAM
├── rl/
│   ├── shared/                   # env.py (Gymnasium HTTP bridge), constants.py, metrics.py
│   ├── state/                    # State-only DQN (no images)
│   ├── visual/                   # Visual RL
│   │   ├── dqn.py / train.py / eval.py
│   │   ├── ppo.py / train_ppo.py
│   │   ├── sac.py / train_sac.py
│   │   ├── hybrid_sac.py / train_hybrid_sac.py
│   │   ├── train_sac_dagger.py   #   SAC + symbolic tutor (DAgger-style)
│   │   ├── symbolic.py           #   reactive rule-based tutor
│   │   ├── train_random.py       #   uniform-random baseline
│   │   └── eval_policy.py        #   greedy eval for PPO / SAC / Hybrid SAC
│   └── minerl/                   # MineRL Treechop-v0 utilities
│       ├── eda.py                #   action-distribution analysis
│       ├── map_actions.py        #   MineRL → hybrid action vector
│       └── bc_pretrain.py        #   BC pretrain of HybridActor
├── llm/                          # LLM agent (vendored mindcraft, experimental)
└── evaluation/                   # HTN metrics + plots + GradCAM icons
scripts/
├── run.js                        # Launch agents (--agents htn,rl --names …)
├── mass_record.js                # Automated HTN episode recording
├── dataset.py / sweep.py / cleanup_runs.py
└── paper_server.js               # Headless Minecraft server helper
docs/
├── IL_PIPELINE.md                # Detailed IL pipeline notes
└── bitacora/                     # Thesis log (Spanish)
```

---

## Getting Started

**Prerequisites:** Minecraft Java Edition server (1.21.x), Node.js 20+, Python 3.11+, CUDA-capable GPU (for training).

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# ── HTN ───────────────────────────────────────────────────────────────────
node scripts/run.js --agents htn --names HTNBot

# Mass-record HTN episodes (builds the IL dataset)
node scripts/mass_record.js

# ── IL ────────────────────────────────────────────────────────────────────
# Train (pick backbone: rnn | convlstm | vit)
python src/il/main.py --dataset data/train.jsonl --model vit --epochs 30

# Inference server + IL agent
python src/il/inference_server.py --model src/il/runs/<run>/best_model.pt
node scripts/run.js --agents il --names ILBot

# ── RL ────────────────────────────────────────────────────────────────────
# Launch the Node bridge first
node scripts/run.js --agents rl --names RLBot

# State-only DQN
python src/rl/state/train.py --episodes 200

# Visual: DQN | PPO | SAC discreto | Hybrid SAC | SAC + tutor simbólico | random
python src/rl/visual/train.py             --episodes 600
python src/rl/visual/train_ppo.py         --episodes 600
python src/rl/visual/train_sac.py         --episodes 600
python src/rl/visual/train_hybrid_sac.py  --episodes 600 [--bc-ckpt PATH]
python src/rl/visual/train_sac_dagger.py  --episodes 600
python src/rl/visual/train_random.py      --episodes 600

# Evaluate a trained checkpoint
python src/rl/visual/eval_policy.py --algo sac --checkpoint <run>/sac_final.pth --episodes 15
```

---

## Context

This is my Bachelor's Thesis (TFG) at UDC. The goal is both a technical implementation and a comparative study — not just making agents work, but understanding *why* each paradigm succeeds or fails in this environment.

The HTN solves the full wood → stone → iron progression; the learned agents focus on the **woodcutting sub-task** because it already exposes the hardest control problems (visual perception, navigation, alignment, sparse reward) without requiring multi-task curricula — making it a meaningful benchmark across all four approaches.


<!-- Copy-paste in your Readme.md file -->

<a href="https://next.ossinsight.io/widgets/official/analyze-user-contribution-time-distribution?period=all_times&user_id=147641118" target="_blank" style="display: block" align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://next.ossinsight.io/widgets/official/analyze-user-contribution-time-distribution/thumbnail.png?period=all_times&user_id=147641118&image_size=auto&color_scheme=dark" width="721" height="auto">
    <img alt="Contribution Time Distribution of @pabloChantada" src="https://next.ossinsight.io/widgets/official/analyze-user-contribution-time-distribution/thumbnail.png?period=all_times&user_id=147641118&image_size=auto&color_scheme=light" width="721" height="auto">
  </picture>
</a>

<!-- Made with [OSS Insight](https://ossinsight.io/) -->