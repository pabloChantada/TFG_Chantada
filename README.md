# AI Agents in Minecraft — Comparing HTN, Imitation Learning & Reinforcement Learning

> Bachelor's thesis exploring classical and modern AI approaches to autonomous agent control in a complex 3D environment.

![Node.js](https://img.shields.io/badge/Node.js-ES2022-339933?logo=node.js&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![Minecraft](https://img.shields.io/badge/Minecraft-Java%201.21-62B47A?logo=minecraft&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Overview

This project implements and compares three fundamentally different approaches for controlling an autonomous Minecraft bot through a structured progression task (harvesting wood → crafting stone tools → smelting iron):

| Approach | Description |
|---|---|
| **HTN** | Hierarchical Task Network — hand-crafted rule-based planner |
| **IL** | Imitation Learning — ResNet CNN trained on HTN expert demonstrations |
| **RL** | Reinforcement Learning — policy trained via Stable-Baselines3 in a Gymnasium environment |
| **LLM** *(experimental)* | LLM-driven agent using natural language task decomposition |

The core question: *can learned agents match or exceed a hand-crafted planner in a non-deterministic, partially observable 3D world?*

---

## Architecture

```
Minecraft Java Server
        │
        ├── HTN Agent (Node.js)     ← rule-based task decomposition
        ├── IL Agent  (Node.js)     ← queries inference server for each frame
        └── RL Agent  (Node.js)     ← exposes HTTP API to Python Gymnasium env

                    │
            Mineflayer Bot API
         (actions, world state, events)

Python Layer
        ├── IL: ResNet18 classifier (18 discrete actions)
        │       └── FastAPI inference server (port 8765)
        └── RL: Gymnasium environment + Stable-Baselines3
```

**Data pipeline (IL):**
```
HTN gameplay  →  dataset_recorder.js  →  prepare_dataset.py  →  main.py (training)
(expert demos)    (screenshot + state      (balance, filter)      (ResNet + AdamW)
                   + action tuples)
```

---

## Tech Stack

**Bot & game interface**
- [Mineflayer](https://github.com/PrismarineJS/mineflayer) — Minecraft bot API
- `mineflayer-pathfinder`, `mineflayer-pvp`, `mineflayer-collectblock` — bot capabilities
- [Prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) — 3D world viewer (Three.js)

**Machine Learning**
- PyTorch 2.5 + CUDA 12.1 — model training
- TorchVision — ResNet18/34/50 backbones (ImageNet pretrained, partial fine-tuning)
- Stable-Baselines3 — PPO/DQN for RL
- FastAPI + Uvicorn — real-time inference server
- GradCAM — model interpretability

**Data collection**
- Puppeteer — automated screenshot capture from Prismarine-viewer
- Custom JSONL pipeline for (screenshot, state, action) tuples

---

## Results *(preliminary)*

> Full evaluation in progress. Metrics will be updated as experiments complete.

- **HTN**: consistent task completion, zero learning overhead, brittle to edge cases
- **IL**: *in evaluation* — model generalises visual patterns from expert demos
- **RL**: *training in progress* — shaped reward around wood-chopping subtask

---

## Project Structure

```
src/
├── agents/          # BaseAgent + per-algorithm agent wrappers
├── htn/             # HTN planner (primitives, progression, pathfinding)
├── il/              # IL data collection, training, inference server
├── rl/              # Gymnasium env, Node.js HTTP bridge, training scripts
├── llm/             # LLM agent (experimental)
└── evaluation/      # Metrics, plots, GradCAM visualisation
scripts/
├── run.js           # Launch any agent type (--type htn|il|rl)
├── mass_record.js   # Automated HTN episode recording
└── prepare_dataset.py
```

---

## Getting Started

**Prerequisites:** Minecraft Java Edition server, Node.js 20+, Python 3.11+, CUDA-capable GPU (for training)

```bash
# Install JS dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Run HTN agent
node scripts/run.js --type htn

# Run IL agent (requires trained model + inference server)
python src/il/inference_server.py --model checkpoints/best_model.pt &
node scripts/run.js --type il

# Run RL training
python src/rl/train.py
```

---

## Context

This is my Bachelor's Thesis (TFG) at UDC. The goal is both a technical implementation and a comparative study — not just making agents work, but understanding *why* each paradigm succeeds or fails in this environment.

The task (wood → stone → iron progression) was chosen because it requires multi-step planning, spatial navigation, inventory management, and adapting to a procedurally generated world — making it a meaningful benchmark across all three approaches.
