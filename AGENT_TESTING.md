# Quick Agent Testing Guide

This guide explains how to quickly test and launch agents in the TFG_Chantada project.

## Prerequisites

1. **Node.js** (v18 or v20 LTS recommended)
2. **Minecraft Server** running locally and open to LAN on `127.0.0.1:25565`
3. Project dependencies installed:
   ```bash
   npm install
   cd src/llm && npm install
   cd ../..
   ```

## Quick Start

### Option 1: Run Test Script (Recommended)

#### On Linux/macOS:
```bash
chmod +x test_agents.sh
./test_agents.sh
```

#### On Windows (PowerShell):
```powershell
.\test_agents.ps1
```

### Option 2: Manual Launch

#### Step 1: Start MindServer
```bash
node src/server/mindcraft.js --port 8080
```
This starts the MindServer on port 8080 and opens the browser UI at `http://localhost:8080`.

#### Step 2: In a new terminal, launch an agent

**HTN Agent:**
```bash
node src/agents/add_agent.js --name Agent1 --type htn --port 8080 -c 0
```

**Multiple Agents (separate terminals):**
```bash
# Terminal 2
node src/agents/add_agent.js --name Agent1 --type htn --port 8080 -c 0

# Terminal 3
node src/agents/add_agent.js --name Agent2 --type htn --port 8080 -c 1

# Terminal 4
node src/agents/add_agent.js --name Agent3 --type htn --port 8080 -c 2
```

## Agent CLI Options

All agent types use the same CLI interface via `src/agents/add_agent.js`:

```bash
node src/agents/add_agent.js [OPTIONS]
```

### Available Options:

| Option | Alias | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--name` | `-n` | string | `Agent_X` | Agent name |
| `--type` | `-t` | string | `htn` | Agent type: `htn`, `rl`, `llm` |
| `--port` | `-p` | number | `8080` | MindServer port |
| `--minecraft-port` | `-mp` | number | `25565` | Minecraft server port |
| `--count` | `-c` | number | `0` | Agent index (affects viewer port) |
| `--load-memory` | `-l` | boolean | `false` | Load previous memory |
| `--init-message` | `-m` | string | `null` | Initial message for agent |
| `--metrics-path` | - | string | `src/metrics/agent_metrics` | Metrics export path |

### Examples:

```bash
# Simple HTN agent
node src/agents/add_agent.js --name MyAgent --type htn

# Agent with memory loading
node src/agents/add_agent.js --name Agent1 --type htn --load-memory

# Agent with custom Minecraft port
node src/agents/add_agent.js --name Bot --type htn --minecraft-port 25566

# Multiple agents (in separate terminals with different counts)
node src/agents/add_agent.js -n Agent1 -t htn -c 0
node src/agents/add_agent.js -n Agent2 -t htn -c 1
node src/agents/add_agent.js -n Agent3 -t htn -c 2
```

## Server.js Options

The `server.js` file is the main evaluation server. It supports:

```bash
node src/server/server.js [OPTIONS]
```

### Key Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | number | `8080` | Server port |
| `--agent` | string | - | Agent type to create on startup |
| `--name` | string | - | Agent name |
| `--agents` | string | - | Comma-separated agent types |
| `--names` | string | - | Comma-separated agent names |
| `--minecraft-host` | string | `127.0.0.1` | Minecraft server host |
| `--minecraft-port` | number | `25565` | Minecraft server port |
| `--no-ui` | boolean | `false` | Don't auto-open browser UI |
| `--config` | string | - | JSON config file with agent definitions |

### Examples:

```bash
# Launch server only
node src/server/server.js

# Launch server + HTN agent
node src/server/server.js --agent htn --name MyAgent

# Launch server + multiple agents
node src/server/server.js --agents htn,htn --names Agent1,Agent2

# Custom ports
node src/server/server.js --port 8081 --minecraft-port 25566
```

## Accessing Agent Interfaces

Once agents are running:

| Interface | URL | Purpose |
|-----------|-----|---------|
| MindServer UI | `http://localhost:8080` | Control panel |
| Bot Viewer | `http://localhost:3000` | 3D view of Agent1 |
| Bot Viewer | `http://localhost:3001` | 3D view of Agent2 (if running) |
| Inventory | `http://localhost:4001` | Inventory viewer for Agent1 |

Port increments by 1000 for each agent (`3000+c*1000`).

## Metrics and Results

Agent execution metrics are saved to:
- Location: `src/metrics/agent_metrics/`
- Pattern: `{AgentName}_metrics.json`

View results:
```bash
cat src/metrics/agent_metrics/Agent1_metrics.json
```

## Troubleshooting

### Agent won't connect to Minecraft
- Ensure Minecraft world is open to LAN
- Check that port `25565` is correct
- Try: `node src/agents/add_agent.js --name Test -mp 25565`

### MindServer port already in use
```bash
# Try a different port
node src/server/mindcraft.js --port 8081
node src/agents/add_agent.js -n Agent1 --port 8081
```

### Memory file conflicts
```bash
# Clear all agent memories
rm -rf src/agents/memories/*.json

# Or use load-memory flag to reuse
node src/agents/add_agent.js -n Agent1 --load-memory
```

### Too many processes
```bash
# Kill all node processes
pkill -f node  # Linux/macOS
Get-Process node | Stop-Process -Force  # PowerShell
```

## Architecture Overview

```
├── src/server/
│   ├── server.js          ← Main evaluation server
│   ├── mindcraft.js       ← MindServer implementation
│   └── mindserver.js      ← Server proxy
│
├── src/agents/
│   ├── add_agent.js       ← Universal agent launcher ⭐
│   ├── htn_agent.js       ← HTN agent implementation
│   └── types/
│       └── base_agent.js  ← Base class for all agents
│
└── src/metrics/
    └── agent_metrics/     ← Metrics output directory
```

## Next Steps

- **HTN Development**: Edit `src/htn/` for task definitions
- **Agent Configuration**: Create JSON configs in `src/agents/`
- **Metrics Analysis**: Check scripts in `src/evaluation/`

For more details, see [README.md](./README.md) and `.github/copilot-instructions.md`.
