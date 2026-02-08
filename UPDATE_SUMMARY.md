# Server.js Update Summary

## Changes Made

### 1. Updated Imports
**Old:**
```javascript
import { buildAgentSettings } from '../agents/create_agent.js';
```

**New:**
```javascript
import { HTNAgent } from '../agents/htn_agent.js';
```

The old `create_agent.js` file no longer exists. The refactored structure uses:
- `add_agent.js` - Universal agent launcher CLI
- `htn_agent.js` - HTN agent class
- `types/base_agent.js` - Base agent class

### 2. Moved buildAgentSettings Function
The `buildAgentSettings()` function is now defined directly in `server.js` (copied from `add_agent.js`) to ensure compatibility with the new structure.

**Key additions in settings object:**
- `profile.agent_type` - Agent type identifier
- `mindserver_port` - Port for MindServer communication
- `init_message` - Initial message for agent
- `metrics_enabled` - Flag to enable metrics collection
- `metrics_export_path` - Path to export metrics JSON

### 3. Refactored createAgentFromCLI()
**Changes:**
- Now uses `HTNAgent` class directly from `src/agents/htn_agent.js`
- Generates viewer port dynamically based on agent name/index
- Sets metrics export path automatically
- Added error handling for unsupported agent types

**Current support:**
- ✅ HTN agents (fully supported)
- ⏳ RL agents (not yet implemented)
- ⏳ LLM agents (use `add_agent.js` instead)

## New Test Scripts

### 1. `test_agents.sh` (Linux/macOS)
Bash script for quick agent testing with:
- Minecraft server connectivity check
- Automatic MindServer launch
- HTN agent testing
- Graceful shutdown
- Metrics cleanup option

**Usage:**
```bash
chmod +x test_agents.sh
./test_agents.sh --cleanup
```

### 2. `test_agents.ps1` (Windows PowerShell)
PowerShell equivalent with same features:
- Port availability checks
- Background process management
- Colored output
- Timeout handling

**Usage:**
```powershell
.\test_agents.ps1 -Cleanup
```

### 3. `AGENT_TESTING.md`
Comprehensive guide including:
- Quick start instructions
- CLI option reference
- Port mappings
- Troubleshooting guide
- Architecture overview

## Recommended Usage

### Quick Test (Use Test Scripts)
```bash
./test_agents.sh
# or
.\test_agents.ps1
```

### Custom Testing (Manual CLI)
```bash
# Terminal 1: Start MindServer
node src/server/mindcraft.js --port 8080

# Terminal 2+: Launch agents via add_agent.js (preferred for new structure)
node src/agents/add_agent.js --name Agent1 --type htn --port 8080 -c 0
node src/agents/add_agent.js --name Agent2 --type htn --port 8080 -c 1
```

### Integration Testing (Use server.js)
```bash
# Launch server with agent
node src/server/server.js --agent htn --name MyAgent
```

## File Structure Summary

```
src/
├── agents/
│   ├── add_agent.js           ← New: Universal agent launcher
│   ├── htn_agent.js           ← New: HTN agent class
│   ├── types/
│   │   ├── base_agent.js      ← New: Base class
│   │   └── htn_agent.js       ← Previous htn_agent.js (kept for reference)
│   └── memories/              ← Agent memory files
│
├── server/
│   ├── server.js              ← ✅ Updated with new imports
│   ├── mindcraft.js           ← MindServer
│   └── mindserver.js          ← Server proxy
│
├── htn/                        ← HTN task definitions
├── llm/                        ← LLM/Mindcraft integration
└── metrics/                    ← Metrics collection
    └── agent_metrics/         ← Output metrics

Root/
├── test_agents.sh             ← ✨ New: Bash test script
├── test_agents.ps1            ← ✨ New: PowerShell test script
├── AGENT_TESTING.md           ← ✨ New: Quick reference guide
└── UPDATE_SUMMARY.md          ← This file
```

## Migration Notes

If you have old code using the removed `create_agent.js`:
1. Use `add_agent.js` for launching agents via CLI
2. Use the `buildAgentSettings()` function now in `server.js`
3. Import `HTNAgent` directly from `src/agents/htn_agent.js`

## Next Steps

1. **Test the agents:**
   ```bash
   ./test_agents.sh
   ```

2. **Read the guide:**
   Open [AGENT_TESTING.md](./AGENT_TESTING.md) for detailed information

3. **Customize as needed:**
   - Edit agent parameters in test scripts
   - Add new agent types in `add_agent.js`
   - Extend `BaseAgent` for custom agent types

## Verification Checklist

- [x] `server.js` updated to use refactored agents
- [x] `buildAgentSettings()` function moved to server.js
- [x] `HTNAgent` imported and used in createAgentFromCLI()
- [x] Bash test script created (`test_agents.sh`)
- [x] PowerShell test script created (`test_agents.ps1`)
- [x] Documentation guide created (`AGENT_TESTING.md`)
- [x] Error handling for unsupported agent types
- [x] Metrics path configuration
- [x] Viewer port generation logic

All updates complete and ready for testing! 🚀
