# 🗂️ File Organization Guide

## 📁 Project Structure After Update

```
TFG_Chantada/
│
├── 📄 Testing & Documentation (NEW)
│   ├── test_agents.sh              ⭐ Quick test script (Linux/macOS)
│   ├── test_agents.ps1             ⭐ Quick test script (PowerShell)
│   ├── test_agents.bat             ⭐ Quick test script (CMD)
│   ├── AGENT_TESTING.md            📖 Complete testing guide
│   ├── QUICK_REFERENCE.md          📖 One-page quick reference
│   ├── UPDATE_SUMMARY.md           📖 Technical change details
│   └── COMPLETION_REPORT.md        📖 This update's summary
│
├── 📂 src/
│   ├── 📂 agents/ (REFACTORED)
│   │   ├── add_agent.js            ✨ Universal agent launcher
│   │   ├── htn_agent.js            ✨ HTN agent class
│   │   ├── types/
│   │   │   ├── base_agent.js       ✨ Base agent class
│   │   │   └── htn_agent.js        (archived)
│   │   ├── memories/               📦 Agent memory storage
│   │   └── agents.example.json     📄 Example config
│   │
│   ├── 📂 server/ (UPDATED)
│   │   ├── server.js               ✅ Updated with new imports
│   │   ├── mindcraft.js            MindServer implementation
│   │   ├── mindserver.js           Server proxy
│   │   └── public/                 Web UI assets
│   │
│   ├── 📂 htn/                     HTN task definitions
│   │   ├── main_htn.js
│   │   ├── progression.js
│   │   └── primitives/
│   │       ├── mining.js
│   │       ├── crafting.js
│   │       └── ...
│   │
│   ├── 📂 llm/                     LLM/Mindcraft integration
│   │   ├── main.js
│   │   ├── my_agent/
│   │   ├── profiles/
│   │   └── src/
│   │
│   ├── 📂 metrics/                 Metrics collection
│   │   ├── metrics_collector.js
│   │   └── agent_metrics/          📊 Output metrics
│   │
│   ├── 📂 evaluation/              Evaluation scripts
│   └── 📂 server/
│
├── 📄 Configuration Files
│   ├── package.json
│   ├── requirements.txt
│   ├── README.md
│   └── LICENSE
│
└── 📂 docs/                        Documentation
    ├── QUICK.md
    ├── TODO.md
    └── bitacora/
```

---

## 🎯 Which File to Use?

### For Testing Agents Quickly
| OS | File | Command |
|:--:|:----:|---------|
| 🐧 Linux | `test_agents.sh` | `./test_agents.sh --cleanup` |
| 🍎 macOS | `test_agents.sh` | `./test_agents.sh --cleanup` |
| 🪟 Windows (PS) | `test_agents.ps1` | `.\test_agents.ps1 -Cleanup` |
| 🪟 Windows (CMD) | `test_agents.bat` | `test_agents.bat --cleanup` |

### For Learning/Reference
| Need | File | Time |
|:----:|:----:|:---:|
| Quick commands | `QUICK_REFERENCE.md` | 2 min |
| Testing guide | `AGENT_TESTING.md` | 5 min |
| Technical details | `UPDATE_SUMMARY.md` | 3 min |
| What was done | `COMPLETION_REPORT.md` | 5 min |

### For Development
| Task | File | Location |
|:----:|:----:|:--------:|
| Launch server | `src/server/server.js` | New: uses HTNAgent directly |
| Create agents | `src/agents/add_agent.js` | Refactored agent launcher |
| HTN logic | `src/htn/main_htn.js` | Task definitions |
| Metrics | `src/metrics/agent_metrics/` | Output directory |

---

## 🔀 File Relationships

```
User Command
    ↓
[test_agents.sh/ps1/bat]
    ↓
Launches MindServer
    ├─→ [src/server/mindcraft.js]
    │
Launches Agent
    ├─→ [src/agents/add_agent.js]
        ├─→ [src/agents/htn_agent.js]
        ├─→ [src/agents/types/base_agent.js]
        ├─→ [src/htn/main_htn.js]
        └─→ [src/metrics/metrics_collector.js]
        
Results
    └─→ [src/metrics/agent_metrics/*.json]
```

---

## ⚡ Quick Access

### Start Testing (Copy & Paste)
```bash
# Linux/macOS
chmod +x test_agents.sh && ./test_agents.sh --cleanup

# Windows PowerShell
.\test_agents.ps1 -Cleanup

# Windows Command Prompt
test_agents.bat --cleanup
```

### Manual Agent Launch
```bash
# Terminal 1: Start server
node src/server/mindcraft.js --port 8080

# Terminal 2: Start agent
node src/agents/add_agent.js --name Agent1 --type htn --port 8080 -c 0
```

### View Results
```bash
# Linux/macOS
cat src/metrics/agent_metrics/Agent1_metrics.json

# Windows PowerShell
Get-Content src\metrics\agent_metrics\Agent1_metrics.json

# Windows Command Prompt
type src\metrics\agent_metrics\Agent1_metrics.json
```

---

## 📊 Documentation Map

```
START HERE
    ↓
┌─────────────────────────────────────────┐
│  QUICK_REFERENCE.md                    │
│  (Common commands, quick lookup)        │
└─────────────────────────────────────────┘
    ↓ Want more details?
┌─────────────────────────────────────────┐
│  AGENT_TESTING.md                       │
│  (Complete guide, troubleshooting)      │
└─────────────────────────────────────────┘
    ↓ Need technical info?
┌─────────────────────────────────────────┐
│  UPDATE_SUMMARY.md                      │
│  (Code changes, migration)              │
└─────────────────────────────────────────┘
    ↓ What happened?
┌─────────────────────────────────────────┐
│  COMPLETION_REPORT.md                   │
│  (What was done, files changed)         │
└─────────────────────────────────────────┘
```

---

## 🗄️ File Categories

### 🆕 New Files (Created in This Update)
- `test_agents.sh` - Shell test script
- `test_agents.ps1` - PowerShell test script
- `test_agents.bat` - Batch test script
- `AGENT_TESTING.md` - Testing guide
- `QUICK_REFERENCE.md` - Quick reference
- `UPDATE_SUMMARY.md` - Change documentation
- `COMPLETION_REPORT.md` - Update report

### ✏️ Modified Files
- `src/server/server.js` - Updated agent imports

### 🔄 Refactored Files (Already Changed)
- `src/agents/add_agent.js` - New universal launcher
- `src/agents/htn_agent.js` - HTN agent class
- `src/agents/types/base_agent.js` - Base class

---

## 📱 Web Interfaces Available

Once agents are running:

```
┌──────────────────────────────────────────┐
│  MindServer Control Panel                │
│  http://localhost:8080                   │
│  (Server controls & status)              │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Bot Viewer (3D World View)              │
│  http://localhost:3000 (Agent 0)         │
│  http://localhost:3001 (Agent 1)         │
│  http://localhost:3002 (Agent 2)         │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Inventory Viewer                        │
│  http://localhost:4001 (Agent 0)         │
│  http://localhost:4002 (Agent 1)         │
└──────────────────────────────────────────┘
```

---

## ✅ Verification Checklist

- [ ] All new files created (check with: `ls test_agents.* *.md`)
- [ ] server.js updated (check: `grep "HTNAgent" src/server/server.js`)
- [ ] Test scripts are executable (run: `chmod +x test_agents.sh`)
- [ ] Documentation files are readable (open in editor)
- [ ] No syntax errors (run test: `./test_agents.sh`)

---

## 🚀 Next Steps

1. **Choose your OS** and run the appropriate test script
2. **Read** [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. **Follow** [AGENT_TESTING.md](AGENT_TESTING.md) for detailed instructions
4. **Check** [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) for technical changes
5. **Customize** agent parameters for your experiments

---

**All files organized and ready to use! 🎉**
