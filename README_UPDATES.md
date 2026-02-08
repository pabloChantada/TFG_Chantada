# 📑 Documentation Index

Welcome! This guide helps you navigate all the documentation created for the agent testing update.

## 🎯 Start Here (Choose Your Path)

### ⚡ I want to test agents RIGHT NOW
1. **Run**: `./test_agents.sh` (Linux/macOS) or `.\test_agents.ps1` (Windows)
2. **Wait** for MindServer and Agent to start
3. **Visit**: http://localhost:3000 to see the bot
4. **Done!** ✅

→ **Read**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands

---

### 📚 I want to understand everything
1. **Read**: [COMPLETION_REPORT.md](COMPLETION_REPORT.md) - What was done
2. **Read**: [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) - Technical changes
3. **Read**: [AGENT_TESTING.md](AGENT_TESTING.md) - Complete guide
4. **Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick lookup

---

### 🔧 I'm developing/debugging
1. **Check**: [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) - File locations
2. **Read**: [AGENT_TESTING.md](AGENT_TESTING.md#troubleshooting) - Troubleshooting
3. **Use**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
4. **Debug**: Run `./test_agents.sh` in test mode

---

## 📖 Documentation Overview

### Quick Start Guides

| Document | Time | Best For |
|:--------:|:----:|:--------:|
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 2-3 min | Command lookup, common tasks |
| **[FILE_ORGANIZATION.md](FILE_ORGANIZATION.md)** | 3-4 min | Finding files, understanding structure |
| **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** | 5 min | Understanding what was done |

### Detailed Guides

| Document | Time | Best For |
|:--------:|:----:|:--------:|
| **[AGENT_TESTING.md](AGENT_TESTING.md)** | 10-15 min | Complete testing guide |
| **[UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)** | 5-10 min | Technical change details |

---

## 🎬 Test Scripts

All three test scripts do the same thing, just in different languages:

| Script | OS | Run With |
|:------:|:--:|:--------:|
| `test_agents.sh` | Linux/macOS | `./test_agents.sh --cleanup` |
| `test_agents.ps1` | Windows (PS) | `.\test_agents.ps1 -Cleanup` |
| `test_agents.bat` | Windows (CMD) | `test_agents.bat --cleanup` |

**Features**:
- ✅ Check Minecraft server connectivity
- ✅ Launch MindServer automatically
- ✅ Test HTN agent creation
- ✅ Clean metrics on request
- ✅ Graceful shutdown

---

## 📋 Document Descriptions

### 1. QUICK_REFERENCE.md
**What**: One-page cheat sheet  
**Contains**:
- Most common commands
- Web interface URLs
- Key files location
- Troubleshooting quick fixes

**Use When**: You need a command quickly

---

### 2. AGENT_TESTING.md
**What**: Comprehensive testing guide  
**Contains**:
- Prerequisites
- Quick start options
- CLI reference
- Web interface details
- Full troubleshooting guide
- Architecture overview

**Use When**: Setting up or debugging

---

### 3. UPDATE_SUMMARY.md
**What**: Technical change documentation  
**Contains**:
- Code changes explained
- Before/after comparisons
- New functions
- File structure
- Migration notes

**Use When**: Understanding the refactoring

---

### 4. FILE_ORGANIZATION.md
**What**: File structure and relationships  
**Contains**:
- Full project structure
- File categorization
- File relationships diagram
- Quick access commands
- Documentation map

**Use When**: Finding files or understanding the layout

---

### 5. COMPLETION_REPORT.md
**What**: Summary of update  
**Contains**:
- What was done
- Files modified/created
- Quick start options
- Improvements summary
- Testing checklist

**Use When**: Understanding the full scope of changes

---

### 6. test_agents.sh/ps1/bat
**What**: Automated test scripts  
**Purpose**: Run complete testing workflow

**Options**:
```bash
./test_agents.sh --cleanup           # Clean metrics before running
./test_agents.sh --help              # Show options
```

---

## 🗺️ Navigation Quick Links

### By Task

**I want to:**
- [Launch agents quickly](AGENT_TESTING.md#quick-start)
- [Understand the CLI options](AGENT_TESTING.md#agent-cli-options)
- [Access web interfaces](AGENT_TESTING.md#accessing-agent-interfaces)
- [Debug issues](AGENT_TESTING.md#troubleshooting)
- [Find a specific file](FILE_ORGANIZATION.md)
- [Know what changed](UPDATE_SUMMARY.md)
- [See common commands](QUICK_REFERENCE.md)

### By Document

**[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
- ⚡ Quick Start Commands (line 5-25)
- 📋 Common Commands (line 28-40)
- 🌐 Web Interfaces (line 43-54)
- 📁 Key Files (line 57-67)
- ⚙️ Agent CLI Options (line 70-82)
- 🐛 Troubleshooting (line 85-101)

**[AGENT_TESTING.md](AGENT_TESTING.md)**
- 📖 Prerequisites (line 1-10)
- 🚀 Quick Start (line 14-60)
- 📝 Agent CLI Options (line 63-150)
- 🖥️ Server.js Options (line 153-200)
- 🌐 Accessing Interfaces (line 203-220)
- 📊 Metrics (line 223-230)
- 🐛 Troubleshooting (line 233-260)

---

## 🎓 Learning Path

**Complete Beginner**
1. Run `./test_agents.sh` to see it work
2. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Read [COMPLETION_REPORT.md](COMPLETION_REPORT.md)
4. Try manual commands from [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Intermediate User**
1. Read [AGENT_TESTING.md](AGENT_TESTING.md) completely
2. Run tests with different options
3. Check [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) to understand structure
4. Review [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) for technical details

**Advanced Developer**
1. Review [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md) for code changes
2. Read `src/agents/add_agent.js` source code
3. Check `src/agents/htn_agent.js` for HTN implementation
4. Review [AGENT_TESTING.md](AGENT_TESTING.md#troubleshooting) for debugging

---

## 🔍 Finding Specific Information

### I need to know...

**About Commands**
→ [QUICK_REFERENCE.md - Common Commands](QUICK_REFERENCE.md#-common-commands)

**About CLI Options**
→ [AGENT_TESTING.md - Agent CLI Options](AGENT_TESTING.md#agent-cli-options)
→ [AGENT_TESTING.md - Server.js Options](AGENT_TESTING.md#serverjs-options)

**About Web Interfaces**
→ [AGENT_TESTING.md - Accessing Interfaces](AGENT_TESTING.md#accessing-agent-interfaces)
→ [QUICK_REFERENCE.md - Web Interfaces](QUICK_REFERENCE.md#-web-interfaces)

**About File Locations**
→ [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md)
→ [QUICK_REFERENCE.md - Key Files](QUICK_REFERENCE.md#-key-files)

**About Code Changes**
→ [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)

**About Troubleshooting**
→ [AGENT_TESTING.md - Troubleshooting](AGENT_TESTING.md#troubleshooting)
→ [QUICK_REFERENCE.md - Troubleshooting](QUICK_REFERENCE.md#-troubleshooting)

---

## 🚀 Next Steps

1. **Choose Your First Action**
   - [ ] Run test script: `./test_agents.sh`
   - [ ] Read QUICK_REFERENCE.md
   - [ ] Read AGENT_TESTING.md

2. **Test Your Setup**
   - [ ] Minecraft running on localhost:25565
   - [ ] Test script executes successfully
   - [ ] Agent connects and loads

3. **Understand Your Tools**
   - [ ] Review CLI options
   - [ ] Check available interfaces
   - [ ] Explore metrics output

4. **Start Developing**
   - [ ] Create custom agents
   - [ ] Modify HTN tasks
   - [ ] Analyze metrics

---

## 📞 Quick Help

**Can't decide where to start?**
→ Run `./test_agents.sh` then read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Script not working?**
→ Check [AGENT_TESTING.md#troubleshooting](AGENT_TESTING.md#troubleshooting)

**Need specific command?**
→ Search [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Want to understand changes?**
→ Read [UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)

**Finding a file?**
→ Check [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md)

---

## 📊 Document Statistics

| Document | Lines | Topics | Read Time |
|:--------:|:-----:|:------:|:---------:|
| QUICK_REFERENCE.md | ~120 | 5 | 2-3 min |
| FILE_ORGANIZATION.md | ~180 | 6 | 3-4 min |
| COMPLETION_REPORT.md | ~200 | 8 | 5 min |
| AGENT_TESTING.md | ~350 | 12 | 10-15 min |
| UPDATE_SUMMARY.md | ~200 | 9 | 5-10 min |

**Total Documentation**: ~1,050 lines of guides and examples

---

## ✅ Ready to Go!

You now have:
- ✅ 3 test scripts (bash, PowerShell, batch)
- ✅ 5 comprehensive documentation files
- ✅ Updated server.js
- ✅ Complete file organization guide
- ✅ Quick reference for common tasks

**Everything is ready to use!** 🎉

Pick a document above and get started!

---

**Last Updated**: February 7, 2026  
**Total Files Created**: 8  
**Documentation Pages**: 6  
**Test Scripts**: 3
