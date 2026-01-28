# Copilot Instructions for TFG_Chantada

## Overview
This repository focuses on integrating multiple AI algorithms (right now HTN and LLM's) with Minecraft bots using the Mindcraft and Mineflayer framework. The project is structured to facilitate bot interactions with Minecraft worlds, leveraging AI models and modular task definitions.

### Key Components
- **`src/htn/`**: Contains HTN logic and primitive tasks (e.g., mining, crafting, smelting).
- **`src/llm/`**: A fork of Mindcraft, used for connecting bots to Minecraft servers. Includes:
  - `main.js`: Entry point for the Mindcraft server.
  - `my_agent/`: Custom agent logic.
  - `profiles/`: Configuration files for different AI models.
  - `services/viaproxy/`: Proxy service for connecting to unsupported Minecraft server versions.
- **`api_test/`**: Scripts for testing various functionalities.

## Developer Workflows

### Installation
1. Install Node.js (v18 or v20 LTS recommended).
2. Install dependencies:
   ```bash
   npm install
   cd src/llm
   npm install
   ```
   > Note: `src/llm/` uses `patch-package` to apply patches during `npm install`.

### Running the Project
1. Start a Minecraft world and open it to LAN.
2. Launch the Mindcraft server:
   ```bash
   cd src/llm
   node main.js
   ```
   - Viewer: [http://localhost:3000](http://localhost:3000)
   - Web Inventory: [http://localhost:3001](http://localhost:3001)
   - MindServer: [http://localhost:8080](http://localhost:8080)
3. Launch additional agents:
   ```bash
   cd src/llm/my_agent
   node start.js -n HTNAgent -p 8080 -m "start htn" -c 0
   ```

## Project-Specific Conventions
- **HTN Tasks**: Defined in `src/htn/tasks/`. Follow the modular structure for adding new tasks.
- **Profiles**: AI model configurations are stored in `src/llm/profiles/`. Use `andy.json` as a reference.
- **Patch Management**: Patches for dependencies are stored in `src/llm/profiles/patches/` and applied automatically.

## External Dependencies
- **Mineflayer**: Core library for Minecraft bot interactions.
- **prismarine-viewer**: Provides a 3D viewer for the Minecraft world.
- **mineflayer-web-inventory**: Web-based inventory management.
- **viaproxy**: Proxy service for unsupported server versions.

## Examples
- **HTN Execution**: `src/htn/main_htn.js` demonstrates task progression towards crafting an iron pickaxe.
- **Agent Launch**: `src/llm/my_agent/start.js` shows how to initialize a custom agent.

## References
- [Mineflayer Documentation](https://github.com/PrismarineJS/mineflayer)
- [Mindcraft Upstream](https://github.com/mindcraft-bots/mindcraft)
- [viaproxy Setup](src/llm/services/viaproxy/README.md)

---

For further details, consult the `README.md` files in the root and `src/llm/` directories.