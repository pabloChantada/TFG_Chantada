/**
 * Abstract base class for all agents.
 * Handles bot connection, memory management, viewer setup, and shutdown.
 * 
 * @class BaseAgent
 * @param {string} agentName - The name of the agent.
 * @param {string} [agentType='base'] - The type of the agent.
 */
 
/**
 * Initialize memory file. Clears existing memory if loadMemory is false.
 * 
 * @async
 * @param {boolean} [loadMemory=false] - Flag to determine if memory should be loaded instead of cleared.
 * @returns {Promise<void>}
 */
 
/**
 * Connects to a Minecraft server and sets up the bot.
 * Waits for bot to spawn or timeout (30s).
 * 
 * @async
 * @param {Object} settings - The settings for connecting to the server.
 * @param {string} settings.host - The server host (default: 127.0.0.1).
 * @param {number} settings.port - The server port (default: 25565).
 * @param {string} settings.minecraft_version - The version of Minecraft (default: auto).
 * @param {string} settings.auth - The authentication method (default: offline).
 * @param {boolean} settings.load_memory - Whether to load existing agent memory.
 * @param {boolean} settings.render_bot_view - Whether to render the bot viewer.
 * @param {number} settings.spawn_timeout - Timeout for spawning in the world (default: 30s).
 * @param {boolean} settings.metrics_enabled - Whether to collect metrics.
 * @param {string} [settings.metrics_export_path] - Path to export metrics (if enabled).
 * @returns {Promise<void>}
 * @throws {Error} If connection fails, bot is kicked, or spawn timeout occurs.
 */
 
/**
 * Sets up the browser viewer for the bot.
 * 
 * @param {number} port - The port for the browser viewer.
 * @param {number} [viewDistance=10] - The view distance for the viewer.
 * @param {boolean} [firstPerson=true] - Whether to render in first person view.
 * @returns {void}
 */
 
/**
 * Contains agent-specific logic. Must be implemented in subclass.
 * 
 * @async
 * @throws {Error} If not implemented in subclass.
 * @returns {Promise<void>}
 */
 
/**
 * Gracefully shuts down the bot and closes connections.
 * 
 * @async
 * @returns {Promise<void>}
 */
 
/**
 * Checks if the agent is idle (not connected or not spawned).
 * 
 * @returns {boolean} True if the agent is idle, false otherwise.
 */
 
/**
 * Responds to a chat message. Can be overridden in subclass if needed.
 * 
 * @param {string} from - The sender of the message.
 * @param {string} message - The message content.
 * @returns {void}
 */


import { initBot } from '../../llm/src/utils/mcdata.js';
import { addBrowserViewer } from '../../llm/src/agent/vision/browser_viewer.js';
import { serverProxy } from '../../llm/src/agent/mindserver_proxy.js';
import fs from 'fs';
import path from 'path';


export class BaseAgent {
    constructor(agentName, agentType = 'base') {
        this.name = agentName;
        this.agentType = agentType;
        this.bot = null;
        this.memoryPath = null;
    }

    /**
     * Initialize memory file
     * @param {boolean} loadMemory - Whether to load existing memory instead of clearing it
     * @returns {Promise<void>}
     */
    async clearMemory(loadMemory = false) {
        // We dont load the memory, and the memory file exists
        if (!loadMemory && this.memoryPath) {
            try {
                // Clear memory by writing an empty object to the file
                const dir = path.dirname(this.memoryPath);
                if (!fs.existsSync(dir)) {
                    fs.mkdirSync(dir, { recursive: true });
                }
                fs.writeFileSync(this.memoryPath, JSON.stringify({}));
                console.log(`[${this.name}] Memory cleared at ${this.memoryPath}`);
            } catch (e) {
                console.error(`[${this.name}] Error clearing memory:`, e);
            }
        }
    }

    /**
     * Connect to Minecraft server and setup bot
     * @param {Object} settings - The settings for connecting to the server
     * @returns {Promise<void>}
     * NOTE: for more settings, see create_agent.js -> buildAgentSettings()
     */
    async connectBot(settings) {
        console.log(`[INFO] [${this.name}] Connecting to Minecraft...`);
        console.log(`[INFO] [${this.name}] Settings:`, {
            load_memory: settings.load_memory,
            render_bot_view: settings.render_bot_view,
            spawn_timeout: settings.spawn_timeout,
            metrics_enabled: settings.metrics_enabled,
            metrics_export_path: settings.metrics_export_path,
        });

        // McData function to initialize the bot connection
        this.bot = initBot(this.name, settings);

        // Return a promise that resolves when the bot successfully spawns in the world
        // and rejects if there is an error or if it gets kicked from the server
        return new Promise((resolve, reject) => {
            // Connecting to the server (25565)
            this.bot.on('login', () => {
                console.log(`[INFO] [${this.name}] Logged in to server`);
                serverProxy.login();
            });
            // Handle connection errors
            this.bot.on('error', (err) => {
                console.error(`[ERROR] [${this.name}] Bot error:`, err);
                reject(err);
            });
            // Handle being kicked from the server
            this.bot.on('kicked', (reason) => {
                console.error(`[ERROR] [${this.name}] Bot kicked:`, reason);
                reject(new Error(`[ERROR] Kicked: ${reason}`));
            });
            // Handle spawn event
            this.bot.once('spawn', async () => {
                console.log(`[INFO] [${this.name}] Spawned in world`);
                resolve();
            });

            // Timeout if spawn takes too long
            setTimeout(() => {
                reject(new Error('[ERROR] Spawn timeout'));
            }, 30000);
        });
    }

    /**
     * Setup browser viewer
     * 
     * @param {number} port - The port for the browser viewer
     * @param {number} viewDistance - The view distance for the viewer
     * @param {boolean} firstPerson - Whether to render in first person view or not
     * @returns {void}
     */
    setupViewer(port, viewDistance = 10, firstPerson = true) {
        try {
            // Function from browser_viewer.js to add a browser viewer for the bot
            addBrowserViewer(this.bot, port, viewDistance, firstPerson);
            console.log(`[INFO] [${this.name}] Viewer available at http://localhost:${port}`);
        } catch (e) {
            console.warn(`[INFO] [${this.name}] Failed to setup viewer:`, e.message);
        }
    }

    /**
     * Implement in subclass - contains agent-specific logic
     */
    async runLogic() {
        throw new Error('runLogic() must be implemented in subclass');
    }

    /**
     * Graceful shutdown
     */
    async shutdown() {
        if (this.bot) {
            console.log(`[INFO] [${this.name}] Shutting down...`);
            try {
                await this.bot.quit();
            } catch (e) {
                console.error(`[ERROR] [${this.name}] Error during shutdown:`, e.message);
            }
        }
    }

    /**
     * Check if agent is idle
     */
    isIdle() {
        return !this.bot || !this.bot.players[this.name];
    }

    /**
     * Respond to chat message (override in subclass if needed)
     */
    respondFunc(from, message) {
        if (this.bot) {
            this.bot.chat(`[INFO] [${this.agentType.toUpperCase()}] Received: ${message}`);
        }
    }
}
