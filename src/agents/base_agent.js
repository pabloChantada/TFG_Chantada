/**
 * BaseAgent - Abstract base class for all agent types (HTN, LLM, RL)
 * Provides common lifecycle management and bot initialization
 */

import { initBot } from '../llm/src/utils/mcdata.js';
import { addBrowserViewer } from '../llm/src/agent/vision/browser_viewer.js';
import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import fs from 'fs';
import path from 'path';

export class BaseAgent {
    constructor(agentName, agentType = 'base') {
        this.name = agentName;
        this.agentType = agentType;
        this.bot = null;
        this.countId = 0;
        this.memoryPath = null;
    }

    /**
     * Initialize memory file
     */
    async clearMemory(loadMemory = false) {
        if (!loadMemory && this.memoryPath) {
            try {
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
     */
    async connectBot(settings) {
        console.log(`[${this.name}] Connecting to Minecraft...`);
        console.log(`[${this.name}] Settings:`, {
            host: settings.host,
            port: settings.port,
            version: settings.minecraft_version,
            auth: settings.auth
        });

        this.bot = initBot(this.name, settings);

        return new Promise((resolve, reject) => {
            this.bot.on('login', () => {
                console.log(`[${this.name}] Logged in to server`);
                serverProxy.login();
            });

            this.bot.on('error', (err) => {
                console.error(`[${this.name}] Bot error:`, err);
                reject(err);
            });

            this.bot.on('kicked', (reason) => {
                console.error(`[${this.name}] Bot kicked:`, reason);
                reject(new Error(`Kicked: ${reason}`));
            });

            this.bot.once('spawn', async () => {
                console.log(`[${this.name}] Spawned in world`);
                resolve();
            });

            // Timeout if spawn takes too long
            setTimeout(() => {
                reject(new Error('Spawn timeout'));
            }, 30000);
        });
    }

    /**
     * Setup browser viewer
     */
    setupViewer(viewerPort) {
        try {
            addBrowserViewer(this.bot, viewerPort);
            console.log(`[${this.name}] Viewer available at http://localhost:${viewerPort}`);
        } catch (e) {
            console.warn(`[${this.name}] Failed to setup viewer:`, e.message);
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
            console.log(`[${this.name}] Shutting down...`);
            try {
                await this.bot.quit();
            } catch (e) {
                console.error(`[${this.name}] Error during shutdown:`, e.message);
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
            this.bot.chat(`[${this.agentType.toUpperCase()}] Received: ${message}`);
        }
    }
}
