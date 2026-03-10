import fs from 'fs';
import path from 'path';
import { initBot } from '../../llm/src/utils/mcdata.js';
import { logInfo, logError } from '../logging.js';
import { mineflayer as mineflayerViewer } from 'prismarine-viewer';


const DEFAULT_SPAWN_TIMEOUT = 30000; // 30 seconds
const DEFAULT_VIEW_DISTANCE = 10;
const VIEWER_INIT_DELAY = 1000; // Wait 1 second before initializing viewer

export class BaseAgent {
    constructor(agentName, agentType = `base`) {
        this.name = agentName;
        this.agentType = agentType;
        this.bot = null;
        this.memoryPath = null;
        this.viewerInitialized = false;
    }

    /**
     * Clear the agent's memory
     */
    async clearMemory() {
        try {
            logInfo(this.name, `Clearing memory...`);

            const dir = path.dirname(this.memoryPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            fs.writeFileSync(this.memoryPath, JSON.stringify({}));

            logInfo(this.name, `Memory cleared at ${this.memoryPath}`);
        } catch (error) {
            logError(this.name, error);
        }
    }

    /**
     * Connect the bot to the Minecraft server
     * @param {Object} settings - The settings for the bot
     * @returns {Promise<void>}
     */
    async connectBot(settings) {
        logInfo(this.name, `Connecting to Minecraft...`);
        logInfo(this.name, `Settings: ${JSON.stringify(settings)}`);

        this.bot = initBot(this.name, settings);

        return new Promise((resolve, reject) => {
            let settled = false;

            const finalize = (err = null) => {
                if (settled) return;
                settled = true;
                clearTimeout(spawnTimeout);
                if (err) reject(err);
                else resolve();
            };

            this.bot.once(`login`, () => {
                logInfo(this.name, `Logged in to server`);
            });

            this.bot.once(`error`, (err) => {
                logError(this.name, err);
                finalize(err);
            });

            this.bot.once(`kicked`, (reason) => {
                const err = new Error(`Kicked: ${reason}`);
                logError(this.name, new Error(`Bot kicked: ${reason}`));
                finalize(err);
            });

            this.bot.once(`spawn`, async () => {
                logInfo(this.name, `Spawned in world`);

                // Suppress deprecated physicTick event warnings
                if (this.bot.listeners('physicTick').length > 0) {
                    this.bot.removeAllListeners('physicTick');
                }

                finalize();
            });

            const spawnTimeout = setTimeout(() => {
                const timeoutError = new Error(`Spawn timeout after ${DEFAULT_SPAWN_TIMEOUT / 1000} seconds`);
                logError(this.name, timeoutError);
                finalize(timeoutError);
            }, DEFAULT_SPAWN_TIMEOUT);
        });
    }

    /**
     * Setup the viewer for the bot
     * @param {number} port - The port for the viewer
     * @param {number} viewDistance - The view distance for the viewer
     * @param {boolean} firstPerson - Whether to use first-person view
     * @returns {Promise<boolean>} True if viewer initialized successfully
     */
    async setupViewer(port, viewDistance = DEFAULT_VIEW_DISTANCE, firstPerson = true) {
        try {
            await new Promise(resolve => setTimeout(resolve, VIEWER_INIT_DELAY));
            logInfo(this.name, `Initializing viewer on port ${port}...`);
            // addBrowserViewer(this.bot, port, viewDistance, firstPerson);
            mineflayerViewer(this.bot, { viewDistance, firstPerson, port }) // Start the viewing server on the specified port
            // Esperar a que el puerto esté realmente disponible
            await this._waitForPort(port, 15000);
            
            this.viewerInitialized = true;
            logInfo(this.name, `Viewer available at http://localhost:${port}`);
            return true;
        } catch (error) {
            logError(this.name, new Error(`Failed to setup viewer: ${error.message}`));
            logError(this.name, new Error(`Screenshots will be disabled for this session`));
            this.viewerInitialized = false;
            return false;
        }
    }

    /**
     * Check if the bot is idle
     * @returns {boolean} - True if the bot is idle, false otherwise
     */
    isIdle() {
        return !this.bot || !this.bot.players[this.name];
    }

    /**
     * Run the agent's main logic loop (to be implemented by subclasses)
     */
    async runLogic() {
        throw new Error(`runLogic() must be implemented in subclass`);
    }

    /**
     * Shutdown the bot and clean up resources
     */
    async shutdown() {
        if (this.bot) {
            logInfo(this.name, `Shutting down...`);

            // Close prismarine-viewer server to free port and memory
            if (this.viewerInitialized) {
                try {
                    if (this.bot.viewer) {
                        this.bot.viewer.close();
                        logInfo(this.name, `Viewer server closed`);
                    }
                } catch (error) {
                    logError(this.name, new Error(`Error closing viewer: ${error.message}`));
                }
                this.viewerInitialized = false;
            }

            try {
                this.bot.quit();
            } catch (error) {
                logError(this.name, new Error(`Error during shutdown: ${error.message}`));
            }
            this.bot = null;
        }
    }

    async _waitForPort(port, timeout = 15000) {
        const start = Date.now();
        const { createConnection } = await import('net');
        
        while (Date.now() - start < timeout) {
            const isOpen = await new Promise((resolve) => {
                const socket = createConnection({ port, host: 'localhost' }, () => {
                    socket.destroy();
                    resolve(true);
                });
                socket.on('error', () => {
                    socket.destroy();
                    resolve(false);
                });
                socket.setTimeout(2000, () => {
                    socket.destroy();
                    resolve(false);
                });
            });

            if (isOpen) {
                logInfo(this.name, `Port ${port} is ready (took ${Date.now() - start}ms)`);
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        throw new Error(`Port ${port} not ready after ${timeout}ms`);
    }
}