import { BaseAgent } from './base_agent.js';
import { startChopTrees, startHTN } from '../../htn/main_htn.js';
import { logInfo, logError } from '../logging.js';

const FORCE_EXIT_TIMEOUT = 30000;

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, `htn`);
        this.memoryPath = `src/agents/memories/${agentName}_memory.json`;
    }

    /**
     * Full startup sequence for HTN agent
     * @param {Object} settings - Minecraft connection settings
     * @param {number} viewerPort - Port for browser viewer (prismarine)
     */
    async start(settings, viewerPort) {
        try {
            await this.connectBot(settings);
            await this.clearMemory();
            await this.setupViewer(viewerPort);
            this._setupBotErrorHandlers();

            logInfo(this.name, `Starting HTN execution...`);
            await this.runLogic();

        } catch (error) {
            logError(this.name, new Error(`Failed to start: ${error.message}`));
            await this._shutdown();
            process.exit(1);
        }
    }

    /**
     * Setup bot error and disconnect handlers
     */
    _setupBotErrorHandlers() {
        this._onError = (err) => {
            logError(this.name, new Error(`Bot error: ${err.message}`));
            this._handleBotDisconnect();
        };

        this._onKicked = (reason) => {
            logError(this.name, new Error(`Bot kicked: ${reason}`));
            this._handleBotDisconnect();
        };

        this._onEnd = (reason) => {
            console.warn(`[WARN] [${this.name}] Bot connection ended: ${reason}`);
        };

        this.bot.on('error', this._onError);
        this.bot.on('kicked', this._onKicked);
        this.bot.on('end', this._onEnd);
    }

    async _handleBotDisconnect() {
        await this._shutdown();
        process.exit(1);
    }

    /**
     * Execute HTN task progression
     */
    async runLogic() {
        try {
            const result = await startChopTrees(this.bot);
            logInfo(this.name, `HTN execution completed - Success: ${result?.success}`);
            await this._cleanupAndExit(result?.success ? 0 : 1);

        } catch (error) {
            logError(this.name, new Error(`HTN execution failed: ${error.message}`));
            await this._cleanupAndExit(1);
        }
    }

    async _cleanupAndExit(exitCode) {
        const forceExitTimeout = setTimeout(() => {
            console.warn(`[WARN] [${this.name}] Cleanup timed out, forcing exit...`);
            process.exit(exitCode);
        }, FORCE_EXIT_TIMEOUT);
        forceExitTimeout.unref();

        await this._shutdown();
        clearTimeout(forceExitTimeout);

        logInfo(this.name, `Exiting with code ${exitCode}`);
        process.exit(exitCode);
    }

    async _shutdown() {
        try {
            this._removeBotErrorHandlers();
            await this.shutdown();
        } catch (error) {
            console.warn(`[WARN] [${this.name}] Shutdown error: ${error.message}`);
        }
    }

    /**
     * Shutdown HTN agent and cleanup resources
     */
    async shutdown() {
        logInfo(this.name, `Shutting down HTN agent...`);
        this._removeBotErrorHandlers();
        await super.shutdown();
        logInfo(this.name, `Shutdown complete`);
    }

    _removeBotErrorHandlers() {
        if (this.bot) {
            if (this._onError)  this.bot.removeListener('error',  this._onError);
            if (this._onKicked) this.bot.removeListener('kicked', this._onKicked);
            if (this._onEnd)    this.bot.removeListener('end',    this._onEnd);
        }
        this._onError = null;
        this._onKicked = null;
        this._onEnd = null;
    }
}