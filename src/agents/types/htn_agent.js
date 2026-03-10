import { BaseAgent } from './base_agent.js';
import { startChopTrees, startHTN } from '../../htn/main_htn.js';
import { MetricsCollector } from '../../metrics/metrics_collector.js';
import { logInfo, logError } from '../logging.js';

const FORCE_EXIT_TIMEOUT = 45000; // 45 seconds (15s screenshot flush + 30s buffer)

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, `htn`);
        this.metricsCollector = new MetricsCollector(agentName, `htn`);
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
            await this._initializeMetrics(settings, viewerPort);
            await this._setupTracking(viewerPort);
            this._setupBotErrorHandlers();

            logInfo(this.name, `Starting HTN execution...`);
            await this.runLogic();

        } catch (error) {
            logError(this.name, new Error(`Failed to start: ${error.message}`));
            this.metricsCollector.recordError(error.message);
            await this._exportAndShutdown();
            process.exit(1);
        }
    }

    /**
     * Initialize metrics collection
     */
    async _initializeMetrics(settings, viewerPort) {
        const task = settings.task?.goal || `Default HTN progression`;
        const metricsPath = settings.metrics_export_path || './metrics';
        
        try {
            // No habilitar screenshots aquí; se habilitan solo si viewer arranca
            await this.metricsCollector.initialize(
                metricsPath,
                task,
                false,
                null
            );

            await this.clearMemory();
            logInfo(this.name, `Metrics initialized at ${metricsPath}`);
        } catch (error) {
            logError(this.name, new Error(`Metrics initialization failed: ${error.message}`));
            throw error;
        }
    }

    /**
     * Setup world and control tracking
     */
    async _setupTracking(viewerPort) {
        try {
            logInfo(this.name, `Setting up tracking...`);
            
            const viewerOk = await this.setupViewer(viewerPort);
            
            // Pre-initialize the screenshot browser BEFORE starting tracking
            // so the first action doesn't have a cold-start delay
            if (viewerOk) {
                logInfo(this.name, `Warming up screenshot browser...`);
                this.metricsCollector.screenshotManager.enable(viewerPort);
                const browserReady = await this.metricsCollector.warmupScreenshots();
                if (browserReady) {
                    logInfo(this.name, `Screenshot browser is ready`);
                } else {
                    logError(this.name, new Error(`Screenshot browser failed to warm up`));
                }
            }

            logInfo(this.name, `Starting control tracking...`);
            this.metricsCollector.startControlTracking(
                this.bot,
                150,       // poll interval — must be >= screenshot capture time (~100ms)
                viewerOk,
                viewerOk ? viewerPort : null
            );
        } catch (error) {
            logError(this.name, new Error(`Tracking setup failed: ${error.message}`));
            process.exit(1);
        }
    }

    /**
     * Setup bot error and disconnect handlers
     */
    _setupBotErrorHandlers() {
        this.bot.on('error', (err) => {
            logError(this.name, new Error(`Bot error: ${err.message}`));
            this.metricsCollector.recordError(`Bot error: ${err.message}`);
            this._handleBotDisconnect();
        });

        this.bot.on('kicked', (reason) => {
            logError(this.name, new Error(`Bot kicked: ${reason}`));
            this.metricsCollector.recordError(`Bot kicked: ${reason}`);
            this._handleBotDisconnect();
        });

        this.bot.on('end', (reason) => {
            console.warn(`[WARN] [${this.name}] Bot connection ended: ${reason}`);
            this.metricsCollector.stopControlTracking();
            // Stop new screenshot captures (existing queue will finish)
            this.metricsCollector.stopScreenshots();
        });
    }

    /**
     * Handle bot disconnection with cleanup
     */
    async _handleBotDisconnect() {
        this.metricsCollector.completeTask(false);
        await this._exportAndShutdown();
        process.exit(1);
    }

    /**
     * Execute HTN task progression
     */
    async runLogic() {
        try {
            const result = await startChopTrees(this.bot);
            this.metricsCollector.completeTask(result?.success || false);
            
            logInfo(this.name, `HTN execution completed - Success: ${result?.success}`);
            await this._cleanupAndExit(0);

        } catch (error) {
            this.metricsCollector.recordError(error?.message || String(error));
            this.metricsCollector.completeTask(false);
            
            logError(this.name, new Error(`HTN execution failed: ${error.message}`));
            await this._cleanupAndExit(1);
        }
    }

    /**
     * Cleanup and exit with timeout protection
     */
    async _cleanupAndExit(exitCode) {
        const forceExitTimeout = setTimeout(() => {
            console.warn(`[WARN] [${this.name}] Cleanup timed out, forcing exit...`);
            process.exit(exitCode);
        }, FORCE_EXIT_TIMEOUT);
        forceExitTimeout.unref();

        await this._exportAndShutdown();
        clearTimeout(forceExitTimeout);
        
        logInfo(this.name, `Exiting with code ${exitCode}`);
        process.exit(exitCode);
    }

    /**
     * Export metrics and shutdown
     */
    async _exportAndShutdown() {
        try {
            await this.metricsCollector.export(this.bot);
            logInfo(this.name, `Metrics exported`);
        } catch (error) {
            console.warn(`[WARN] [${this.name}] Metrics export failed: ${error.message}`);
        }

        try {
            await this.shutdown();
            logInfo(this.name, `Shutdown complete`);
        } catch (error) {
            console.warn(`[WARN] [${this.name}] Shutdown error: ${error.message}`);
        }
    }

    /**
     * Track action (called by HTN primitives)
     */
    trackAction(actionName) {
        this.metricsCollector.controlTracker?.trackAction(actionName);
    }

    /**
     * Shutdown HTN agent and cleanup resources
     */
    async shutdown() {
        logInfo(this.name, `Shutting down HTN agent...`);
        this._stopTracking();
        await super.shutdown();
        logInfo(this.name, `Shutdown complete`);
    }

    /**
     * Stop all tracking
     */
    _stopTracking() {
        this.metricsCollector.stopControlTracking();
    }
}