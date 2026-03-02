

import { BaseAgent } from './base_agent.js';
import { serverProxy } from '../../llm/src/agent/mindserver_proxy.js';
import { startHTN } from '../../htn/main_htn.js';
import { MetricsCollector } from '../../metrics/metrics_collector.js';

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, `htn`);
        // Recolect metrics specific to HTN execution
        this.metricsCollector = new MetricsCollector(agentName, `htn`);
        this.memoryPath = `src/agents/memories/${agentName}_memory.json`;
        // Store signal handlers for cleanup
        this.signalHandlers = [];
    }

    /**
     * Full startup sequence for HTN agent
     * 
     * @param {Object} settings - Minecraft connection settings
     * @param {number} viewerPort - Port for browser viewer
     * @param {boolean} clearMemory - Clear previous memory if true
     */
    async start(settings, viewerPort, clearMemory = true) {
        try {
            // Initialize metrics collection
            await this.metricsCollector.initialize(
                settings.metrics_export_path,
                settings.task?.goal || `Default HTN progression`,
                true,
                viewerPort
            );

            // Clear/load memory
            await this.clearMemory(clearMemory);

            // Register with mindserver
            console.log(`[INFO] [${this.name}] Registering with MindServer...`);
            await serverProxy.connect(this.name, settings.mindserver_port || 8080);

            // Connect to Minecraft
            await this.connectBot(settings);
            
            // Start metrics tracking with control tracking and screenshots
            console.log(`[INFO] [${this.name}] Starting world tracking...`);
            this.metricsCollector.startWorldTracking(this.bot);
            
            console.log(`[INFO] [${this.name}] Calling startControlTracking with bot=${this.bot ? 'valid' : 'null'}, viewerPort=${viewerPort}...`);
            try {
                this.metricsCollector.startControlTracking(this.bot, 50, true, viewerPort);
            } catch (err) {
                console.error(`[ERROR] [${this.name}] Failed to start control tracking:`, err.message);
            }

            // Setup viewer
            this.setupViewer(viewerPort);

            // Mock components required by getFullState
            // Required by ServerProxy
            this.bot.modes = { getMiniDocs: () => `HTN Mode` };
            this.actions = { currentActionLabel: `HTN Task Execution` };

            // Attach trackAction method to bot for HTN tasks
            this.bot.trackAction = this.trackAction.bind(this);

            // Handle graceful shutdown on Ctrl+C
            const handleSIGINT = async () => {
                console.log(`\n[INFO] [${this.name}] Received SIGINT, shutting down gracefully...`);
                this.metricsCollector.recordError(`Interrupted by SIGINT`);
                await this.metricsCollector.export(this.bot);
                await this.shutdown();
                process.exit(130); // Standard exit code for SIGINT
            };
            const handleSIGTERM = async () => {
                console.log(`\n[INFO] [${this.name}] Received SIGTERM, shutting down gracefully...`);
                this.metricsCollector.recordError(`Interrupted by SIGTERM`);
                await this.metricsCollector.export(this.bot);
                await this.shutdown();
                process.exit(143); // Standard exit code for SIGTERM
            };
            process.on(`SIGINT`, handleSIGINT);
            process.on(`SIGTERM`, handleSIGTERM);
            this.signalHandlers = [
                { signal: 'SIGINT', handler: handleSIGINT },
                { signal: 'SIGTERM', handler: handleSIGTERM }
            ];

            // Handle bot disconnection during execution (keepalive timeout, kicked, etc.)
            this.bot.on('error', (err) => {
                console.error(`[ERROR] [${this.name}] Bot error during execution: ${err.message}`);
                this.metricsCollector.recordError(`Bot error: ${err.message}`);
                this.metricsCollector.completeTask(false);
                this.metricsCollector.export(this.bot)
                    .catch(() => {})
                    .finally(() => {
                        this.shutdown()
                            .catch(() => {})
                            .finally(() => process.exit(1));
                    });
            });
            this.bot.on('kicked', (reason) => {
                console.error(`[ERROR] [${this.name}] Bot kicked during execution: ${reason}`);
                this.metricsCollector.recordError(`Bot kicked: ${reason}`);
                this.metricsCollector.completeTask(false);
                this.metricsCollector.export(this.bot)
                    .catch(() => {})
                    .finally(() => {
                        this.shutdown()
                            .catch(() => {})
                            .finally(() => process.exit(1));
                    });
            });
            this.bot.on('end', (reason) => {
                console.warn(`[WARN] [${this.name}] Bot connection ended: ${reason}`);
            });

            // Start HTN logic
            console.log(`[INFO] [${this.name}] Starting HTN execution...`);
            const inventoryPort = viewerPort + 1000;
            await this.runLogic(inventoryPort);

        } catch (error) {
            console.error(`[ERROR] [${this.name}] Failed to start:`, error.message);
            this.metricsCollector.recordError(error.message);
            await this.metricsCollector.export(this.bot);
            await this.shutdown();
            process.exit(1);
        }
    }


    /**
     * Execute HTN task progression
     * @param {number} inventoryPort - Port for inventory viewer (if used)
     * @return {Promise<void>}
     */
    async runLogic(inventoryPort = 3001) {
        try {
            // startHTN is an async function that manages task execution
            const result = await startHTN(this.bot, inventoryPort, this.metricsCollector);
            
            // Record final result in metrics
            this.metricsCollector.completeTask(result?.success || false);
            
            console.log(`[INFO] [${this.name}] HTN execution completed - Success: ${result?.success}`);
            
            // Safety net: force exit after 10 seconds if cleanup hangs
            const forceExitTimeout = setTimeout(() => {
                console.warn(`[WARN] [${this.name}] Cleanup timed out, forcing exit...`);
                process.exit(0);
            }, 10000);
            forceExitTimeout.unref(); // Don't let this timer keep the process alive
            
            try {
                await this.metricsCollector.export(this.bot);
                console.log(`[INFO] [${this.name}] Metrics exported.`);
            } catch (e) {
                console.warn(`[WARN] [${this.name}] Metrics export failed: ${e.message}`);
            }

            try {
                await this.shutdown();
                console.log(`[INFO] [${this.name}] Shutdown complete.`);
            } catch (e) {
                console.warn(`[WARN] [${this.name}] Shutdown error: ${e.message}`);
            }
            
            clearTimeout(forceExitTimeout);
            console.log(`[INFO] [${this.name}] Exiting process...`);
            process.exit(0);
            
        } catch (error) {
            this.metricsCollector.recordError(error?.message || String(error));
            this.metricsCollector.completeTask(false);
            
            console.error(`[ERROR] [${this.name}] HTN execution failed:`, error);
            
            // Safety net for error path too
            const forceExitTimeout = setTimeout(() => {
                console.warn(`[WARN] [${this.name}] Error cleanup timed out, forcing exit...`);
                process.exit(1);
            }, 10000);
            forceExitTimeout.unref();
            
            try { await this.metricsCollector.export(this.bot); } catch (_) {}
            try { await this.shutdown(); } catch (_) {}
            
            clearTimeout(forceExitTimeout);
            process.exit(1);
        }
    }         

    /**
     * Track action (called by HTN primitives)
     * @param {string} actionName - Name of the action being executed
     * @return {void}
     */
    async trackAction(actionName) {
        this.metricsCollector.trackAction(actionName);
    }

    /**
     * Shutdown HTN agent and cleanup resources
     * @return {Promise<void>}
     */
    async shutdown() {
        console.log(`[INFO] [${this.name}] Shutting down HTN agent...`);
        
        // Remove signal handlers to prevent interference
        if (this.signalHandlers) {
            for (const { signal, handler } of this.signalHandlers) {
                process.removeListener(signal, handler);
            }
            this.signalHandlers = [];
        }
        
        // Stop metrics tracking and close browser
        await this.metricsCollector.stopWorldTracking();
        this.metricsCollector.stopControlTracking();
        
        // Disconnect from MindServer
        try {
            await serverProxy.disconnect();
        } catch (e) {
            console.warn(`[WARN] [${this.name}] Error disconnecting from MindServer:`, e.message);
        }
        
        // Call base shutdown to quit bot
        await super.shutdown();
        
        console.log(`[INFO] [${this.name}] Shutdown complete`);
    }
}