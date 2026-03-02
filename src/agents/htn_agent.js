

import { BaseAgent } from './types/base_agent.js';
import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import { startHTN } from '../htn/main_htn.js';
import { MetricsCollector } from '../metrics/metrics_collector.js';

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, `htn`);
        // Recolect metrics specific to HTN execution
        this.metricsCollector = new MetricsCollector(agentName, `htn`);
        this.memoryPath = `src/agents/memories/${agentName}_memory.json`;
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
            
            // Start metrics tracking
            console.log(`[INFO] [${this.name}] Starting world tracking...`);
            this.metricsCollector.startWorldTracking(this.bot);
            
            console.log(`[INFO] [${this.name}] Starting control tracking with bot=${this.bot ? 'valid' : 'null'}, viewerPort=${viewerPort}`);
            try {
                this.metricsCollector.startControlTracking(this.bot, 50, true, viewerPort);
            } catch (err) {
                console.error(`[ERROR] [${this.name}] Failed to start control tracking:`, err.message);
                console.error(err.stack);
            }

            // Setup viewer
            this.setupViewer(viewerPort);

            // Mock components required by getFullState
            // Required by ServerProxy
            this.bot.modes = { getMiniDocs: () => `HTN Mode` };
            this.actions = { currentActionLabel: `HTN Task Execution` };

            // Handle graceful shutdown on Ctrl+C
            const handleShutdown = async (signal) => {
                console.log(`\n[INFO] [${this.name}] Received ${signal}, shutting down gracefully...`);
                this.metricsCollector.recordError(`Interrupted by ${signal}`);
                await this.metricsCollector.export(this.bot);
                await this.shutdown();
                process.exit(0);
            };
            process.on(`SIGINT`, () => handleShutdown(`SIGINT`));
            process.on(`SIGTERM`, () => handleShutdown(`SIGTERM`));

            // Start HTN logic
            console.log(`[INFO] [${this.name}] Starting HTN execution...`);
            const inventoryPort = viewerPort + 1000;
            await this.runLogic(inventoryPort);

        } catch (error) {
            console.error(`[ERROR] [${this.name}] Failed to start:`, error.message);
            this.metricsCollector.recordError(error.message);
            await this.metricsCollector.export(this.bot);
            await this.shutdown();
            throw error;
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
            await this.metricsCollector.export(this.bot);
            
        } catch (error) {
            this.metricsCollector.recordError(error.message);
            this.metricsCollector.completeTask(false);
            
            console.error(`[ERROR] [${this.name}] HTN execution failed:`, error);
            await this.metricsCollector.export(this.bot);
            throw error;
        }
    }
}