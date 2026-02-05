/**
 * HTNAgent - Agent that uses Hierarchical Task Network planning
 * Extends BaseAgent to implement HTN-specific logic
 */

import { BaseAgent } from './base_agent.js';
import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import { startHTN } from '../htn/main_htn.js';
import { MetricsCollector } from '../metrics/metrics_collector.js';
import fs from 'fs';
import path from 'path';

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, 'htn');
        this.htnRunner = null;
        this.metricsCollector = new MetricsCollector(agentName, 'htn');
    }

    /**
     * Full startup sequence for HTN agent
     * 
     * @param {Object} settings - Minecraft connection settings
     * @param {number} viewerPort - Port for browser viewer
     * @param {number} countId - ID for multi-agent scenarios
     * @param {boolean} loadMemory - Load previous memory if true
     * @param {string} initMessage - Initial message to send (unused for HTN)
     */
    async start(settings, viewerPort, countId = 0, loadMemory = false, initMessage = null) {
        try {
            this.countId = countId;
            this.memoryPath = `src/agents/memories/${this.name}_memory.json`;
            
            // Initialize metrics collection
            await this.metricsCollector.initialize(
                settings.metrics_export_path,
                settings.task?.goal || 'Default HTN progression',
                true,
                viewerPort
            );

            // Clear/load memory
            await this.clearMemory(loadMemory);

            // Register with mindserver
            console.log(`[${this.name}] Registering with MindServer...`);
            await serverProxy.connect(this.name, settings.mindserver_port || 8080);

            // Connect to Minecraft
            await this.connectBot(settings);
            
            if (this.bot) {
                // Start movement tracking
                this.metricsCollector.startWorldTracking(this.bot);
            }

            // Setup viewer
            this.setupViewer(viewerPort);

            // Mock components required by getFullState
            this.bot.modes = { getMiniDocs: () => 'HTN Mode' };
            this.actions = { currentActionLabel: 'HTN Task Execution' };

            // Attach trackAction method to bot for HTN tasks
            this.bot.trackAction = this.trackAction.bind(this);

            // Start HTN logic
            console.log(`[${this.name}] Starting HTN execution...`);
            const inventoryPort = viewerPort + 1000;
            await this.runLogic(inventoryPort);

        } catch (error) {
            console.error(`[${this.name}] Failed to start:`, error.message);
            this.metricsCollector.recordError(error.message);
            await this.metricsCollector.export(this.bot);
            await this.shutdown();
            throw error;
        }
    }

    /**
     * Execute HTN task progression
     */
    async runLogic(inventoryPort = 3001) {
        try {
            // startHTN is an async function that manages task execution
            const result = await startHTN(this.bot, inventoryPort);
            
            this.metrics.success = result?.success || false;
            this.metrics.end_time = new Date().toISOString();
            this.metrics.time_elapsed_s = 
                (new Date(this.metrics.end_time) - new Date(this.metrics.start_time)) / 1000;
            
            console.log(`[${this.name}] HTN execution completed - Success: ${this.metrics.success}`);
            await this.exportMetrics();
            
        } catch (error) {
            this.metrics.success = false;
            this.metrics.errors.push(error.message);
            this.metrics.end_time = new Date().toISOString();
            this.metrics.time_elapsed_s = 
                (new Date(this.metrics.end_time) - new Date(this.metrics.start_time)) / 1000;
            
            console.error(`[${this.name}] HTN execution failed:`, error);
            await this.exportMetrics();
            throw error;
        }
    }

    /**
     * Execute HTN task progression
     */
    async runLogic(inventoryPort = 3001) {
        try {
            // startHTN is an async function that manages task execution
            const result = await startHTN(this.bot, inventoryPort, this.metricsCollector);
            
            this.metricsCollector.completeTask(result?.success || false);
            
            console.log(`[${this.name}] HTN execution completed - Success: ${result?.success}`);
            await this.metricsCollector.export(this.bot);
            
        } catch (error) {
            this.metricsCollector.recordError(error.message);
            this.metricsCollector.completeTask(false);
            
            console.error(`[${this.name}] HTN execution failed:`, error);
            await this.metricsCollector.export(this.bot);
            throw error;
        }
    }         
    /**
     * Track action (called by HTN primitives)
     */
    async trackAction(actionName) {
        this.metricsCollector.trackAction(actionName);
    }
}