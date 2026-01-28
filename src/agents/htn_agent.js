/**
 * HTNAgent - Agent that uses Hierarchical Task Network planning
 * Extends BaseAgent to implement HTN-specific logic
 */

import { BaseAgent } from './base_agent.js';
import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import { startHTN } from '../htn/main_htn.js';
import fs from 'fs';
import path from 'path';

export class HTNAgent extends BaseAgent {
    constructor(agentName) {
        super(agentName, 'htn');
        this.htnRunner = null;
        this.metrics = {
            agent_name: agentName,
            agent_type: 'htn',
            task: null,
            success: false,
            start_time: null,
            end_time: null,
            time_elapsed_s: 0,
            steps_taken: 0,
            exploration_distance: 0,
            actions: {},
            errors: []
        };
        this.metricsExportPath = null;
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
            this.metricsExportPath = settings.metrics_export_path;
            this.metrics.task = settings.task?.goal || 'Default HTN progression';
            this.metrics.start_time = new Date().toISOString();

            // Clear/load memory
            await this.clearMemory(loadMemory);

            // Register with mindserver
            console.log(`[${this.name}] Registering with MindServer...`);
            await serverProxy.connect(this.name, settings.mindserver_port || 8080);

            // Connect to Minecraft
            await this.connectBot(settings);

            // Setup viewer
            this.setupViewer(viewerPort);

            // Mock components required by getFullState
            this.bot.modes = { getMiniDocs: () => 'HTN Mode' };
            this.actions = { currentActionLabel: 'HTN Task Execution' };

            // Start HTN logic
            console.log(`[${this.name}] Starting HTN execution...`);
            const inventoryPort = viewerPort + 1000;
            await this.runLogic(inventoryPort);

        } catch (error) {
            console.error(`[${this.name}] Failed to start:`, error.message);
            this.metrics.errors.push(error.message);
            await this.exportMetrics();
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
            
            this.metrics.success = result?.success || true;
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
     * Export metrics to JSON file
     */
    async exportMetrics() {
        if (!this.metricsExportPath) {
            console.log(`[${this.name}] Metrics export disabled (no path specified)`);
            return;
        }

        try {
            // Ensure directory exists
            const dir = path.dirname(this.metricsExportPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }

            // Add current bot state to metrics
            if (this.bot) {
                this.metrics.final_position = this.bot.entity.position;
                this.metrics.final_inventory = this.bot.inventory.items().map(item => ({
                    name: item.name,
                    count: item.count
                }));
            }

            fs.writeFileSync(
                this.metricsExportPath,
                JSON.stringify(this.metrics, null, 2)
            );
            
            console.log(`[${this.name}] Metrics exported to ${this.metricsExportPath}`);
        } catch (error) {
            console.error(`[${this.name}] Failed to export metrics:`, error.message);
        }
    }

    /**
     * Track action (called by HTN primitives)
     */
    trackAction(actionName) {
        this.metrics.steps_taken++;
        this.metrics.actions[actionName] = (this.metrics.actions[actionName] || 0) + 1;
    }

    /**
     * Return HTN-specific status
     */
    getStatus() {
        return {
            name: this.name,
            type: 'htn',
            online: this.bot && this.bot.players[this.name],
            position: this.bot ? this.bot.entity.position : null,
            inventory: this.bot ? this.bot.inventory.items() : []
        };
    }
}
