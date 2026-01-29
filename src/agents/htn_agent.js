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
        this.lastPosition = null; // Track previous position for distance calc
        this.metrics = {
            version: null,
            agent_name: agentName,
            agent_type: 'htn',
            task: null,
            success: false,
            start_time: null,
            end_time: null,
            time_elapsed_s: 0,
            steps_taken: 0,
            exploration_distance: 0,
            movement_path: [], // Array of {x,y,z,timestamp} for 3D path export
            actions: [], // Changed to array to store sequence of actions
            action_counts: {}, // New object for counting occurrences
            // Podemos hacer que esto sea de los fallos que ocurren durante la ejecución. i.e: se le rompe el pico
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
            
            // Set metrics export path - if it's a directory, add filename; if null or empty, set default
            if (settings.metrics_export_path) {
                const metricsPath = settings.metrics_export_path;
                // If path ends with / or has no extension, treat as directory and add filename
                if (metricsPath.endsWith('/') || !metricsPath.includes('.')) {
                    this.metricsExportPath = `${metricsPath.replace(/\/$/, '')}/metrics_${this.name}_${Date.now()}.json`;
                } else {
                    this.metricsExportPath = metricsPath;
                }
            } else {
                // Default metrics path with timestamp and agent name
                this.metricsExportPath = `src/agents/metrics/metrics_${this.name}_${Date.now()}.json`;
            }
            
            this.metrics.task = settings.task?.goal || 'Default HTN progression';
            this.metrics.start_time = new Date().toISOString();

            // Clear/load memory
            await this.clearMemory(loadMemory);

            // Register with mindserver
            console.log(`[${this.name}] Registering with MindServer...`);
            await serverProxy.connect(this.name, settings.mindserver_port || 8080);

            // Connect to Minecraft
            await this.connectBot(settings);
            
            if (this.bot) {
                this.lastPosition = this.bot.entity.position.clone();
                
                // Track movement distance
                this.bot.on('move', () => {
                    if (this.lastPosition) {
                        const currentPos = this.bot.entity.position;
                        const dist = this.lastPosition.distanceTo(currentPos);
                        // Only add significant movement to avoid jitter
                        if (dist > 0.05) { 
                            this.metrics.exploration_distance += dist;
                            this.metrics.movement_path.push({
                                x: currentPos.x,
                                y: currentPos.y,
                                z: currentPos.z,
                                timestamp: new Date().toISOString()
                            });
                            this.lastPosition = currentPos.clone();
                        }
                    }
                });
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
                console.log(`[${this.name}] Created metrics directory: ${dir}`);
            }

            // Add current bot state to metrics
            if (this.bot) {
                this.metrics.version = this.bot.version;
                try {
                    // Ir al contenido de la referencia (hash del ultimo commit)
                    const refPath = '.git/' + fs.readFileSync('.git/HEAD', 'utf-8').trim().split(' ')[1];
                    const refContent = fs.readFileSync(refPath, 'utf-8');
                    this.metrics.version = refContent.toString().trim();
                } catch (e) {
                    console.warn(`[${this.name}] No se pudo obtener la versión de Git: ${e.message}`);
                    this.metrics.version = 'unknown';
                }
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

            // Export CSV path for Blender (x,y,z,timestamp)
            const csvPath = this.metricsExportPath.replace(/\.json$/, '') + '_path.csv';
            const csvHeader = 'x,y,z,timestamp\n';
            const csvBody = this.metrics.movement_path
                .map(p => `${p.x},${p.y},${p.z},${p.timestamp}`)
                .join('\n');
            fs.writeFileSync(csvPath, csvHeader + csvBody + (csvBody ? '\n' : ''));
            
            console.log(`[${this.name}] Metrics exported to ${this.metricsExportPath}`);
            console.log(`[${this.name}] Path CSV exported to ${csvPath}`);
        } catch (error) {
            console.error(`[${this.name}] Failed to export metrics:`, error.message);
        }
    }

    /**
     * Track action (called by HTN primitives)
     */
    async trackAction(actionName) {
        this.metrics.steps_taken++;
        
        // Record sequence of actions
        this.metrics.actions.push({
            name: actionName,
            timestamp: new Date().toISOString()
        });

        // Count occurrences
        this.metrics.action_counts[actionName] = (this.metrics.action_counts[actionName] || 0) + 1;
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
