/**
 * MetricsCollector - Handles collection and export of agent metrics
 * Tracks execution metrics, movement paths, actions, and errors
 */

import fs from 'fs';
import path from 'path';

export class MetricsCollector {
    constructor(agentName, agentType) {
        this.agentName = agentName;
        this.agentType = agentType;
        this.exportPath = null;
        this.lastPosition = null;
        
        this.metrics = {
            version: null,
            agent_name: agentName,
            agent_type: agentType,
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
            errors: []
        };
    }

    /**
     * Initialize metrics collection
     * @param {string} exportPath - Path to export metrics (file or directory)
     * @param {string} task - Task description
     */
    initialize(exportPath, task) {
        // Set metrics export path - if it's a directory, add filename; if null or empty, set default
        if (exportPath) {
            // If path ends with / or has no extension, treat as directory and add filename
            if (exportPath.endsWith('/') || !exportPath.includes('.')) {
                this.exportPath = `${exportPath.replace(/\/$/, '')}/metrics_${this.agentName}_${Date.now()}.json`;
            } else {
                this.exportPath = exportPath;
            }
        } else {
            // Default metrics path with timestamp and agent name
            this.exportPath = `src/metrics/agent_metrics/metrics_${this.agentName}_${Date.now()}.json`;
        }

        this.metrics.task = task || 'Default task';
        this.metrics.start_time = new Date().toISOString();
        
        console.log(`[${this.agentName}] Metrics initialized - Export path: ${this.exportPath}`);
    }

    /**
     * Start tracking bot movement
     * @param {Object} bot - Mineflayer bot instance
     */
    startMovementTracking(bot) {
        if (!bot) return;

        this.lastPosition = bot.entity.position.clone();

        bot.on('move', () => {
            if (this.lastPosition) {
                const currentPos = bot.entity.position;
                const dist = this.lastPosition.distanceTo(currentPos);
                // Only add significant movement to avoid jitter
                if (dist > 0.1) {
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

        console.log(`[${this.agentName}] Movement tracking started`);
    }

    /**
     * Track an action performed by the agent
     * @param {string} actionName - Name of the action
     */
    trackAction(actionName) {
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
     * Record an error
     * @param {string} errorMessage - Error message to record
     */
    recordError(errorMessage) {
        this.metrics.errors.push({
            message: errorMessage,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Mark task completion
     * @param {boolean} success - Whether the task succeeded
     */
    completeTask(success) {
        this.metrics.success = success;
        this.metrics.end_time = new Date().toISOString();
        this.metrics.time_elapsed_s = 
            (new Date(this.metrics.end_time) - new Date(this.metrics.start_time)) / 1000;
    }

    /**
     * Get current metrics snapshot
     * @returns {Object} Current metrics
     */
    getMetrics() {
        return { ...this.metrics };
    }

    /**
     * Export metrics to JSON file and CSV path data
     * @param {Object} bot - Optional bot instance to capture final state
     */
    async export(bot = null, csvExport = false) {
        if (!this.exportPath) {
            console.log(`[${this.agentName}] Metrics export disabled (no path specified)`);
            return;
        }

        try {
            // Ensure directory exists
            const dir = path.dirname(this.exportPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`[${this.agentName}] Created metrics directory: ${dir}`);
            }

            // Add final bot state to metrics if available
            if (bot) {
                // Get git version
                try {
                    const refPath = '.git/' + fs.readFileSync('.git/HEAD', 'utf-8').trim().split(' ')[1];
                    const refContent = fs.readFileSync(refPath, 'utf-8');
                    this.metrics.version = refContent.toString().trim();
                } catch (e) {
                    console.warn(`[${this.agentName}] No se pudo obtener la versión de Git: ${e.message}`);
                    this.metrics.version = bot.version || 'unknown';
                }

                this.metrics.final_position = bot.entity.position;
                this.metrics.final_inventory = bot.inventory.items().map(item => ({
                    name: item.name,
                    count: item.count
                }));
            }

            // Export JSON metrics
            fs.writeFileSync(
                this.exportPath,
                JSON.stringify(this.metrics, null, 2)
            );

            if (csvExport) {
                try {
                    // Export CSV path for 3D visualization (e.g., Blender)
                    const csvPath = this.exportPath.replace(/\.json$/, '') + '_path.csv';
                    const csvHeader = 'x,y,z,timestamp\n';
                    const csvBody = this.metrics.movement_path
                        .map(p => `${p.x},${p.y},${p.z},${p.timestamp}`)
                        .join('\n');
                    fs.writeFileSync(csvPath, csvHeader + csvBody + (csvBody ? '\n' : ''));

                    console.log(`[${this.agentName}] Metrics exported to ${this.exportPath}`);
                    console.log(`[${this.agentName}] Path CSV exported to ${csvPath}`);
                } catch (csvError) {
                    console.error(`[${this.agentName}] Failed to export path CSV:`, csvError.message);
                }
            } else {
                console.log(`[${this.agentName}] No CSV export requested.`);
            }
        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics:`, error.message);
        }
    }
}
