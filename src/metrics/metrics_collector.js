import fs from 'fs';
import path from 'path';

export class MetricsCollector {
    constructor(agentName, agentType) {
        this.agentName = agentName;
        this.agentType = agentType;
        this.exportPath = null;
        this.lastPosition = null;
        this.trackingInterval = null; // How many intervals to track in a second
        this.currentAction = null; // Track ongoing action
        
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
            // Movements, actions, and observations for world model export
            // The actions are minimal; i.e: mine, move, jump, etc.
            // A list containing: {
            //   x, y, z: floats, 
            //   timestamp: date,
            //   action: str,
            //   success: bool,
            //   observation: png
            // }
            world_model: [], 
            actions: [], // Name of the action and timestamp
            action_counts: {}, // Count of each action performed
            errors: []
        };
    }

    /**
     * Capture current inventory state
     * @param {Object} bot - Mineflayer bot instance
     * @returns {Object} Inventory snapshot
     */
    captureInventoryState(bot) {
        if (!bot) return {};
        return bot.inventory.items().map(item => ({
            name: item.name,
            id: item.type,
            count: item.count
        }));
    }

    /**
     * Start tracking an action
     * @param {string} actionName - Name of the action
     * @param {Object} bot - Mineflayer bot instance
     */
    trackActionStart(actionName, bot) {
        this.currentAction = {
            name: actionName,
            startTime: new Date().toISOString(),
            startInventory: this.captureInventoryState(bot)
        };
    }

    /**
     * End action tracking with success/failure status
     * Records actions to world_model with detailed state information
     * @param {boolean} success - Whether the action succeeded
     * @param {Object} bot - Mineflayer bot instance
     */
    trackActionEnd(success, bot) {
        if (!this.currentAction) return;
        
        const endInventory = this.captureInventoryState(bot);
        const completedAction = {
            name: this.currentAction.name,
            success,
            startTime: this.currentAction.startTime,
            endTime: new Date().toISOString(),
            duration: (new Date() - new Date(this.currentAction.startTime)) / 1000
        };

        this.metrics.actions.push(completedAction);
        
        // Record action completion to world_model as a snapshot entry with all completed actions so far
        this.metrics.world_model.push({
            name: this.currentAction.name,
            x: bot.entity.position.x,
            y: bot.entity.position.y,
            z: bot.entity.position.z,
            success: success,
            img: null, // placeholder for image data
            timestamp: new Date().toISOString()
        });
        
        this.metrics.action_counts[this.currentAction.name] = 
            (this.metrics.action_counts[this.currentAction.name] || 0) + 1;
        
        this.currentAction = null;
    }

    /**
     * Compare two inventory states
     * @param {Array} startInv - Starting inventory
     * @param {Array} endInv - Ending inventory
     * @returns {Array} Changed items
     */
    didInventoryChange(startInv, endInv) {
        const changes = [];
        const endMap = new Map(endInv.map(item => [item.name, item.count]));
        
        for (const startItem of startInv) {
            const endCount = endMap.get(startItem.name) || 0;
            if (endCount !== startItem.count) {
                changes.push({
                    item: startItem.name,
                    delta: endCount - startItem.count
                });
            }
        }
        
        // Check for new items
        const startMap = new Map(startInv.map(item => [item.name, item.count]));
        for (const endItem of endInv) {
            if (!startMap.has(endItem.name)) {
                changes.push({
                    item: endItem.name,
                    delta: endItem.count
                });
            }
        }
        
        return changes.length > 0 ? changes : null;
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
                // exportPath/metrics_agentName_timestamp.json
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
     * Start tracking bot observations and movements
     * Distinguishes between 'idle' (not moving) and actual actions being performed
     * @param {Object} bot - Mineflayer bot instance
     * @param {number} samplingRate - Samples per second (default: 10)
     */
    startWorldTracking(bot, samplingRate = 1) {
        if (!bot) return;

        // Copy the initial position
        this.lastPosition = bot.entity.position.clone();

        const intervalMs = 1000 / samplingRate;
        const movementThreshold = 0.05; // Minimum distance to consider as "moving"

        // Sample bot state at fixed intervals
        this.trackingInterval = setInterval(() => {
            const currentPos = bot.entity.position;
            const dist = this.lastPosition.distanceTo(currentPos);

            // Update exploration distance if moved
            this.metrics.exploration_distance += dist; // If we don't move, the sum stays the same

            // Determine current action state
            let actionState = 'idle';

            // If there's a current action being tracked, use that
            if (this.currentAction) {
                actionState = this.currentAction.name;
            } else {
                // Check if bot is actually moving (not idle)
                if (dist > movementThreshold) {
                    actionState = 'moving';
                } else {
                    actionState = 'idle';
                }
            }

            // Update last position after computing movement
            this.lastPosition = currentPos.clone();
            this.metrics.action_counts[actionState] = (this.metrics.action_counts[actionState] || 0) + 1;
            // Capture complete bot state with all completed actions
            this.metrics.world_model.push({
                name: actionState,
                x: currentPos.x,
                y: currentPos.y,
                z: currentPos.z,
                success: this.currentAction ? null : undefined, // null if action in progress, undefined if idle
                img: null, // placeholder for image data
                timestamp: new Date().toISOString()
            });
        }, intervalMs);

        console.log(`[${this.agentName}] World tracking started at ${samplingRate} samples/second`);
    }





    /**
     * Stop world tracking and clear interval
     */
    stopWorldTracking() {
        if (this.trackingInterval) {
            clearInterval(this.trackingInterval);
            this.trackingInterval = null;
            console.log(`[${this.agentName}] World tracking stopped`);
        }
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
     * Export metrics to JSON file
     * @param {Object} bot - Optional bot instance to capture final state
     */
    async export(bot = null) {
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
                await this.getVersion(bot);
            }

            // Export JSON metrics
            await this.exportJSON();

        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics:`, error.message);
        }
    }

    /**
     * 
     * @param {*} bot 
     */
    async getVersion(bot) {
        // Get git version
        try {
            // Obtain the current git branch
            const refPath = '.git/' + fs.readFileSync('.git/HEAD', 'utf-8').trim().split(' ')[1];
            // Obtain the commit hash from the ref file
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

    /**
     * Export metrics to JSON file
     */
    async exportJSON() {
        fs.writeFileSync(
            this.exportPath,
            JSON.stringify(this.metrics, null, 2)
        );
    }
}
