import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';

export class MetricsCollector {
    constructor(agentName, agentType) {
        this.agentName = agentName;
        this.agentType = agentType;
        this.exportPath = null;
        this.lastPosition = null;
        this.trackingInterval = null; // How many intervals to track in a second
        this.currentAction = null; // Track ongoing action
        this.captureScreenshots = false;
        this.viewerPort = null;
        this.screenshotsDir = null;
        this.screenshotFormat = 'png';
        // Could be modified to be 256x256 and not resize un the VAE
        this.screenshotWidth = 1024;
        this.screenshotHeight = 768;
        this.browser = null;
        this.page = null;
        this.screenshotQueue = Promise.resolve(null);
        
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
            //   success: bool
            // }
            world_model: [], 
            actions: [], // Name of the action and timestamp
            action_counts: {}, // Count of each action performed
            errors: []
        };
    }

    /**
     * =============================================================================
     * ================================ UTILS FUNCTIONS ============================
     * =============================================================================
     */

    /**
     * Capture current inventory state
     * @param {Object} bot - Mineflayer bot instance
     * @returns {Object} Inventory snapshot
     */
    captureInventoryState(bot) {
        if (!bot) return {};
        // Iterate through bot inventory and capture item names and counts
        return bot.inventory.items().map(item => ({
            name: item.name,
            id: item.type,
            count: item.count
        }));
    }


    /**
     * Compare two inventory states
     * @param {Array} startInv - Starting inventory
     * @param {Array} endInv - Ending inventory
     * @returns {Array} Changed items
     */
    didInventoryChange(startInv, endInv) {
        const changes = [];
        // Obtain a map of end inventory for lookup
        const endMap = new Map(endInv.map(item => [item.name, item.count]));
        
        // Check for changed or removed items
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
     * Records actions
     * @param {boolean} success - Whether the action succeeded
     * @param {Object} bot - Mineflayer bot instance
     */
    async trackActionEnd(success, bot) {
        if (!this.currentAction) return;
        
        // Capture current action immediately 
        const currentActionData = this.currentAction;
        // Reset the current action
        this.currentAction = null;
        
        const completedAction = {
            name: currentActionData.name,
            success,
            startTime: currentActionData.startTime,
            endTime: new Date().toISOString(),
            duration: (new Date() - new Date(currentActionData.startTime)) / 1000
        };

        // Push to the actions list in the json 
        this.metrics.actions.push(completedAction);
        
        const screenshotPath = await this._queueScreenshotCapture();

        // Record action completion to world_model
        this.metrics.world_model.push({
            x: bot.entity.position.x,
            y: bot.entity.position.y,
            z: bot.entity.position.z,
            action: completedAction,
            timestamp: completedAction.endTime,
            screenshot: screenshotPath
        });
        
        // Count occurrences of this action
        this.metrics.action_counts[currentActionData.name] = 
            (this.metrics.action_counts[currentActionData.name] || 0) + 1;
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
     * =============================================================================
     * ================================ MAIN FUNCTIONS =============================
     * =============================================================================
     */

    /**
     * Initialize metrics collection
     * @param {string} exportPath - Path to export metrics (file or directory)
     * @param {string} task - Task description
     * @param {boolean} captureScreenshots - Enable viewer screenshots for world_model entries
     * @param {number} viewerPort - Port for prismarine viewer
     */
    async initialize(exportPath, task, captureScreenshots = false, viewerPort = null) {
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

        this.captureScreenshots = Boolean(captureScreenshots);
        this.viewerPort = viewerPort;
        if (this.captureScreenshots) {
            this.screenshotsDir = `src/metrics/agent_metrics/${this.agentName}_screenshots/`;
        }
        
        console.log(`[${this.agentName}] Metrics initialized - Export path: ${this.exportPath}`);
    }

    /**
     * Start tracking bot observations and movements
     * Distinguishes between 'idle' (not moving) and actual actions being performed
     * @param {Object} bot - Mineflayer bot instance
     * @param {number} samplingRate - Samples per second (default: 1)
     */
    startWorldTracking(bot, samplingRate = 1) {
        if (!bot) return;

        // Copy the initial position
        this.lastPosition = bot.entity.position.clone();

        const intervalMs = 1000 / samplingRate;
        // More threshold than 0.05 can cause to always be considered as idle
        const movementThreshold = 0.05; // Minimum distance to consider as "moving"

        // Sample bot state at fixed intervals
        this.trackingInterval = setInterval(async () => {
            const currentPos = bot.entity.position;
            // Obtain the traveled distance since last sample
            const dist = this.lastPosition.distanceTo(currentPos);

            this.metrics.exploration_distance += dist;

            // Determine current action state
            let actionState = 'idle';
            if (this.currentAction) {
                actionState = this.currentAction.name;
            } else if (dist > movementThreshold) {
                actionState = 'moving';
            }

            // Update last position after computing movement
            this.lastPosition = currentPos.clone();
            this.metrics.action_counts[actionState] = (this.metrics.action_counts[actionState] || 0) + 1;
            
            // Create action object
            let actionObj = null;
            // Right now we only use 'mine'
            if (this.currentAction) {
                actionObj = {
                    name: this.currentAction.name,
                    success: null,
                    startTime: this.currentAction.startTime,
                    endTime: null,
                    duration: null
                };
            } else if (actionState === 'moving' || actionState === 'idle') {
                actionObj = {
                    name: actionState,
                    success: undefined,
                    startTime: null,
                    endTime: null,
                    duration: null
                };
            }
            
            const screenshotPath = await this._queueScreenshotCapture();

            // Capture bot state
            this.metrics.world_model.push({
                x: currentPos.x,
                y: currentPos.y,
                z: currentPos.z,
                action: actionObj,
                timestamp: new Date().toISOString(),
                screenshot: screenshotPath
            });
        }, intervalMs);

        console.log(`[${this.agentName}] World tracking started at ${samplingRate} samples/second`);
    }

    /**
     * Stop world tracking and clear interval
     */
    async stopWorldTracking() {
        if (this.trackingInterval) {
            clearInterval(this.trackingInterval);
            this.trackingInterval = null;
            console.log(`[${this.agentName}] World tracking stopped`);
        }

        if (this.browser) {
            await this.browser.close();
            this.browser = null;
            this.page = null;
        }
    }

    /**
     * =============================================================================
     * ================================ SCREENSHOT FUNCTIONS =======================
     * =============================================================================
     */

    /**
     * Checks if screenshots directory exists and creates it if not
     * @returns true if the directory exists or was created successfully, false otherwise
     */
    async _ensureScreenshotDir() {
        if (!this.screenshotsDir) return;
        if (!fs.existsSync(this.screenshotsDir)) {
            fs.mkdirSync(this.screenshotsDir, { recursive: true });
        }
    }

    /**
     * Checks if the browser and page are initialized, and initializes them if not
     */
    async _ensureBrowser() {
        if (!this.captureScreenshots || !this.viewerPort) return;
        if (this.browser && this.page) return;

        this.browser = await puppeteer.launch({ headless: 'new' });
        this.page = await this.browser.newPage();
        await this.page.setViewport({ width: this.screenshotWidth, height: this.screenshotHeight });
        await this.page.goto(`http://localhost:${this.viewerPort}`, { waitUntil: 'networkidle0' });
        await this.page.waitForSelector('canvas');
    }

    /**
     * Captures a screenshot of the viewer canvas and saves it to the screenshots directory
     */
    async _captureViewerScreenshot() {
        if (!this.captureScreenshots || !this.viewerPort) return null;
        try {
            await this._ensureScreenshotDir();
            await this._ensureBrowser();

            const canvas = await this.page.$('canvas');
            if (!canvas) return null;

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `worldmodel_${timestamp}.${this.screenshotFormat}`;
            const filePath = path.join(this.screenshotsDir, filename);

            await canvas.screenshot({ path: filePath, type: this.screenshotFormat });
            return filePath;
        } catch (error) {
            console.warn(`[${this.agentName}] Screenshot capture failed: ${error.message}`);
            return null;
        }
    }

    /**
     * Queues a screenshot capture to avoid overlapping captures
     * May have some delay if captures take longer than the tracking interval, but ensures we get a screenshot for each recorded action without conflicts
     */
    async _queueScreenshotCapture() {
        if (!this.captureScreenshots) return null;
        this.screenshotQueue = this.screenshotQueue.then(() => this._captureViewerScreenshot());
        return this.screenshotQueue;
    }


    /**
     * =============================================================================
     * ================================ GET'S AND EXPORTS ==========================
     * =============================================================================
     */

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

            // Add th commit version as an ID of the metric
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
     * Capture final version (git commit hash), then export metrics to JSON file
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