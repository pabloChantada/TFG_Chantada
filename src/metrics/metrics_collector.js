import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer';
import { ActionTracker } from './action_tracker.js';
import { ControlTracker } from './control_tracker.js';

export class MetricsCollector {
    constructor(agentName, agentType) {
        this.agentName = agentName;
        this.agentType = agentType;
        this.exportPath = null;
        this.lastPosition = null;
        this.trackingInterval = null; // How many intervals to track in a second
        this.currentAction = null; // Track ongoing action
        this.actionTracker = null; // Automatic action tracker
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
        this.actionQueue = Promise.resolve(null);
        this.controlTracker = null; // Low-level control input tracker (includes camera data)
        
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

        const currentActionData = this.currentAction;
        this.currentAction = null;

        this.actionQueue = this.actionQueue.then(async () => {
            // Actions are now tracked through control_sequence with full context
        });

        return this.actionQueue;
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
     * @param {boolean} captureScreenshots - Enable viewer screenshots for control state tracking
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
     * Start automatic action tracking based on bot state
     * @param {Object} bot - Mineflayer bot instance
     * @param {number} pollInterval - How often to check state (ms, default 100ms)
     */
    startActionTracking(bot, pollInterval = 50) {
        if (!bot) {
            console.warn('[MetricsCollector] Cannot start action tracking: bot not provided');
            return;
        }

        this.actionTracker = new ActionTracker(bot, this);
        this.actionTracker.start(pollInterval);
        console.log('[MetricsCollector] Automatic action tracking started');
    }

    /**
     * Stop automatic action tracking
     */
    stopActionTracking() {
        if (this.actionTracker) {
            this.actionTracker.stop();
            this.actionTracker = null;
        }
    }

    /**
     * Start tracking low-level bot control inputs (WASD, jump, mine, etc.)
     * @param {Object} bot - Mineflayer bot instance
     * @param {number} pollInterval - How often to check state (ms, default 50ms)
     * @param {boolean} captureScreenshots - Enable screenshot capture (requires viewerPort)
     * @param {number} viewerPort - Port for prismarine viewer screenshots
     */
    startControlTracking(bot, pollInterval = 50, captureScreenshots = false, viewerPort = null) {
        console.log(`[TRACE] startControlTracking called with bot=${bot ? 'valid' : 'null'}, interval=${pollInterval}, screenshots=${captureScreenshots}, viewer=${viewerPort}`);
        
        if (!bot) {
            console.warn('[MetricsCollector] Cannot start control tracking: bot not provided');
            return;
        }

        // Enable screenshots first if requested
        if (captureScreenshots && viewerPort) {
            console.log(`[INFO] [${this.agentName}] Enabling screenshots to viewer port ${viewerPort}`);
            this.enableScreenshots(viewerPort);
        }

        console.log(`[INFO] [${this.agentName}] Initializing ControlTracker...`);
        try {
            // Pass 'this' (MetricsCollector) so ControlTracker can access screenshot methods
            this.controlTracker = new ControlTracker(bot, this);
            console.log(`[INFO] [${this.agentName}] ControlTracker instance created successfully`);
            
            this.controlTracker.start(pollInterval);
            console.log(`[INFO] [${this.agentName}] ControlTracker.start() completed`);
            
            console.log(`[INFO] [${this.agentName}] Low-level control tracking started successfully`);
        } catch (err) {
            console.error(`[ERROR] [${this.agentName}] Failed to initialize ControlTracker:`, err);
            console.error(`[ERROR] Stack trace:`, err.stack);
            this.controlTracker = null;
        }
    }


    /**
     * Stop tracking low-level bot control inputs
     * Note: Data remains accessible after stopping for analysis/export
     */
    stopControlTracking() {
        if (this.controlTracker) {
            this.controlTracker.stop();
            // Don't set to null - keep data available for export/analysis
        }
    }

    /**
     * Get current control input states
     * @returns {Object|null} Current state of all controls or null if not tracking
     */
    getControlStates() {
        if (!this.controlTracker) return null;
        return this.controlTracker.getAllControlStates();
    }

    /**
     * Get control input history/sequence
     * @param {number} limit - Maximum number of events to return
     * @returns {Array|null} Recent control events or null if not tracking
     */
    getControlSequence(limit = null) {
        if (!this.controlTracker) return null;
        return this.controlTracker.getControlSequence(limit);
    }

    /**
     * Get control input statistics
     * @returns {Object|null} Control usage statistics or null if not tracking
     */
    getControlStats() {
        if (!this.controlTracker) return null;
        return this.controlTracker.getControlStats();
    }

    /**
     * Enable screenshot capture for control tracking
     * @param {number} viewerPort - Port for prismarine viewer
     */
    enableScreenshots(viewerPort) {
        this.captureScreenshots = true;
        this.viewerPort = viewerPort;
        this.screenshotsDir = `src/metrics/agent_metrics/${this.agentName}_screenshots/`;
        console.log(`[${this.agentName}] Screenshots enabled for control tracking`);
    }

    /**
     * Start tracking bot observations and movements (deprecated - use startControlTracking instead)
     * Kept for backwards compatibility
     * @param {Object} bot - Mineflayer bot instance
     * @param {number} samplingRate - Samples per second (default: 1)
     */
    startWorldTracking(bot, samplingRate = 1) {
        if (!bot) return;

        // Start automatic action tracking alongside control tracking
        this.startActionTracking(bot, 100);
        console.log(`[${this.agentName}] World tracking started (use startControlTracking for low-level inputs)`);
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
            if (bot && bot.entity && bot.inventory) {
                await this.getVersion(bot);
            }

            // Wait for pending screenshots before exporting control tracking data
            if (this.controlTracker) {
                console.log(`[${this.agentName}] Waiting for pending screenshot captures...`);
                await this.controlTracker.waitForPendingScreenshots();
                console.log(`[${this.agentName}] All screenshots captured`);
                this.metrics.control_tracking = this.controlTracker.exportControlData();
            }

            // Export JSON metrics
            await this.exportJSON();

        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics:`, error.message);
        }
    }

    /**
     * Capture final version (git commit hash), then export metrics to JSON file
     * @param {Object} bot - Mineflayer bot instance to capture final state and version information
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

        if (bot?.entity?.position) {
            this.metrics.final_position = bot.entity.position;
        }
        if (bot?.inventory?.items) {
            this.metrics.final_inventory = bot.inventory.items().map(item => ({
                name: item.name,
                count: item.count
            }));
        }
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