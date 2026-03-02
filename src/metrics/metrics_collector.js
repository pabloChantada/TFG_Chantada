/**
 * Main metrics collector
 * Orchestrates tracking, screenshots, and export
 */

import { ControlTracker } from './trackers/control_tracker.js'
import { ScreenshotManager } from './collector/screenshot.js'
import { VersionManager } from './collector/version_manager.js'
import { MetricsExporter } from './collector/export_metrics.js'

export class MetricsCollector {
    constructor(agentName, agentType) {
        this.agentName = agentName
        this.agentType = agentType
        
        // Sub-modules
        this.controlTracker = null
        this.screenshotManager = new ScreenshotManager(agentName)
        this.exporter = new MetricsExporter(agentName)
        
        // Core metrics
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
        }
    }

    // =========================================================================
    // --- INITIALIZATION
    // =========================================================================

    /**
     * Initialize metrics collection
     */
    async initialize(exportPath, task, captureScreenshots = false, viewerPort = null) {
        this.exporter.setExportPath(exportPath)
        this.metrics.task = task || 'Iron task'
        this.metrics.start_time = new Date().toISOString()

        if (captureScreenshots && viewerPort) {
            this.screenshotManager.enable(viewerPort)
        }

        console.log(`[${this.agentName}] Metrics initialized - Export: ${this.exporter.getExportPath()}`)
    }

    // =========================================================================
    // --- CONTROL TRACKING
    // =========================================================================

    /**
     * Start low-level control tracking
     */
    startControlTracking(bot, pollInterval = 50, captureScreenshots = false, viewerPort = null) {
        if (!bot) {
            console.warn('[MetricsCollector] Cannot start control tracking: bot not provided')
            return
        }

        // Enable screenshots if requested
        if (captureScreenshots && viewerPort) {
            this.screenshotManager.enable(viewerPort)
        }

        // Initialize control tracker
        try {
            this.controlTracker = new ControlTracker(bot, this)
            this.controlTracker.start(pollInterval)
            console.log(`[${this.agentName}] Control tracking started`)
        } catch (error) {
            console.error(`[${this.agentName}] Failed to start control tracking:`, error)
            this.controlTracker = null
        }
    }

    /**
     * Stop control tracking
     */
    stopControlTracking() {
        if (this.controlTracker) {
            this.controlTracker.stop()
        }
    }

    /**
     * Get control states (for external queries)
     */
    getControlStates() {
        return this.controlTracker?.getAllControlStates() || null
    }

    /**
     * Get control sequence
     */
    getControlSequence(limit = null) {
        return this.controlTracker?.getControlSequence(limit) || null
    }

    /**
     * Get control statistics
     */
    getControlStats() {
        return this.controlTracker?.getControlStats() || null
    }

        /**
     * Capture bot version and final state
     */
    async getFinalState(bot) {
        const state = {
            version: await VersionManager.getGitVersion()
        }

        if (bot?.entity?.position) {
            state.final_position = bot.entity.position
        }

        if (bot?.inventory?.items) {
            state.final_inventory = bot.inventory.items().map(item => ({
                name: item.name,
                count: item.count
            }))
        }

        return state
    }

    // =========================================================================
    // --- SCREENSHOTS (exposed to ControlTracker)
    // =========================================================================

    /**
     * Queue a screenshot capture (called by ControlTracker)
     */
    async _queueScreenshotCapture() {
        return this.screenshotManager.queueCapture()
    }

    /**
     * Check if screenshots are enabled
     */
    captureScreenshots() {
        return this.screenshotManager.isEnabled()
    }

    /**
     * Enable screenshots
     */
    enableScreenshots(viewerPort) {
        this.screenshotManager.enable(viewerPort)
    }

    // =========================================================================
    // --- TASK COMPLETION & ERRORS
    // =========================================================================

    /**
     * Mark task completion
     */
    completeTask(success) {
        this.metrics.success = success
        this.metrics.end_time = new Date().toISOString()
        this.metrics.time_elapsed_s = 
            (new Date(this.metrics.end_time) - new Date(this.metrics.start_time)) / 1000
    }

    /**
     * Record an error
     */
    recordError(errorMessage) {
        this.metrics.errors.push({
            message: errorMessage,
            timestamp: new Date().toISOString()
        })
    }

    // =========================================================================
    // --- EXPORT
    // =========================================================================

    /**
     * Get current metrics snapshot
     */
    getMetrics() {
        return { ...this.metrics }
    }

    /**
     * Export metrics to JSON
     */
    async export(bot = null) {
        try {
            // Capture version and final state
            if (bot) {
                const finalState = await this.getFinalState(bot)
                Object.assign(this.metrics, finalState)
            }

            // Wait for pending screenshots
            if (this.controlTracker) {
                console.log(`[${this.agentName}] Waiting for pending screenshots...`)
                await this.controlTracker.waitForPendingScreenshots()
                this.metrics.control_tracking = this.controlTracker.exportControlData()
            }

            // Export JSON
            await this.exporter.exportJSON(this.metrics)

            // Cleanup browser
            await this.screenshotManager.cleanup()

        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics:`, error.message)
        }
    }
}