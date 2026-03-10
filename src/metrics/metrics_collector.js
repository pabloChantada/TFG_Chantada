/**
 * Main metrics collector
 * Orchestrates tracking, screenshots, and export
 */

import { ControlTracker } from './trackers/control_tracker.js'
import { ScreenshotManager } from './collector/screenshot.js'
import { VersionManager } from './collector/version_manager.js'
import { MetricsExporter } from './collector/export_metrics.js'

const POLL_INTERVAL = 150 // ms

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
     * Pre-initialize the screenshot browser so first capture has no cold-start delay.
     * Must be called AFTER enable() and BEFORE startControlTracking().
     */
    async warmupScreenshots() {
        if (this.screenshotManager.isEnabled()) {
            return await this.screenshotManager.warmup()
        }
        return false
    }

    /**
     * Start low-level control tracking
     */
    startControlTracking(bot, pollInterval = POLL_INTERVAL, captureScreenshots = false, viewerPort = null) {
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
     * Check whether this run was configured to capture screenshots
     */
    hadScreenshotsEnabled() {
        return this.screenshotManager.wasEnabled()
    }
    /**
     * Stop screenshot capture (when bot disconnects)
     */
    stopScreenshots() {
        this.screenshotManager.stop();
    }
    /**
     * Get control states (for external queries)
     */
    getControlStates() {
        return this.controlTracker?.getAllControlStates() || null
    }

    /**
     * Get RL action sequence
     */
    getRLActionSequence(limit = null) {
        return this.controlTracker?.getRLActionSequence(limit) || null
    }

    /**
     * Get current RL action state
     */
    getCurrentRLAction() {
        return this.controlTracker?.getCurrentRLAction() || null
    }

    /**
     * Get RL action statistics
     */
    getRLActionStats() {
        return this.controlTracker?.getRLActionStats() || null
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
     * Record an error (capped at 100 to prevent unbounded growth)
     */
    recordError(errorMessage) {
        if (this.metrics.errors.length >= 100) {
            // Keep first 50 and last 49, add overflow marker
            if (!this.metrics.errors._truncated) {
                const first = this.metrics.errors.slice(0, 50)
                const last = this.metrics.errors.slice(-49)
                this.metrics.errors = [...first, { message: `... ${this.metrics.errors.length - 99} errors truncated ...`, timestamp: new Date().toISOString() }, ...last]
                this.metrics.errors._truncated = true
            } else {
                this.metrics.errors.pop() // remove last before adding new
            }
        }
        this.metrics.errors.push({
            message: errorMessage,
            timestamp: new Date().toISOString()
        })
    }

    // =========================================================================
    // --- EXPORT
    // =========================================================================

    /**
     * Free tracker memory after export to prevent accumulation
     */
    _freeTrackerMemory() {
        if (this.controlTracker) {
            this.controlTracker.resetSequence()
        }
        // Clear control_tracking from metrics (already exported)
        delete this.metrics.control_tracking
        console.log(`[${this.agentName}] Tracker memory freed`)
    }

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
            console.log(`[${this.agentName}] ========== STARTING METRICS EXPORT ==========`)

            // Freeze control/action recording before finalizing dataset export
            this.stopControlTracking()
            
            // Capture version and final state (safe even if bot is partially disconnected)
            if (bot) {
                try {
                    console.log(`[${this.agentName}] Capturing final bot state...`)
                    const finalState = await this.getFinalState(bot)
                    Object.assign(this.metrics, finalState)
                    console.log(`[${this.agentName}] Final state captured`)
                } catch (err) {
                    console.warn(`[${this.agentName}] Final state capture failed: ${err.message}`)
                }
            }

            // Screenshots are captured synchronously with each action record,
            // so there's nothing to flush. Just log stats.
            if (this.screenshotManager.wasEnabled()) {
                const tracker = this.controlTracker?.rlActionTracker
                if (tracker) {
                    console.log(`[${this.agentName}] Screenshot stats: ${tracker.capturedCount} captured, ${tracker.skippedCount} skipped`)
                }
            }

            // Collect control tracking data (includes rl_actions with any screenshots)
            if (this.controlTracker) {
                console.log(`[${this.agentName}] Exporting control tracking data...`)
                this.metrics.control_tracking = this.controlTracker.exportControlData()
                console.log(`[${this.agentName}] Control tracking data ready`)
            } else {
                console.log(`[${this.agentName}] No control tracker active`)
            }

            // === CRITICAL: Export JSON (this MUST succeed) ===
            console.log(`[${this.agentName}] Calling exporter.exportJSON()...`)
            const exportResult = await this.exporter.exportJSON(this.metrics)
            console.log(`[${this.agentName}] Export result:`, exportResult)

            // Cleanup browser (best-effort)
            try {
                await this.screenshotManager.cleanup()
            } catch (err) {
                console.warn(`[${this.agentName}] Screenshot cleanup failed: ${err.message}`)
            }

            // Free tracker memory after successful export
            this._freeTrackerMemory()
            
            console.log(`[${this.agentName}] ========== METRICS EXPORT COMPLETE ==========`)

        } catch (error) {
            console.error(`[${this.agentName}] ========== METRICS EXPORT FAILED ==========`)
            console.error(`[${this.agentName}] Error:`, error.message)
            console.error(`[${this.agentName}] Stack:`, error.stack)
            
            // Last-resort: try to export whatever metrics we have, without screenshots
            try {
                console.log(`[${this.agentName}] Attempting emergency JSON export...`)
                if (this.controlTracker && !this.metrics.control_tracking) {
                    this.metrics.control_tracking = this.controlTracker.exportControlData()
                }
                await this.exporter.exportJSON(this.metrics)
                console.log(`[${this.agentName}] ✓ Emergency export succeeded`)
            } catch (emergencyErr) {
                console.error(`[${this.agentName}] Emergency export also failed: ${emergencyErr.message}`)
            }
        }
    }
}