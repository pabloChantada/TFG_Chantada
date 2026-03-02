/**
 * ControlTracker - Low-level control input tracking
 * Modular architecture: interception, recording, and analytics
 */

import { CameraTracker } from './camera_tracker.js'
import { EventRecorder } from './event.js'
import { ControlAnalytics } from './analytics.js'
import { ControlInterceptor } from './interceptor.js'
import { getBotName } from '../utils.js'

export class ControlTracker {
    constructor(bot, metricsCollector) {
        this.bot = bot
        this.metrics = metricsCollector
        this.cameraTracker = new CameraTracker(bot)

        this.controlStates = {
            forward: false,
            back: false,
            left: false,
            right: false,
            jump: false,
            sprint: false,
            sneak: false,  // Used when near an edge
            mine: false,
            place: false,
            openWindow: false,  // Using a furnace, chest, crafting table, etc. 
        }

        // Initialize modules
        // EventRecorder will handle recording control changes and capturing screenshots
        this.eventRecorder = new EventRecorder(bot, this.cameraTracker, metricsCollector)
        // ControlAnalytics will analyze the recorded control events for patterns and stats  
        this.analytics = new ControlAnalytics(this.eventRecorder)
        // ControlInterceptor will hook into bot control state changes to trigger recording
        this.interceptor = new ControlInterceptor(
            bot,
            this.controlStates,
            (control, state) => this.eventRecorder.recordControlChange(control, state)
        )
    }

    /**
     * Start monitoring bot control states
     */
    start(pollInterval = 50) {
        const botName = getBotName(this.bot)
        console.log(`[ControlTracker] [${botName}] Control tracking enabled`)
        console.log(`[ControlTracker] [${botName}] Monitoring setControlState() calls`)

        // Start to track control state changes 
        this.interceptor.start(pollInterval)
    }

    /**
     * Stop monitoring bot control states
     */
    stop() {
        this.interceptor.stop()

        const botName = getBotName(this.bot)
        console.log(`[ControlTracker] [${botName}] Low-level control tracking stopped`)
    }

    /**
     * Get current state of a specific control
     */
    getControlState(control) {
        return this.controlStates[control] ?? false
    }

    /**
     * Get all current control states
     */
    getAllControlStates() {
        return { ...this.controlStates }
    }

    /**
     * Get control event sequence
     */
    getControlSequence(limit = null) {
        return this.eventRecorder.getControlSequence(limit)
    }

    /**
     * Get control statistics
     */
    getControlStats() {
        return this.analytics.getControlStats()
    }

    /**
     * Get movement patterns
     */
    getMovementPatterns() {
        return this.analytics.getMovementPatterns()
    }

    /**
     * Reset sequence history
     */
    resetSequence() {
        this.eventRecorder.resetSequence()
    }

    /**
     * Export all control data
     */
    exportControlData() {
        return this.analytics.exportControlData()
    }

    /**
     * Wait for pending screenshots
     */
    async waitForPendingScreenshots() {
        return this.eventRecorder.waitForPendingScreenshots()
    }
}