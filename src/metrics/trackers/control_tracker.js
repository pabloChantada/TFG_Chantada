/**
 * ControlTracker - Low-level control input tracking
 * Modular architecture: interception, recording (RL actions), and analytics
 */

import { CameraTracker } from './camera_tracker.js'
import { ControlAnalytics } from './analytics.js'
import { ControlInterceptor } from './interceptor.js'
import { RLActionTracker } from './rl_action_tracker.js'
import { EventRecorder } from './event_recorder.js'
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
            sneak: false,
            mine: false,
            place: false,
            openWindow: false,
        }

        // RL action tracker (MultiDiscrete action space recording)
        this.rlActionTracker = new RLActionTracker(bot, this.cameraTracker, metricsCollector)

        // Event recorder for raw control sequence (analytics)
        this.eventRecorder = new EventRecorder(bot, this.cameraTracker)
        this.analytics = new ControlAnalytics(this.eventRecorder)

        // Interceptor hooks into bot methods to detect control changes
        this.interceptor = new ControlInterceptor(
            bot,
            this.controlStates,
            (control, state, detail) => this.eventRecorder.recordControlChange(control, state, detail),
            (actionType, value) => this._handleRLAction(actionType, value),
            metricsCollector
        )

        this.pollInterval = null
    }

    /**
     * Handle RL action callbacks from interceptor
     */
    _handleRLAction(actionType, value) {
        switch (actionType) {
            case 'attack':
                this.rlActionTracker.updateAttackAction(value)
                break
            case 'craft':
                this.rlActionTracker.updateCraftAction(value)
                break
            case 'smelt':
                this.rlActionTracker.updateSmeltAction(value)
                break
            case 'place':
                this.rlActionTracker.updatePlaceAction(value)
                break
            case 'equip':
                this.rlActionTracker.updateEquipAction(value)
                break
        }
    }

    /**
     * Start monitoring bot control states.
     * Uses an async loop so each tick awaits the screenshot capture.
     */
    start(pollInterval = 150) {
        const botName = getBotName(this.bot)
        const hasScreenshots = this.metrics?.captureScreenshots?.() || false

        console.log(`[ControlTracker] [${botName}] RL action tracking enabled${hasScreenshots ? ' (with screenshots)' : ''}`)

        // Start interceptor (hooks into bot.setControlState, placeBlock, etc.)
        this.interceptor.start(pollInterval)

        // Async recording loop: awaits each tick so screenshot is synchronous with action
        this._running = true
        this._runLoop(pollInterval)
    }

    /**
     * Async recording loop.
     * Each iteration: update state → capture screenshot → record action → wait interval.
     */
    async _runLoop(interval) {
        while (this._running) {
            try {
                this.rlActionTracker.updateMovementAction(this.controlStates)
                this.rlActionTracker.updateCameraAction()
                await this.rlActionTracker.recordAction()
            } catch (err) {
                // Don't crash the loop on transient errors
            }
            await new Promise(r => setTimeout(r, interval))
        }
    }

    /**
     * Stop monitoring bot control states
     */
    stop() {
        this._running = false
        this.interceptor.stop()

        const botName = getBotName(this.bot)
        console.log(`[ControlTracker] [${botName}] RL action tracking stopped`)
    }

    /**
     * Full cleanup - release all references to allow garbage collection
     */
    dispose() {
        this.stop()
        this.resetSequence()
        this.bot = null
        this.metrics = null
    }

    /**
     * Track a named high-level action (called by HTN primitives)
     */
    trackAction(actionName) {
        // Could be extended to map HTN action names to RL action updates
    }

    // --- State queries ---

    getControlState(control) {
        return this.controlStates[control] ?? false
    }

    getAllControlStates() {
        return { ...this.controlStates }
    }

    // --- RL action queries ---

    getRLActionSequence(limit = null) {
        return this.rlActionTracker.getActionSequence(limit)
    }

    getCurrentRLAction() {
        return this.rlActionTracker.getCurrentAction()
    }

    getRLActionStats() {
        return this.rlActionTracker.getActionStats()
    }

    // --- Export ---

    /**
     * Export all tracking data for metrics JSON.
     */
    exportControlData() {
        const rlData = this.rlActionTracker.exportForDataset(false)
        const analyticsData = this.analytics.exportControlData()

        return {
            ...analyticsData,
            rl_actions: rlData
        }
    }

    /**
     * Reset sequence history
     */
    resetSequence() {
        this.rlActionTracker.resetSequence()
        this.eventRecorder.resetSequence()
    }
}
