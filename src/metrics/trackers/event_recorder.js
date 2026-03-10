/**
 * Simple event recorder for raw control state changes.
 * Used by ControlAnalytics for statistics (press counts, movement patterns).
 */

import { getBotPosition } from '../utils.js'

export class EventRecorder {
    constructor(bot, cameraTracker) {
        this.bot = bot
        this.cameraTracker = cameraTracker
        this.controlSequence = []
        this.maxSequenceLength = 5000
    }

    /**
     * Record a control state change event
     */
    recordControlChange(controlName, pressed, detail = null) {
        const orientation = this.cameraTracker.getCurrentOrientation()

        const event = {
            control: controlName,
            action: pressed ? 'pressed' : 'released',
            timestamp: new Date().toISOString(),
            position: getBotPosition(this.bot),
            camera: orientation ? {
                yaw: orientation.yaw,
                pitch: orientation.pitch
            } : null,
        }

        if (detail) {
            event.detail = detail
        }

        this.controlSequence.push(event)

        if (this.controlSequence.length > this.maxSequenceLength) {
            this.controlSequence.shift()
        }

        return event
    }

    /**
     * Get control event sequence
     */
    getControlSequence(limit = null) {
        if (limit && limit > 0) {
            return this.controlSequence.slice(-limit)
        }
        return [...this.controlSequence]
    }

    /**
     * Reset sequence history
     */
    resetSequence() {
        this.controlSequence = []
    }
}
