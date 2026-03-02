/**
 * Handles recording of control events with orientation and screenshots
 */

import { getBotPosition } from '../utils.js'

export class EventRecorder {
    constructor(bot, cameraTracker, metrics) {
        this.bot = bot
        this.cameraTracker = cameraTracker
        this.metrics = metrics
        this.controlSequence = []
        this.maxSequenceLength = 1000
        this.pendingScreenshots = []
    }

    /**
     * Record a control state change event
     */
    recordControlChange(controlName, pressed) {
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
            screenshot: null
        }

        // Add to sequence and capture screenshot
        this.controlSequence.push(event)
        this.captureScreenshotForEvent(event)

        if (this.controlSequence.length > this.maxSequenceLength) {
            this.controlSequence.shift()  // Remove oldest event to maintain max length
        }

        return event
    }

    /**
     * Capture screenshot for a control event (async - doesn't block)
     */
    captureScreenshotForEvent(event) {
        if (!this.metrics?.captureScreenshots || !this.metrics?._queueScreenshotCapture) return

        // Use a queue to manage pending screenshots and avoid overwhelming the system
        const screenshotPromise = this.metrics._queueScreenshotCapture()
            .then(screenshotPath => {
                if (screenshotPath) {
                    event.screenshot = screenshotPath
                }
                this.removePendingScreenshot(screenshotPromise)
            })
            .catch(() => {
                this.removePendingScreenshot(screenshotPromise)
            })

        this.pendingScreenshots.push(screenshotPromise)
    }

    /**
     * Remove screenshot from pending list
     */
    removePendingScreenshot(promise) {
        const index = this.pendingScreenshots.indexOf(promise)
        if (index > -1) {
            this.pendingScreenshots.splice(index, 1)
        }
    }

    /**
     * Wait for all pending screenshots to complete
     */
    async waitForPendingScreenshots() {
        if (this.pendingScreenshots.length === 0) return
        try {
            await Promise.all(this.pendingScreenshots)
        } catch {
            // Silent fail - screenshots are optional
        }
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