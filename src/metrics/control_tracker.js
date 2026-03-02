/**
 * ControlTracker - Low-level control input tracking
 * Monitors and records individual bot control inputs (WASD, jump, mine, place, etc.)
 * Complements the high-level ActionTracker with granular input data
 * Includes camera orientation (yaw/pitch) with each control event
 */

import { CameraTracker } from './camera_tracker.js';

/**
 * Round a number to a specified number of decimal places
 * @param {number} num - Number to round
 * @param {number} decimals - Number of decimal places
 * @returns {number} Rounded number
 */
function round(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

export class ControlTracker {
    constructor(bot, metricsCollector) {
        this.bot = bot;
        this.metrics = metricsCollector;
        this.cameraTracker = new CameraTracker(bot); // Initialize camera tracker for orientation data
        this.controlStates = {
            forward: false,
            back: false,
            left: false,
            right: false,
            jump: false,
            sprint: false,
            sneak: false,
            mine: false,
            place: false
        };
        this.monitoringInterval = null;
        this.isEnabled = false;
        this.lastState = { ...this.controlStates };
        this.controlSequence = []; // History of all control changes
        this.maxSequenceLength = 1000; // Limit history to prevent memory issues
        this.checkCount = 0; // Debug counter for polling
        this.originalSetControlState = null; // Store original function
        this.pendingScreenshots = []; // Track pending screenshot captures
        this.setupControlInterception();
    }

    /**
     * Intercept bot.setControlState() calls to track control changes
     * @private
     */
    setupControlInterception() {
        if (!this.bot || !this.bot.setControlState) return;

        // Store the original setControlState method
        this.originalSetControlState = this.bot.setControlState.bind(this.bot);

        // Replace with our wrapper
        this.bot.setControlState = (control, state) => {
            // Call the original method
            this.originalSetControlState(control, state);

            // Track the state change
            if (this.controlStates.hasOwnProperty(control)) {
                const oldState = this.controlStates[control];
                this.controlStates[control] = state;

                // Record change if different
                if (oldState !== state) {
                    this.recordControlChange(control, state);
                }
            }
        };
    }

    /**
     * Start monitoring bot control states
     * @param {number} pollInterval - How often to check state (ms, default 50ms)
     */
    start(pollInterval = 50) {
        if (this.isEnabled) return;
        this.isEnabled = true;
        const botName = this.bot.username || this.bot.name || 'unknown';

        console.log(`[ControlTracker] [${botName}] Control tracking enabled`);
        console.log(`[ControlTracker] [${botName}] Monitoring setControlState() calls`);
        console.log(`[ControlTracker] [${botName}] Bot properties: username=${this.bot.username}, name=${this.bot.name}, entity=${!!this.bot.entity}`);

        // Since we're intercepting setControlState, we don't need interval polling
        // But keep a minimal interval for cleanup and stats
        this.monitoringInterval = setInterval(() => {
            this.checkForMinePlace();
        }, pollInterval);
    }

    /**
     * Stop monitoring bot control states
     */
    stop() {
        if (!this.isEnabled) return;
        this.isEnabled = false;

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        // Optionally restore the original setControlState (useful if creating multiple trackers)
        if (this.originalSetControlState && this.bot) {
            this.bot.setControlState = this.originalSetControlState;
        }

        const botName = this.bot.username || this.bot.name || 'unknown';
        console.log(`[ControlTracker] [${botName}] Low-level control tracking stopped`);
    }

    /**
     * Get current state of a specific control
     * @param {string} control - Control name (forward, back, left, right, jump, sprint, sneak, mine, place)
     * @returns {boolean} Current state
     */
    getControlState(control) {
        return this.controlStates[control] || false;
    }

    /**
     * Get all current control states
     * @returns {Object} Current state of all controls
     */
    getAllControlStates() {
        return { ...this.controlStates };
    }

    /**
     * Check for mine/place events which aren't captured by setControlState
     * @private
     */
    checkForMinePlace() {
        if (!this.isEnabled || !this.bot) return;

        // Check for mine by monitoring if there's a target block
        if (this.bot.targetDigBlock && !this.controlStates.mine) {
            this.controlStates.mine = true;
            this.recordControlChange('mine', true);
        } else if (!this.bot.targetDigBlock && this.controlStates.mine) {
            this.controlStates.mine = false;
            this.recordControlChange('mine', false);
        }

        // Check for place (simplified - could be improved with proper event listening)
        // This is harder to detect without explicit events
    }

    /**
     * Record a control state change event
     * Includes camera orientation (yaw/pitch) and position
     * @private
     */
    recordControlChange(controlName, pressed) {
        // Get current camera orientation
        const orientation = this.cameraTracker.getCurrentOrientation();
        
        const event = {
            control: controlName,
            action: pressed ? 'pressed' : 'released',
            timestamp: new Date().toISOString(),
            position: this.bot?.entity?.position ? {
                x: round(this.bot.entity.position.x, 3),
                y: round(this.bot.entity.position.y, 3),
                z: round(this.bot.entity.position.z, 3)
            } : null,
            camera: orientation ? {
                yaw: orientation.yaw,
                pitch: orientation.pitch
            } : null,
            screenshot: null  // Will be populated if screenshot capture is enabled
        };

        this.controlSequence.push(event);

        // Capture screenshot if metrics collector is available and has screenshot capability
        this.captureScreenshotForEvent(event);

        // Limit history size to prevent memory issues
        if (this.controlSequence.length > this.maxSequenceLength) {
            this.controlSequence.shift();  // Remove oldest event
        }

        // Log high-priority controls or when starting
        if (['jump', 'mine', 'place', 'sprint'].includes(controlName)) {
            const botName = this.bot.username || this.bot.name || 'unknown';
            console.log(`[ControlTracker] [${botName}] ${controlName.toUpperCase()} ${pressed ? 'PRESSED' : 'RELEASED'}`);
            console.log(`  Camera: yaw=${event.camera?.yaw}, pitch=${event.camera?.pitch}`);
        }
    }

    /**
     * Capture screenshot for a control event (async - doesn't block event recording)
     * @private
     */
    captureScreenshotForEvent(event) {
        if (!this.metrics || !this.metrics._queueScreenshotCapture) return;
        if (!this.metrics.captureScreenshots) return;

        // Create a promise for this screenshot capture
        const screenshotPromise = this.metrics._queueScreenshotCapture()
            .then(screenshotPath => {
                if (screenshotPath) {
                    event.screenshot = screenshotPath;
                }
                // Remove from pending when done
                const index = this.pendingScreenshots.indexOf(screenshotPromise);
                if (index > -1) {
                    this.pendingScreenshots.splice(index, 1);
                }
            })
            .catch(err => {
                // Silent fail - don't interrupt control tracking if screenshot fails
                const index = this.pendingScreenshots.indexOf(screenshotPromise);
                if (index > -1) {
                    this.pendingScreenshots.splice(index, 1);
                }
            });

        // Track this pending capture
        this.pendingScreenshots.push(screenshotPromise);
    }

    /**
     * Wait for all pending screenshot captures to complete before export
     * @returns {Promise<void>}
     */
    async waitForPendingScreenshots() {
        if (this.pendingScreenshots.length === 0) return;
        try {
            await Promise.all(this.pendingScreenshots);
        } catch (err) {
            // Silent fail - screenshots are optional
        }
    }

    /**
     * Get control event sequence (history)
     * @param {number} limit - Maximum number of events to return (default: all)
     * @returns {Array} Recent control events
     */
    getControlSequence(limit = null) {
        if (limit && limit > 0) {
            return this.controlSequence.slice(-limit);
        }
        return [...this.controlSequence];
    }

    /**
     * Get control statistics for analysis
     * @returns {Object} Statistics about control usage
     */
    getControlStats() {
        const stats = {
            totalPresses: 0,
            controlCounts: {},
            mostUsedControl: null,
            leastUsedControl: null
        };

        // Initialize counts
        for (const control in this.controlStates) {
            stats.controlCounts[control] = 0;
        }

        // Only count presses (release is implicit)
        for (const event of this.controlSequence) {
            if (event.action === 'pressed' && stats.controlCounts[event.control] !== undefined) {
                stats.controlCounts[event.control]++;
                stats.totalPresses++;
            }
        }

        // Find most and least used controls
        let maxCount = 0;
        let minCount = Infinity;
        for (const control in stats.controlCounts) {
            const count = stats.controlCounts[control];
            if (count > maxCount) {
                maxCount = count;
                stats.mostUsedControl = control;
            }
            if (count < minCount && count >= 0) {
                minCount = count;
                stats.leastUsedControl = control;
            }
        }

        return stats;
    }

    /**
     * Reset control sequence history
     */
    resetSequence() {
        this.controlSequence = [];
    }

    /**
     * Export control sequence for analysis
     * @returns {Object} Structured control data for export
     */
    exportControlData() {
        return {
            control_sequence: [...this.controlSequence],
            statistics: this.getControlStats()
        };
    }

    /**
     * Get a summary of movement inputs (common pattern analysis)
     * @returns {Object} Aggregated movement patterns
     */
    getMovementPatterns() {
        const patterns = {
            total_move_time_ms: 0,
            forward_presses: 0,
            backward_presses: 0,
            left_presses: 0,
            right_presses: 0,
            jump_events: 0,
            sprint_events: 0,
            common_combinations: []
        };

        let lastMoveTime = null;
        let currentMovingControls = new Set();

        for (const event of this.controlSequence) {
            switch (event.control) {
                case 'forward':
                    if (event.action === 'pressed') patterns.forward_presses++;
                    break;
                case 'back':
                    if (event.action === 'pressed') patterns.backward_presses++;
                    break;
                case 'left':
                    if (event.action === 'pressed') patterns.left_presses++;
                    break;
                case 'right':
                    if (event.action === 'pressed') patterns.right_presses++;
                    break;
                case 'jump':
                    if (event.action === 'pressed') patterns.jump_events++;
                    break;
                case 'sprint':
                    if (event.action === 'pressed') patterns.sprint_events++;
                    break;
            }

            // Track movement control combinations
            if (['forward', 'back', 'left', 'right'].includes(event.control)) {
                if (event.action === 'pressed') {
                    currentMovingControls.add(event.control);
                } else {
                    currentMovingControls.delete(event.control);
                }

                if (currentMovingControls.size > 0) {
                    const combo = Array.from(currentMovingControls).sort().join('+');
                    const existing = patterns.common_combinations.find(c => c.combination === combo);
                    if (existing) {
                        existing.count++;
                    } else {
                        patterns.common_combinations.push({ combination: combo, count: 1 });
                    }
                }
            }
        }

        // Sort combinations by frequency
        patterns.common_combinations.sort((a, b) => b.count - a.count);

        return patterns;
    }
}
