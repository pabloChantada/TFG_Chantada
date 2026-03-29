/**
 * RL Action Space Tracker
 * Tracks and records DIRECT discrete actions (single label per timestep)
 * for the tree-cutting imitation learning task.
 */

import { getBotPosition } from '../utils.js'
import { extractState } from '../../rl/state.js'

export class RLActionTracker {
    constructor(bot, cameraTracker, metricsCollector = null) {
        this.bot = bot
        this.cameraTracker = cameraTracker
        this.metrics = metricsCollector
        this.actionSequence = []
        this.maxSequenceLength = 5000
        
        // Current low-level control state used to derive one discrete action label
        this.currentAction = this.getEmptyAction()
        
        // Previous camera orientation for delta calculation
        this.lastYaw = null
        this.lastPitch = null
        
        // Track last actions to avoid duplicates
        this.lastRecordedAction = null
        this.recordThrottle = 50 // ms between identical action recordings

        // Counters
        this.capturedCount = 0
        this.skippedCount = 0

        // Priority to collapse simultaneous controls into ONE discrete action
        // Highest priority first (tree-cutting policy)
        this.discretePriority = [
            'attack',
            'equip_wooden_axe',
            'camera_yaw',
            'camera_pitch',
            'move_forward',
            'move_backward',
            'move_lateral',
            'move_vertical'
        ]

        this.actionToId = {
            idle: 0,
            move_forward_walk: 1,
            move_forward_sprint: 2,
            move_backward_walk: 3,
            move_left: 4,
            move_right: 5,
            jump: 6,
            sneak: 7,
            camera_yaw_p15: 8,
            camera_yaw_m15: 9,
            camera_yaw_p45: 10,
            camera_yaw_m45: 11,
            camera_pitch_p15: 12,
            camera_pitch_m15: 13,
            camera_pitch_p45: 14,
            camera_pitch_m45: 15,
            attack: 16,
            equip_wooden_axe: 17
        }
    }

    /**
     * Get empty/neutral action state
     */
    getEmptyAction() {
        return {
            move_forward: 0,    // 0=still, 1=walk, 2=sprint
            move_backward: 0,   // 0=still, 1=walk, it cant run backwards
            move_lateral: 0,    // 0=still, 1=left, 2=right
            move_vertical: 0,   // 0=still, 1=jump, 2=sneak
            camera_yaw: 0,      // 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
            camera_pitch: 0,    // 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
            attack: 0,          // 0=no, 1=attack
            equip_wooden_axe: 0 // 0=no, 1=equip wooden axe
        }
    }

    /**
     * Update movement action based on control states
     */
    updateMovementAction(controlStates) {
        // Reset both axes first to avoid stale values between ticks
        this.currentAction.move_forward = 0
        this.currentAction.move_backward = 0

        // move_forward: 0=still, 1=walk, 2=sprint
        // move_backward: 0=still, 1=walk (cannot sprint backwards)
        if (controlStates.forward) {
            this.currentAction.move_forward = controlStates.sprint ? 2 : 1
        } else if (controlStates.back) {
            this.currentAction.move_backward = 1
        }

        // move_lateral: 0=still, 1=left, 2=right
        if (controlStates.left) {
            this.currentAction.move_lateral = 1
        } else if (controlStates.right) {
            this.currentAction.move_lateral = 2
        } else {
            this.currentAction.move_lateral = 0
        }

        // move_vertical: 0=still, 1=jump, 2=sneak
        if (controlStates.jump) {
            this.currentAction.move_vertical = 1
        } else if (controlStates.sneak) {
            this.currentAction.move_vertical = 2
        } else {
            this.currentAction.move_vertical = 0
        }
    }

    /**
     * Update camera action based on orientation delta
     */
    updateCameraAction() {
        const orientation = this.cameraTracker.getCurrentOrientation()
        
        // Default to neutral camera action each tick to avoid stale values
        this.currentAction.camera_yaw = 0
        this.currentAction.camera_pitch = 0

        if (!orientation) return

        const currentYaw = Number(orientation.yaw)
        const currentPitch = Number(orientation.pitch)

        if (!Number.isFinite(currentYaw) || !Number.isFinite(currentPitch)) return

        // Initialize if first time
        if (this.lastYaw === null) {
            this.lastYaw = currentYaw
            this.lastPitch = currentPitch
            return
        }

        // Calculate deltas (in radians)
        let yawDelta = currentYaw - this.lastYaw
        let pitchDelta = currentPitch - this.lastPitch

        // Convert to degrees before discretization thresholds
        yawDelta = yawDelta * (180 / Math.PI)
        pitchDelta = pitchDelta * (180 / Math.PI)

        // Normalize yaw delta to -180 to 180 range
        while (yawDelta > 180) yawDelta -= 360
        while (yawDelta < -180) yawDelta += 360

        // Map yaw delta to discrete action: 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
        if (Math.abs(yawDelta) < 7.5) {
            this.currentAction.camera_yaw = 0
        } else if (yawDelta >= 37.5) {
            this.currentAction.camera_yaw = 3  // +45°
        } else if (yawDelta >= 7.5) {
            this.currentAction.camera_yaw = 1  // +15°
        } else if (yawDelta <= -37.5) {
            this.currentAction.camera_yaw = 4  // -45°
        } else if (yawDelta <= -7.5) {
            this.currentAction.camera_yaw = 2  // -15°
        }

        // Map pitch delta to discrete action: 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
        if (Math.abs(pitchDelta) < 7.5) {
            this.currentAction.camera_pitch = 0
        } else if (pitchDelta >= 37.5) {
            this.currentAction.camera_pitch = 3  // +45°
        } else if (pitchDelta >= 7.5) {
            this.currentAction.camera_pitch = 1  // +15°
        } else if (pitchDelta <= -37.5) {
            this.currentAction.camera_pitch = 4  // -45°
        } else if (pitchDelta <= -7.5) {
            this.currentAction.camera_pitch = 2  // -15°
        }

        // Update last orientation
        this.lastYaw = currentYaw
        this.lastPitch = currentPitch
    }

    /**
     * Update attack action
     */
    updateAttackAction(isAttacking) {
        this.currentAction.attack = isAttacking ? 1 : 0
    }

    /**
     * Update equip action
     * @param {string} itemName - expects wooden_axe for the tree-cutting task
     */
    updateEquipAction(itemName) {
        this.currentAction.equip_wooden_axe = itemName === 'wooden_axe' ? 1 : 0
    }

    /**
     * Record current action state to sequence.
     * Captures screenshot SYNCHRONOUSLY — no record is created without a screenshot.
     * This guarantees every action in the dataset has a corresponding image.
     * @returns {Object|null} the action record, or null if skipped
     */
    async recordAction() {
        const now = Date.now()
        const discreteAction = this._toDiscreteAction(this.currentAction)
        
        // Avoid recording identical actions too frequently
        // Use direct value comparison instead of JSON.stringify to reduce GC pressure
        if (this.lastRecordedAction && 
            this._actionsEqual(discreteAction, this.lastRecordedAction.action) &&
            (now - this.lastRecordedAction._ts) < this.recordThrottle) {
            return null
        }

        // SCREENSHOT FIRST — if screenshots are enabled, capture one now.
        // If capture fails, skip this tick entirely (no screenshot = no record).
        let screenshotPath = null
        if (this.metrics?.screenshotManager?.isEnabled()) {
            screenshotPath = await this.metrics.screenshotManager.captureNow()
            if (!screenshotPath) {
                this.skippedCount++
                return null
            }
        }

        const orientation = this.cameraTracker.getCurrentOrientation()

        // Extract the full 13-dim state vector for richer offline training data
        let stateVector = null
        try {
            const stateResult = extractState(this.bot)
            stateVector = stateResult.vector
        } catch (_) {
            // Fallback: state extraction may fail if bot is partially disconnected
        }

        const actionRecord = {
            timestamp: new Date().toISOString(),
            position: getBotPosition(this.bot),
            camera: orientation ? {
                yaw: orientation.yaw,
                pitch: orientation.pitch
            } : null,
            state_vector: stateVector,
            action: discreteAction.name,
            action_id: discreteAction.id,
            screenshot: screenshotPath,
        }

        this.actionSequence.push(actionRecord)
        // Store numeric timestamp for efficient throttle comparison
        actionRecord._ts = now
        this.lastRecordedAction = actionRecord
        this.capturedCount++

        // Log progress every 50 captures
        if (this.capturedCount % 50 === 0) {
            console.log(`[RLActionTracker] ${this.capturedCount} actions recorded (${this.skippedCount} skipped)`)
        }

        // Maintain max sequence length
        if (this.actionSequence.length > this.maxSequenceLength) {
            this.actionSequence.shift()
        }

        return actionRecord
    }

    /**
     * Compare two action objects without JSON.stringify (reduces GC pressure)
     */
    _actionsEqual(a, b) {
        if (!a || !b) return false
        return a.id === b.id
    }

    /**
     * Collapse low-level state to one discrete action label.
     */
    _toDiscreteAction(action) {
        if (!action) {
            return { id: this.actionToId.idle, name: 'idle' }
        }

        for (const key of this.discretePriority) {
            if (key === 'attack' && action.attack === 1) {
                return { id: this.actionToId.attack, name: 'attack' }
            }

            if (key === 'equip_wooden_axe' && action.equip_wooden_axe === 1) {
                return { id: this.actionToId.equip_wooden_axe, name: 'equip_wooden_axe' }
            }

            if (key === 'camera_yaw' && action.camera_yaw > 0) {
                if (action.camera_yaw === 1) return { id: this.actionToId.camera_yaw_p15, name: 'camera_yaw_p15' }
                if (action.camera_yaw === 2) return { id: this.actionToId.camera_yaw_m15, name: 'camera_yaw_m15' }
                if (action.camera_yaw === 3) return { id: this.actionToId.camera_yaw_p45, name: 'camera_yaw_p45' }
                if (action.camera_yaw === 4) return { id: this.actionToId.camera_yaw_m45, name: 'camera_yaw_m45' }
            }

            if (key === 'camera_pitch' && action.camera_pitch > 0) {
                if (action.camera_pitch === 1) return { id: this.actionToId.camera_pitch_p15, name: 'camera_pitch_p15' }
                if (action.camera_pitch === 2) return { id: this.actionToId.camera_pitch_m15, name: 'camera_pitch_m15' }
                if (action.camera_pitch === 3) return { id: this.actionToId.camera_pitch_p45, name: 'camera_pitch_p45' }
                if (action.camera_pitch === 4) return { id: this.actionToId.camera_pitch_m45, name: 'camera_pitch_m45' }
            }

            if (key === 'move_forward' && action.move_forward > 0) {
                return action.move_forward === 2
                    ? { id: this.actionToId.move_forward_sprint, name: 'move_forward_sprint' }
                    : { id: this.actionToId.move_forward_walk, name: 'move_forward_walk' }
            }

            if (key === 'move_backward' && action.move_backward > 0) {
                return { id: this.actionToId.move_backward_walk, name: 'move_backward_walk' }
            }

            if (key === 'move_lateral' && action.move_lateral > 0) {
                return action.move_lateral === 1
                    ? { id: this.actionToId.move_left, name: 'move_left' }
                    : { id: this.actionToId.move_right, name: 'move_right' }
            }

            if (key === 'move_vertical' && action.move_vertical > 0) {
                return action.move_vertical === 1
                    ? { id: this.actionToId.jump, name: 'jump' }
                    : { id: this.actionToId.sneak, name: 'sneak' }
            }
        }

        return { id: this.actionToId.idle, name: 'idle' }
    }

    /**
     * Reset action to neutral state
     */
    resetAction() {
        this.currentAction = this.getEmptyAction()
    }

    /**
     * Get action sequence
     */
    getActionSequence(limit = null) {
        if (limit && limit > 0) {
            return this.actionSequence.slice(-limit)
        }
        return [...this.actionSequence]
    }

    /**
     * Get current action state
     */
    getCurrentAction() {
        return this._toDiscreteAction(this.currentAction)
    }

    /**
     * Get action statistics
     */
    getActionStats(sequence = this.actionSequence) {
        const stats = {
            total_actions: sequence.length,
            action_distribution: {}
        }

        // Count label distribution
        for (const record of sequence) {
            const key = String(record.action || 'idle')
            if (!stats.action_distribution[key]) stats.action_distribution[key] = 0
            stats.action_distribution[key]++
        }

        return stats
    }

    /**
     * Export all data for training dataset.
     * With synchronous capture, every record already has a screenshot.
     */
    exportForDataset(requireScreenshots = true) {
        const actionSequence = this.getActionSequence()

        // With the synchronous recording loop, ALL records should have screenshots.
        // Filter as a safety net.
        const filteredSequence = requireScreenshots
            ? actionSequence.filter(record => Boolean(record.screenshot))
            : actionSequence

        // Strip internal _ts property before export
        for (const record of filteredSequence) {
            delete record._ts
        }

        const dropped = actionSequence.length - filteredSequence.length
        if (dropped > 0) {
            console.warn(`[RLActionTracker] WARNING: ${dropped}/${actionSequence.length} records had no screenshot (should be 0)`)
        } else {
            console.log(`[RLActionTracker] ✓ All ${filteredSequence.length} records have screenshots`)
        }

        return {
            action_sequence: filteredSequence,
            statistics: {
                ...this.getActionStats(filteredSequence),
                total_records: actionSequence.length,
                records_with_screenshots: filteredSequence.length,
                dropped_without_screenshot: dropped,
                captured_count: this.capturedCount,
                skipped_count: this.skippedCount,
            },
        }
    }

    /**
     * Reset sequence history
     */
    resetSequence() {
        this.actionSequence = []
        this.lastRecordedAction = null
        this.capturedCount = 0
        this.skippedCount = 0
    }
}
