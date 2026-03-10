/**
 * RL Action Space Tracker
 * Tracks and records actions in the discrete action space format required for RL training
 * 
 * Action Space:
 * - move_forward:  Discrete(3) - 0=still, 1=walk, 2=sprint
 * - move_backward: Discrete(2) - 0=still, 1=walk
 * - move_lateral:  Discrete(3) - 0=still, 1=left, 2=right
 * - move_vertical: Discrete(3) - 0=still, 1=jump, 2=sneak
 * - camera_yaw:    Discrete(5) - 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
 * - camera_pitch:  Discrete(5) - 0=nothing, 1=+15°, 2=-15°, 3=+45°, 4=-45°
 * - attack:        Discrete(2) - 0=no, 1=attack
 * - craft:         Discrete(7) - none/planks/stick/crafting_table/wpick/spick/ipick
 * - smelt:         Discrete(2) - none/iron_ingot
 * - place:         Discrete(4) - none/crafting_table/furnace/torch
 * - equip:         Discrete(5) - none/wpick/spick/ipick/axe
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
        
        // Current action state (what will be encoded as the current action vector)
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
            craft: 0,           // 0=none, 1=planks, 2=stick, 3=crafting_table, 4=wpick, 5=spick, 6=ipick
            smelt: 0,           // 0=none, 1=iron_ingot
            place: 0,           // 0=none, 1=crafting_table, 2=furnace, 3=torch
            equip: 0            // 0=none, 1=wpick, 2=spick, 3=ipick, 4=axe
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
     * Update craft action
     * @param {string} itemName - none/planks/stick/crafting_table/wooden_pickaxe/stone_pickaxe/iron_pickaxe
     */
    updateCraftAction(itemName) {
        const craftMapping = {
            'none': 0,
            'planks': 1,
            'oak_planks': 1,
            'stick': 2,
            'crafting_table': 3,
            'wooden_pickaxe': 4,
            'stone_pickaxe': 5,
            'iron_pickaxe': 6
        }
        this.currentAction.craft = craftMapping[itemName] || 0
    }

    /**
     * Update smelt action
     * @param {string} itemName - none/iron_ingot
     */
    updateSmeltAction(itemName) {
        const smeltMapping = {
            'none': 0,
            'iron_ingot': 1
        }
        this.currentAction.smelt = smeltMapping[itemName] || 0
    }

    /**
     * Update place action
     * @param {string} blockName - none/crafting_table/furnace/torch
     */
    updatePlaceAction(blockName) {
        const placeMapping = {
            'none': 0,
            'crafting_table': 1,
            'furnace': 2,
            'torch': 3
        }
        this.currentAction.place = placeMapping[blockName] || 0
    }

    /**
     * Update equip action
     * @param {string} itemName - none/wooden_pickaxe/stone_pickaxe/iron_pickaxe/wooden_axe
     */
    updateEquipAction(itemName) {
        const equipMapping = {
            'none': 0,
            'wooden_pickaxe': 1,
            'stone_pickaxe': 2,
            'iron_pickaxe': 3,
            'wooden_axe': 4
        }
        this.currentAction.equip = equipMapping[itemName] || 0
    }

    /**
     * Record current action state to sequence.
     * Captures screenshot SYNCHRONOUSLY — no record is created without a screenshot.
     * This guarantees every action in the dataset has a corresponding image.
     * @returns {Object|null} the action record, or null if skipped
     */
    async recordAction() {
        const now = Date.now()
        
        // Avoid recording identical actions too frequently
        // Use direct value comparison instead of JSON.stringify to reduce GC pressure
        if (this.lastRecordedAction && 
            this._actionsEqual(this.currentAction, this.lastRecordedAction.action) &&
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
            action: { ...this.currentAction },
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
        return a.move_forward === b.move_forward &&
               a.move_backward === b.move_backward &&
               a.move_lateral === b.move_lateral &&
               a.move_vertical === b.move_vertical &&
               a.camera_yaw === b.camera_yaw &&
               a.camera_pitch === b.camera_pitch &&
               a.attack === b.attack &&
               a.craft === b.craft &&
               a.smelt === b.smelt &&
               a.place === b.place &&
               a.equip === b.equip
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
        return { ...this.currentAction }
    }

    /**
     * Get action statistics
     */
    getActionStats(sequence = this.actionSequence) {
        const stats = {
            total_actions: sequence.length,
            action_distribution: {},
            total_by_type: {}
        }

        // Initialize counters for each action type
        const actionTypes = Object.keys(this.getEmptyAction())
        for (const type of actionTypes) {
            stats.total_by_type[type] = {}
        }

        // Count action distributions
        for (const record of sequence) {
            for (const [actionType, value] of Object.entries(record.action)) {
                if (!stats.total_by_type[actionType][value]) {
                    stats.total_by_type[actionType][value] = 0
                }
                stats.total_by_type[actionType][value]++
            }
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
