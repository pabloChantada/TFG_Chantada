/**
 * Gym-like environment wrapper for Mineflayer bot.
 * Manages the step loop: observe → act → reward → done.
 * 
 * This does NOT use omniscient APIs (no bot.findBlock for state).
 * The bot interacts with the world through its camera (raycast) and controls.
 */

import { extractState, STATE_DIM, getLogCount, isLog } from './state.js'
import { applyAction, releaseAllControls, NUM_ACTIONS, ACTION_NAMES } from './actions.js'
import { computeReward, DEFAULT_REWARD_CONFIG } from './reward.js'

export class WoodChopEnvironment {
    /**
     * @param {import('mineflayer').Bot} bot - Mineflayer bot instance
     * @param {object} options
     * @param {number} options.targetLogs - Logs to collect per episode (default 5)
     * @param {number} options.maxSteps - Max steps per episode (default 2000)
     * @param {number} options.ticksPerStep - Ticks to wait between steps (default 4 = ~200ms)
     * @param {'clear_inventory'|'relative'|'none'} options.resetMode - Inventory reset strategy
     * @param {number} options.stuckThreshold - Steps without progress before considering stuck (default 100)
     * @param {import('../metrics/metrics_collector.js').MetricsCollector} options.metricsCollector - Optional metrics
     */
    constructor(bot, options = {}) {
        this.bot = bot
        this.targetLogs = options.targetLogs || 5
        this.maxSteps = options.maxSteps || 2000
        this.ticksPerStep = options.ticksPerStep || 4
        this.metricsCollector = options.metricsCollector || null
        
        // Reset mode configuration
        this.resetMode = options.resetMode || 'clear_inventory'
        
        // Stuck detection
        this.stuckThreshold = options.stuckThreshold || 100
        this.lastProgressStep = 0
        this.lastProgressLogCount = 0
        this.stuckCheckInterval = options.stuckCheckInterval || 50

        this.rewardConfig = {
            ...DEFAULT_REWARD_CONFIG,
            targetLogs: this.targetLogs,
        }

        // Episode state
        this.stepCount = 0
        this.totalReward = 0
        this.prevStateMeta = null
        this.initialLogCount = 0
        this.episodeHistory = []
        this.stuckCount = 0

        // Observation/action space info (for external consumers)
        this.stateDim = STATE_DIM
        this.numActions = NUM_ACTIONS
        this.actionNames = ACTION_NAMES
    }

    /**
     * Reset the environment for a new episode.
     * Clears episode counters and optionally clears the bot's inventory.
     * 
     * @returns {{ observation: number[], meta: object }}
     */
    async reset() {
        await releaseAllControls(this.bot)

        // Stop any ongoing digging
        if (this.bot.targetDigBlock) {
            try {
                await this.bot.stopDigging()
            } catch (e) {
                // Ignore errors during stop
            }
        }

        // Reset episode tracking
        this.stepCount = 0
        this.totalReward = 0
        this.episodeHistory = []
        this.lastProgressStep = 0
        this.lastProgressLogCount = 0
        this.stuckCount = 0

        // Handle inventory reset based on mode
        await this._handleInventoryReset()

        const { vector, meta } = extractState(this.bot)
        this.prevStateMeta = meta
        this.initialLogCount = meta.logCount
        this.lastProgressLogCount = meta.logCount

        console.log(`[RL-ENV] Episode reset. Initial logs: ${this.initialLogCount}, target: ${this.targetLogs}`)

        return { observation: vector, meta }
    }

    /**
     * Handle inventory reset based on configured mode.
     * @private
     */
    async _handleInventoryReset() {
        switch (this.resetMode) {
            case 'clear_inventory':
                // Drop all logs to start fresh
                await this._dropAllLogs()
                console.log('[RL-ENV] Inventory cleared (dropped all logs)')
                break

            case 'relative':
                // Keep current inventory, but measure success relative to initial state
                console.log('[RL-ENV] Using relative goal (collect N more logs from current state)')
                break

            case 'none':
                // Keep everything as-is
                console.log('[RL-ENV] No inventory reset')
                break

            default:
                console.warn(`[RL-ENV] Unknown resetMode: ${this.resetMode}, using 'relative'`)
        }
    }

    /**
     * Drop all log items from inventory.
     * @private
     */
    async _dropAllLogs() {
        const logItems = this.bot.inventory.items().filter(item => isLog(item.name))
        
        for (const item of logItems) {
            try {
                await this.bot.tossStack(item)
                await this.bot.waitForTicks(2) // Small delay between drops
            } catch (err) {
                console.warn(`[RL-ENV] Failed to drop ${item.name}: ${err.message}`)
            }
        }
    }

    /**
     * Check if the bot is stuck (mining stone/hard blocks without progress).
     * @private
     * @returns {boolean}
     */
    _isStuck(currentLogCount) {
        // Check if we've made progress recently
        if (currentLogCount > this.lastProgressLogCount) {
            // Progress made!
            this.lastProgressStep = this.stepCount
            this.lastProgressLogCount = currentLogCount
            return false
        }

        // Check every N steps
        if (this.stepCount % this.stuckCheckInterval !== 0) {
            return false
        }

        const stepsSinceProgress = this.stepCount - this.lastProgressStep
        const isStuck = stepsSinceProgress >= this.stuckThreshold

        if (isStuck) {
            console.warn(
                `[RL-ENV] Bot appears stuck! ${stepsSinceProgress} steps without collecting logs. ` +
                `Possibly mining hard blocks (stone, dirt) without proper tool.`
            )
        }

        return isStuck
    }

    /**
     * Unstuck the bot by stopping digging and releasing controls.
     * @private
     */
    async _unstuck() {
        this.stuckCount++
        console.log(`[RL-ENV] Unsticking bot (attempt ${this.stuckCount})...`)

        // Stop digging if active
        if (this.bot.targetDigBlock) {
            try {
                await this.bot.stopDigging()
                console.log(`[RL-ENV] Stopped digging ${this.bot.targetDigBlock.name}`)
            } catch (e) {
                // Ignore
            }
        }

        // Release all controls
        await releaseAllControls(this.bot)

        // Small random movement to get unstuck
        const randomActions = [
            'back',      // Move back
            'turn_left', // Turn left
            'turn_right',// Turn right
            'jump',      // Jump
        ]
        const randomAction = randomActions[Math.floor(Math.random() * randomActions.length)]
        const actionIndex = ACTION_NAMES.indexOf(randomAction)
        
        console.log(`[RL-ENV] Executing recovery action: ${randomAction}`)
        await applyAction(this.bot, actionIndex)
        await this.bot.waitForTicks(10)
        await releaseAllControls(this.bot)

        // Reset progress tracking
        this.lastProgressStep = this.stepCount
    }

    /**
     * Execute one step in the environment.
     * 
     * @param {number} actionIndex - Index into ACTION_NAMES
     * @returns {Promise<{ observation: number[], reward: number, done: boolean, truncated: boolean, info: object }>}
     */
    async step(actionIndex) {
        // 1. Apply the action
        await applyAction(this.bot, actionIndex)

        // 2. Wait for the action to take effect
        await this.bot.waitForTicks(this.ticksPerStep)

        // 3. Observe new state
        const { vector, meta } = extractState(this.bot)

        // 4. Check if stuck (before computing reward)
        const currentLogCount = meta.logCount - this.initialLogCount
        if (this._isStuck(currentLogCount)) {
            await this._unstuck()
            // Re-observe after unstuck
            const { vector: newVector, meta: newMeta } = extractState(this.bot)
            meta.stuck_recovery = true
            Object.assign(meta, newMeta)
            vector.splice(0, vector.length, ...newVector)
        }

        // 5. Compute reward
        const { reward, done, info } = computeReward(
            this.prevStateMeta, meta, this.bot, this.rewardConfig
        )

        // Apply penalty for getting stuck
        let finalReward = reward
        if (meta.stuck_recovery) {
            finalReward -= 2.0 // Penalty for stuck behavior
            info.components.stuck_penalty = -2.0
            console.log(`[RL-ENV] Applied stuck penalty: -2.0`)
        }

        this.stepCount++
        this.totalReward += finalReward

        // 6. Check truncation (max steps)
        const truncated = this.stepCount >= this.maxSteps

        // 7. Record transition
        const transition = {
            step: this.stepCount,
            action: ACTION_NAMES[actionIndex],
            action_index: actionIndex,
            reward: round(finalReward, 4),
            total_reward: round(this.totalReward, 4),
            done,
            truncated,
            stuck_recovery: meta.stuck_recovery || false,
            state_meta: meta,
            info
        }
        this.episodeHistory.push(transition)

        // 8. Log periodically
        if (this.stepCount % 100 === 0 || done || truncated) {
            const logsCollected = meta.logCount - this.initialLogCount
            console.log(
                `[RL-ENV] Step ${this.stepCount}/${this.maxSteps} | ` +
                `Reward: ${round(this.totalReward, 2)} | ` +
                `Logs: ${logsCollected}/${this.targetLogs} | ` +
                `Action: ${ACTION_NAMES[actionIndex]}` +
                (meta.stuck_recovery ? ' [UNSTUCK]' : '')
            )
        }

        // 9. Update previous state
        this.prevStateMeta = meta

        // 10. Release controls if episode is over
        if (done || truncated) {
            await releaseAllControls(this.bot)
            if (this.bot.targetDigBlock) {
                try {
                    await this.bot.stopDigging()
                } catch (e) {
                    // Ignore
                }
            }
            info.done_reason = info.done_reason || (truncated ? 'max_steps' : 'target_reached')
            info.episode_summary = this.getEpisodeSummary()
        }

        return {
            observation: vector,
            reward: finalReward,
            done,
            truncated,
            info
        }
    }

    /**
     * Get a summary of the current/last episode.
     * @returns {object}
     */
    getEpisodeSummary() {
        const currentLogs = getLogCount(this.bot)
        return {
            steps: this.stepCount,
            total_reward: round(this.totalReward, 4),
            logs_collected: currentLogs - this.initialLogCount,
            target_logs: this.targetLogs,
            success: (currentLogs - this.initialLogCount) >= this.targetLogs,
            initial_logs: this.initialLogCount,
            final_logs: currentLogs,
            stuck_recoveries: this.stuckCount,
        }
    }

    /**
     * Export episode data for training.
     * Compatible with data/train.jsonl format.
     * @returns {object[]}
     */
    exportEpisodeData() {
        return [...this.episodeHistory]
    }
}

function round(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals)
}