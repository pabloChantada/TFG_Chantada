/**
 * Reward functions for the wood-chopping RL task.
 * 
 * Design principles:
 *   - Positive reward for collecting logs (primary goal)
 *   - Small shaping rewards for looking at logs (encourages exploration → alignment)
 *   - Small penalty per timestep (encourages efficiency)
 *   - Penalty for taking damage
 *   - Episode terminates when target log count reached or max steps exceeded
 */

import { isLog, getLogCount } from './state.js'

/**
 * @typedef {Object} RewardConfig
 * @property {number} targetLogs - Number of logs to collect for episode success
 * @property {number} logCollectedReward - Reward per log collected
 * @property {number} lookingAtLogReward - Small reward for looking at a log (shaping)
 * @property {number} stepPenalty - Small penalty each step (encourages speed)
 * @property {number} damagePenalty - Penalty per health point lost
 * @property {number} miningLogReward - Reward for actively mining a log block
 */

/** @type {RewardConfig} */
export const DEFAULT_REWARD_CONFIG = {
    targetLogs: 5,
    logCollectedReward: 10.0,
    lookingAtLogReward: 0.5,
    stepPenalty: -0.01,
    damagePenalty: -1.0,
    miningLogReward: 0.5,
}

/**
 * Compute the reward for the current transition.
 * 
 * @param {object} prevState - Previous state meta (from extractState().meta)
 * @param {object} currState - Current state meta
 * @param {import('mineflayer').Bot} bot - Bot instance
 * @param {RewardConfig} config - Reward configuration
 * @returns {{ reward: number, done: boolean, info: object }}
 */
export function computeReward(prevState, currState, bot, config = DEFAULT_REWARD_CONFIG) {
    let reward = config.stepPenalty
    const info = { components: {} }

    // 1. Log collection reward (the main signal)
    const logsDelta = currState.logCount - prevState.logCount
    if (logsDelta > 0) {
        const logReward = logsDelta * config.logCollectedReward
        reward += logReward
        info.components.log_collected = logReward
        console.log(`[RL-REWARD] +${logsDelta} logs collected! Reward: +${logReward.toFixed(2)}`)
    }
    
    // On negative delta (dropped/consumed logs), no penalty
    // This allows the environment to clear inventory between episodes
    if (logsDelta < 0) {
        console.log(`[RL-REWARD] Log count decreased by ${Math.abs(logsDelta)} (inventory cleared or consumed)`)
    }

    // Looking at log (shaping reward - helps the agent learn to orient toward trees)
    if (currState.lookingAtLog) {
        reward += config.lookingAtLogReward
        info.components.looking_at_log = config.lookingAtLogReward
    }

    // Actively mining a log (shaping)
    if (bot.targetDigBlock && isLog(bot.targetDigBlock.name)) {
        reward += config.miningLogReward
        info.components.mining_log = config.miningLogReward
    }

    // 4. Damage penalty
    const prevHealth = prevState.health || 20
    const currHealth = bot.health || 20
    if (currHealth < prevHealth) {
        const dmgPenalty = (prevHealth - currHealth) * config.damagePenalty
        reward += dmgPenalty
        info.components.damage = dmgPenalty
    }

    const done = currState.logCount >= config.targetLogs

    info.total_reward = reward
    info.logs_collected = currState.logCount
    info.target_logs = config.targetLogs
    info.done_reason = done ? 'target_reached' : null

    return { reward, done, info }
}