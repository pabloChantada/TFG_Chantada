import pkg from 'mineflayer-pathfinder'
const { Movements } = pkg
import minecraftData from 'minecraft-data'
import { runFullProgression } from './progression/run_progression_iron.js'
import { runChopProgression } from './progression/run_progression_chop.js'

let mcData
const LOG_COUNT = 5

/**
 * Setup pathfinder with standard HTN settings
 */
function setupPathfinder(bot, mcData) {
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)
    
    // Limit pathfinder computation to prevent event loop blocking (keepalive timeout)
    bot.pathfinder.thinkTimeout = 2000  // Max 2s to compute a path before giving up
    bot.pathfinder.tickTimeout = 15     // Max ms per tick for path computation
}

/**
 * Start bot with HTN-based progression.
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
export async function startHTN(bot) {
    mcData = minecraftData(bot.version)
    setupPathfinder(bot, mcData)
    return await startFullProgression(bot, mcData)
}

/**
 * Start bot for tree chopping only.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {number} logCount - Number of logs to collect (default 5)
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
export async function startChopTrees(bot, logCount = LOG_COUNT) {
    mcData = minecraftData(bot.version)
    setupPathfinder(bot, mcData)
    return await startChopProgression(bot, mcData, logCount)
}

// =========================================================
// --- MAIN TASK: FULL PROGRESSION UP TO IRON ---
// =========================================================

/**
 * Start the full progression process, all tasks from basic resource gathering to iron tools.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {object} mcData - Minecraft data for the current version
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
async function startFullProgression(bot, mcData) {
    try {
        await runFullProgression(bot, mcData)
        return { success: true }
    } catch (err) {
        bot.chat(`Process stopped: ${err.message}`)
        console.error(err)
        return { success: false }
    }
}

/**
 * Start chop progression process.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {object} mcData - Minecraft data for the current version
 * @param {number} logCount - Number of logs to collect
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
async function startChopProgression(bot, mcData, logCount) {
    try {
        await runChopProgression(bot, mcData, logCount)
        return { success: true }
    } catch (err) {
        bot.chat(`Chop progression stopped: ${err.message}`)
        console.error(err)
        return { success: false }
    }
}