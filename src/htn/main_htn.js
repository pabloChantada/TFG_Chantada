import pkg from 'mineflayer-pathfinder'
const { Movements } = pkg
import minecraftData from 'minecraft-data'
import inventoryViewer from 'mineflayer-web-inventory'

// Custom task
import { runFullProgression } from './progression.js'
import { runSimpleProgression } from './progression_simple.js'

let mcData

/**
 * Start bot with HTN-based progression.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {number} inventoryPort - Port for the inventory viewer (default: 3001)
 * @param {MetricsCollector|null} metricsCollector - Optional metrics collector for performance tracking
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
export async function startHTN(bot, inventoryPort = 3001, metricsCollector = null, progressionType = 'full') {
    mcData = minecraftData(bot.version)
    
    // Increase max listeners to avoid warnings from multiple plugins
    // inventoryViewer(bot, { port: inventoryPort })
    
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)
    
    // Limit pathfinder computation to prevent event loop blocking (keepalive timeout)
    bot.pathfinder.thinkTimeout = 2000  // Max 2s to compute a path before giving up
    bot.pathfinder.tickTimeout = 15     // Max ms per tick for path computation

    bot.on('chat', async (username, message) => {
        if (username === bot.username) return
        if (message === 'craft') await startFullProgression(bot, mcData, metricsCollector)
        if (message === 'simple') await startSimpleProgression(bot, mcData, metricsCollector)
    })
    if (progressionType === 'simple') {
        return await startSimpleProgression(bot, mcData, metricsCollector)
    } else {
        return await startFullProgression(bot, mcData, metricsCollector)
    }
}

// =========================================================
// --- MAIN TASK: FULL PROGRESSION UP TO IRON ---
// =========================================================

/**
 * Start the full progression process, all tasks from basic resource gathering to iron tools.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {object} mcData - Minecraft data for the current version
 * @param {MetricsCollector|null} metricsCollector - Optional metrics collector for performance tracking
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
async function startFullProgression(bot, mcData, metricsCollector = null) {
    try {
        await runFullProgression(bot, mcData, metricsCollector)
        await bot.quit()
        return { success: true };
    } catch (err) {
        bot.chat(`Process stopped: ${err.message}`)
        console.error(err)
        await bot.quit()
        return { success: false };
    }
}

/**
 * Start the simple progression process, dig a 2x2.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {object} mcData - Minecraft data for the current version
 * @param {MetricsCollector|null} metricsCollector - Optional metrics collector for performance tracking
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
async function startSimpleProgression(bot, mcData, metricsCollector = null) {
    try {
        const result = await runSimpleProgression(bot, mcData, metricsCollector)
        await bot.quit()
        return result;
    } catch (err) {
        bot.chat(`Process stopped: ${err.message}`)
        console.error(err)
        await bot.quit()
        return { success: false };
    }
}