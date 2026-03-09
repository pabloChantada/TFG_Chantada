import pkg from 'mineflayer-pathfinder'
const { Movements } = pkg
import minecraftData from 'minecraft-data'
import { runFullProgression } from './progression/run_progression.js'

let mcData

/**
 * Start bot with HTN-based progression.
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
export async function startHTN(bot) {
    mcData = minecraftData(bot.version)
    
    // Increase max listeners to avoid warnings from multiple plugins
    // inventoryViewer(bot, { port: inventoryPort })
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    // Strongly discourage routes through water during progression tasks.
    defaultMove.liquidCost = 20
    bot.pathfinder.setMovements(defaultMove)
    
    // Limit pathfinder computation to prevent event loop blocking (keepalive timeout)
    bot.pathfinder.thinkTimeout = 2000  // Max 2s to compute a path before giving up
    bot.pathfinder.tickTimeout = 15     // Max ms per tick for path computation

    return await startFullProgression(bot, mcData)
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
        await bot.quit()
        return { success: true }
    } catch (err) {
        bot.chat(`Process stopped: ${err.message}`)
        console.error(err)
        await bot.quit()
        return { success: false }
    }
}