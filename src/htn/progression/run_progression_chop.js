import { getBotLabel } from './utils.js'
import { phaseChopTrees } from './phases/chopTree.js'

/**
 * Run a simple progression: chop X trees and quit.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Object} mcData - Minecraft data
 * @param {number} logCount - Number of logs to collect (default 5)
 */
async function runChopProgression(bot, mcData, logCount = 5) {
    const botLabel = getBotLabel(bot)
    
    try {
        console.log(`[${botLabel}] === CHOP TREES PROGRESSION ===`)
        console.log(`[${botLabel}] Target: ${logCount} logs`)
        
        await phaseChopTrees(bot, mcData, logCount)
        
        console.log(`[${botLabel}] === CHOP TREES PROGRESSION COMPLETE ===`)
        
    } catch (err) {
        console.error(`[${botLabel}] Chop progression failed:`, err.message)
        throw err
    }
}

export { runChopProgression }
