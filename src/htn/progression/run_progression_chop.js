import { getBotLabel } from './utils.js'
import { phaseChopTrees } from './phases/chopTree.js'

const PROGRESSION_TIMEOUT_MS = 3 * 60 * 1000 // 3 minutes safety net

/**
 * Run a simple progression: chop X trees and quit.
 * Includes a global timeout to prevent the bot from hanging indefinitely.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Object} mcData - Minecraft data
 * @param {number} logCount - Number of logs to collect (default 5)
 */
async function runChopProgression(bot, mcData, logCount = 5) {
    const botLabel = getBotLabel(bot)
    
    try {
        console.log(`[${botLabel}] === CHOP TREES PROGRESSION ===`)
        console.log(`[${botLabel}] Target: ${logCount} logs`)
        
        const taskPromise = phaseChopTrees(bot, mcData, logCount)
        const timeoutPromise = new Promise((_, reject) => {
            const id = setTimeout(() => {
                try { bot.pathfinder.stop() } catch (_) {}
                reject(new Error(`Progression global timeout (${PROGRESSION_TIMEOUT_MS / 1000}s)`))
            }, PROGRESSION_TIMEOUT_MS)
            id.unref?.()   // don't keep process alive just for the timer
        })

        await Promise.race([taskPromise, timeoutPromise])
        
        console.log(`[${botLabel}] === CHOP TREES PROGRESSION COMPLETE ===`)
        
    } catch (err) {
        console.error(`[${botLabel}] Chop progression failed:`, err.message)
        throw err
    }
}

export { runChopProgression }
