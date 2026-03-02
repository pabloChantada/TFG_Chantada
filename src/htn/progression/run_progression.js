import { phaseWood } from './phases/wood.js'
import { phaseStone } from './phases/stone.js'
import { phaseIron } from './phases/iron.js'
import { getBotLabel } from './utils.js'

/**
 * Main orchestrator for full progression
 * @param {Bot} bot - Bot instance
 * @param {Object} mcData - Minecraft data
 * @param {Object} metricsCollector - (Optional) Metrics collector
 * @returns {Promise<void>}
 */
async function runFullProgression(bot, mcData) {
    console.log(`[INFO] [${getBotLabel(bot)}] Starting Progression`)
    
    try {
        await phaseWood(bot, mcData)
        console.log(`[INFO] [${getBotLabel(bot)}] --- Wood Phase Complete ---`)
        
        await phaseStone(bot, mcData)
        console.log(`[INFO] [${getBotLabel(bot)}] --- Stone Phase Complete ---`)
        
        await phaseIron(bot, mcData)
        console.log(`[SUCCESS] [${getBotLabel(bot)}] Mission Complete.`)
        
    } catch (error) {
        console.log(`[FATAL] [${getBotLabel(bot)}] Progression stopped: ${error.message}`)
        console.error(error)
        throw error
    }
}

export { runFullProgression }