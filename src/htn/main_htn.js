import pkg from 'mineflayer-pathfinder'
const { Movements } = pkg
import minecraftData from 'minecraft-data'
import { runFullProgression } from './progression/run_progression_iron.js'
import { runChopProgression } from './progression/run_progression_chop.js'
import { setupRecorder } from '../il/dataset_recorder.js'

let mcData
const LOG_COUNT = 5

/**
 * Setup pathfinder with standard HTN settings
 */
function setupPathfinder(bot, mcData, options = {}) {
    const { canDig = true } = options
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = canDig
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)
    
    // Limit pathfinder computation to prevent event loop blocking (keepalive timeout)
    bot.pathfinder.thinkTimeout = 2000  // Max 2s to compute a path before giving up
    bot.pathfinder.tickTimeout = 15     // Max ms per tick for path computation

    // ── Stuck-detection watchdog ────────────────────────────────────
    // Every STUCK_CHECK_INTERVAL ms we sample the bot's position.
    // If the bot hasn't moved more than STUCK_THRESHOLD blocks in
    // STUCK_MAX_SAMPLES consecutive checks while the pathfinder is
    // active, force-stop the pathfinder so the higher-level retry
    // logic can kick in.
    const STUCK_CHECK_INTERVAL = 3000   // ms between samples
    const STUCK_THRESHOLD      = 1.5    // blocks
    const STUCK_MAX_SAMPLES    = 4      // consecutive stale samples → stuck
    let lastPos = null
    let staleCount = 0

    const watchdog = setInterval(() => {
        if (!bot.entity) return
        const pos = bot.entity.position

        // Only check when pathfinder is actively moving
        const isPathfinding = bot.pathfinder.isMoving?.() ?? 
                              (bot.pathfinder.goal != null)
        if (!isPathfinding) {
            lastPos = null
            staleCount = 0
            return
        }

        if (lastPos) {
            const dist = pos.distanceTo(lastPos)
            if (dist < STUCK_THRESHOLD) {
                staleCount++
                if (staleCount >= STUCK_MAX_SAMPLES) {
                    console.warn(`[Watchdog] Bot stuck (${staleCount} checks, moved ${dist.toFixed(2)} blocks). Forcing pathfinder stop.`)
                    try { bot.pathfinder.stop() } catch (_) {}
                    staleCount = 0
                }
            } else {
                staleCount = 0
            }
        }
        lastPos = pos.clone()
    }, STUCK_CHECK_INTERVAL)
    watchdog.unref?.()

    // Clean up watchdog when bot disconnects
    bot.once('end', () => clearInterval(watchdog))
}

/**
 * Start bot with HTN-based progression.
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {Promise<{success: boolean}>} Result of the progression process
 */
export async function startHTN(bot) {
    mcData = minecraftData(bot.version)
    setupPathfinder(bot, mcData, { canDig: true })
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
    setupPathfinder(bot, mcData, { canDig: false })
    const viewerPort = parseInt(process.env.VIEWER_PORT || '3000', 10)
    await setupRecorder(bot, mcData, viewerPort)
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