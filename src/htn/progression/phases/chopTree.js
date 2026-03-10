import { getBotLabel } from '../utils.js'
import { runSmartTask } from '../smart_task.js'
import { hasItem } from '../../primitives/inventory.js'
import { chop, obtainWoodType } from '../../primitives/wood.js'

/**
 * Phase for collecting logs by chopping trees.
 * Simple progression: collect X logs and quit.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Object} mcData - Minecraft data
 * @param {number} logCount - Number of logs to collect (default 5)
 */
async function phaseChopTrees(bot, mcData, logCount = 15) {
    const botLabel = getBotLabel(bot)
    console.log(`[${botLabel}] Starting tree chopping progression: target ${logCount} logs`)

    // Determine wood type
    const woodType = await obtainWoodType(bot, mcData)
    console.log(`[${botLabel}] Wood type: ${woodType}`)

    // Task: Collect logs by chopping trees
    await runSmartTask(
        bot,
        `Chop ${logCount} Trees`,
        async () => hasItem(bot, mcData, woodType, logCount),
        async () => chop(bot, mcData, logCount),
        null,
        { exploreOnFail: false }
    )

    console.log(`[${botLabel}] Successfully collected ${logCount} logs!`)
}

export { phaseChopTrees }
