import { getBotLabel } from './utils.js'
import { hasItem, getItemId } from '../primitives/inventory.js'
import { chop, obtainWoodType, obtainPlankType } from '../primitives/wood.js'
import { craftItem } from '../tasks/crafting.js'
import { ensureCraftingTable } from '../tasks/block_placement.js'
import { mineBlock } from '../primitives/mining.js'
import { exploreRandom } from '../primitives/movement.js'

/**
 * Ensures the bot has the specified item equipped, with internal logic to recover if missing
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Object} mcData - The minecraft data for the bot's version
 * @param {string} itemName - The name of the item to ensure is equipped
 * @param {Function} materialSourceFn - (Optional) Async function to provide necessary materials
 * @returns {Promise<void>}
 */
async function ensureEquipped(bot, mcData, itemName, materialSourceFn = null) {
    const item = bot.inventory.findInventoryItem(getItemId(mcData, itemName), null)
    
    if (item) {
        await bot.equip(item, 'hand')
        return
    }

    console.log(`[REPLAN] I don't have ${itemName}. Crafting a new one...`)
    
    // Prerequisite: To craft tools we need wood/sticks and a crafting table
    if (!hasItem(bot, mcData, 'stick', 2)) {
        await ensureWoodAndPlanks(bot, mcData, 2)
        await craftItem(bot, mcData, 'stick', 1)
    }

    await ensureCraftingTable(bot, mcData)

    if (materialSourceFn) {
        await materialSourceFn()
    }

    await craftItem(bot, mcData, itemName, 1)
    
    const newItem = bot.inventory.findInventoryItem(getItemId(mcData, itemName), null)
    if (newItem) await bot.equip(newItem, 'hand')
    else throw new Error(`Could not equip ${itemName} even after crafting it.`)
}

/**
 * Ensures the bot has wood and planks
 * @param {Bot} bot - Bot instance
 * @param {Object} mcData - Minecraft data
 * @param {number} amountLogs - Amount of logs to chop
 * @returns {Promise<void>}
 */
async function ensureWoodAndPlanks(bot, mcData, amountLogs = 3) {
    const woodType = await obtainWoodType(bot, mcData)
    if (!hasItem(bot, mcData, woodType, 1)) {
        await chop(bot, mcData, amountLogs)
    }
    
    const plankType = await obtainPlankType(bot, mcData, woodType)
    if (!hasItem(bot, mcData, plankType, 4)) {
        await craftItem(bot, mcData, plankType, 1)
    }
}

export { ensureEquipped, ensureWoodAndPlanks }