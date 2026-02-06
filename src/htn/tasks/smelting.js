import { getSmeltInputName } from '../primitives/helpers.js'
import { getFurnace } from '../primitives/structures.js'
import { moveToBlock } from '../primitives/movement.js'

/**
 * Smelts an item using a furnace stored in memory.
 * Assumes the planning layer has already ensured the required raw materials and fuel.
 *
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} itemName - The output item to smelt (e.g., "iron_ingot").
 * @param {number} amountNeeded - The number of items to obtain (default is 1).
 * @throws {Error} - If furnace missing, materials missing, or item invalid.
 */
async function smeltItem(bot, mcData, itemName, amountNeeded = 1) {
    const item = mcData.itemsByName[itemName]
    if (!item) throw new Error(`[ERROR] ${bot.username} Invalid item: ${itemName}`)

    const furnaceBlock = getFurnace(bot, mcData)
    if (!furnaceBlock) throw new Error(`[ERROR] ${bot.username} Furnace not found in memory`)
    
    // Get the required input item for the desired smelting output
    const inputName = getSmeltInputName(itemName)
    if (!inputName) throw new Error(`[ERROR] ${bot.username} No smelting input for ${itemName}`)
    const inputItem = mcData.itemsByName[inputName]
    if (!inputItem) throw new Error(`[ERROR] ${bot.username} Input item missing: ${inputName}`)

    // Check for raw materials and fuel in inventory
    const rawMaterial = bot.inventory.findInventoryItem(inputItem.id)
    const fuel = bot.inventory.findInventoryItem(mcData.itemsByName.coal.id) // TODO: generalize fuels

    if (!rawMaterial || !fuel) throw new Error(`[ERROR] ${bot.username} Missing smelting materials or fuel`)

    // Move to furnace and open it
    await moveToBlock(bot, furnaceBlock, 3)
    const furnace = await bot.openFurnace(furnaceBlock)
    await furnace.putFuel(fuel.type, null, fuel.count)
    await furnace.putInput(rawMaterial.type, null, rawMaterial.count)

    let retries = 0
    while (retries < 5) { // Max 50 segundos
        await new Promise(resolve => setTimeout(resolve, 10000))

        try {
            await furnace.takeOutput()
        } catch (_e) {
            // Sometimes empty; ignore
        }

        const currentOutput = bot.inventory.count(item.id)
        if (currentOutput >= amountNeeded) {
            break
        }
        retries++
    }

    furnace.close()
}

export { smeltItem }