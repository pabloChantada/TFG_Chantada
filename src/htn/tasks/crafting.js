import { getCraftingTable } from '../primitives/structures.js'
import { moveToBlock } from '../primitives/movement.js'

/**
 * Crafts an item. If the recipe requires a crafting table, retrieves it from memory and moves to it.
 * Assumes all required materials are already in the bot's inventory.
 * The planning layer is responsible for ensuring materials and crafting table are available.
 * 
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} itemName - The name of the item to craft (e.g., "stick", "iron_pickaxe").
 * @param {number} amount - The number of items to craft (default is 1).
 * @throws {Error} - If the item is invalid, recipe not found, or crafting table missing from memory.
 */
async function craftItem(bot, mcData, itemName, amount = 1) {
    const item = mcData.itemsByName[itemName]
    if (!item) throw new Error(`[ERROR] ${bot.username} Invalid item: ${itemName}`)

    // Try recipe without table first
    let table = null
    let recipe = bot.recipesFor(item.id, null, amount, null)[0]

    if (!recipe) {
        // Try with crafting table from memory (throws if not found)
        table = getCraftingTable(bot, mcData)
        recipe = bot.recipesFor(item.id, null, amount, table)[0]
    }

    if (!recipe) throw new Error(`[ERROR] ${bot.username} Cannot craft ${itemName} (missing materials or recipe not found)`)

    // If recipe needs table, move to it
    if (recipe.requiresTable) {
        if (!table) table = getCraftingTable(bot, mcData)

        await moveToBlock(bot, table, 3)
        await bot.waitForTicks(10)

        // Re-verify table after moving (chunk might have changed)
        table = getCraftingTable(bot, mcData)
    }

    await bot.craft(recipe, amount, recipe.requiresTable ? table : null)
    bot.chat(`[CRAFT] ${amount}x ${itemName}`)
}

export { craftItem }