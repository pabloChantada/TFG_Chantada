/**
 * Checks if the bot has at least a certain count of an item in its inventory.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} itemName - The name of the item to check (e.g., "coal").
 * @param {number} count - The minimum count of the item required (default is 1).
 * @returns {boolean} - True if the bot has at least the specified count of the item, false otherwise.
 */
function hasItem(bot, mcData, itemName, count = 1) {
    const id = getItemId(mcData, itemName)
    return id ? bot.inventory.count(id) >= count : false
}

/**
 * Gets the item ID for a given item name using the minecraft data.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} itemName - The name of the item (e.g., "coal").
 * @returns {number|null} - The item ID if found, or null if not found.
 */
function getItemId(mcData, itemName) {
    return mcData.itemsByName[itemName]?.id
}

/**
 * Clears the bot's inventory by tossing all items.
 * @param {Bot} bot - The mineflayer bot instance.
 * @returns {Promise<void>}
 */
async function clearInventory(bot) {
    const items = bot.inventory.items()
    for (const item of items) await bot.toss(item.type, null, item.count)
}

export {
    hasItem,
    getItemId,
    clearInventory
}