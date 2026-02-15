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
 * Gets the block ID for a given block name using the minecraft data.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block (e.g., "coal_ore").
 * @returns {number|null} - The block ID if found, or null if not found.
 * Note: This function is used in mining tasks to find ores in the world.
 */
function getBlockId(mcData, blockName) {
    return mcData.blocksByName[blockName]?.id
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

/**
 * Finds the nearest block of a given type within a specified distance.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block to find (e.g., "coal_ore").
 * @param {number} maxDistance - The maximum distance to search for the block (default is 32).
 * @returns {Block|null} - The nearest block if found, or null if not found.
 */
function findNearestBlock(bot, mcData, blockName, maxDistance = 32) {
    const blockId = getBlockId(mcData, blockName)
    if (!blockId) return null
    return bot.findBlock({ matching: blockId, maxDistance })
}

export {
    hasItem,
    getItemId,
    getBlockId,
    clearInventory,
    findNearestBlock
}