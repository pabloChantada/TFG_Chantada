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
 * Finds the nearest block of a given type within a specified distance.
 * Note: This is an omniscient search — it finds blocks even if buried underground.
 * Use findNearestVisibleBlock for a realistic, non-omniscient search.
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

/**
 * Checks if a block has at least one exposed face (adjacent to air or non-solid block).
 * A block with an exposed face is "visible" — a player could actually see it in-game.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Vec3} blockPos - The position of the block to check.
 * @returns {boolean} - True if the block has at least one exposed face.
 */
function isBlockExposed(bot, blockPos) {
    // Check the 6 adjacent blocks (up, down, north, south, east, west)
    const offsets = [
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
    ]
    for (const [dx, dy, dz] of offsets) {
        // Check neightbours blocks
        const neighbor = bot.blockAt(blockPos.offset(dx, dy, dz))
        // If any is air or non-solid, the block is exposed
        if (!neighbor || neighbor.boundingBox === 'empty') {
            return true
        }
    }
    return false
}

/**
 * Finds the nearest block that is visible (has at least one exposed face).
 * Unlike findNearestBlock, this won't return blocks completely buried underground.
 * Simulates realistic player vision — only blocks in caves, on surfaces, or in dug tunnels.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block to find (e.g., "iron_ore").
 * @param {number} maxDistance - Maximum search distance (default 32).
 * @returns {Block|null} - The nearest visible block, or null if none found.
 */
function findNearestVisibleBlock(bot, mcData, blockName, maxDistance = 32) {
    const blockId = getBlockId(mcData, blockName)
    if (!blockId) return null

    const positions = bot.findBlocks({
        matching: blockId,
        maxDistance,
        count: 256
    })

    for (const pos of positions) {
        if (isBlockExposed(bot, pos)) {
            return bot.blockAt(pos)
        }
    }

    return null
}

export {
    getBlockId,
    findNearestBlock,
    findNearestVisibleBlock,
    isBlockExposed
}
