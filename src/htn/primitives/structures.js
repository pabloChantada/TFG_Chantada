import fs from 'fs'
import Vec3 from 'vec3'
import { getBlockId } from './inventory.js'

/**
 * Obtains a stored structure (crafting_table, furnace, etc) from bot memory
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Object} mcData - The minecraft data
 * @param {string} structureName - Name of the structure ('crafting_table', 'furnace')
 * @param {boolean} throwOnMissing - If true, throws error; if false, returns null (default: false)
 * @returns {Block|null|Error} - The block if found, null if not found (or error if throwOnMissing)
 */
function getStructure(bot, mcData, structureName, throwOnMissing = false) {
    const path = `src/agents/memories/${bot.username}_memory.json`
    
    // Try to load from memory
    let memoryData = null
    if (fs.existsSync(path)) {
        try {
            memoryData = JSON.parse(fs.readFileSync(path, 'utf8'))
        } catch (e) {
            console.warn(`[getStructure] Failed to read memory: ${e.message}`)
        }
    }
    
    // Get structure position from memory
    const pos = memoryData?.[structureName]
    if (!pos) {
        if (throwOnMissing) {
            throw {
                type: `${structureName.toUpperCase()}_NOT_FOUND`,
                message: `${structureName} not found in memory`,
                reason: `${structureName}_missing`
            }
        }
        return null
    }
    
    // Verify block still exists and is correct type
    const structurePos = new Vec3(pos.x, pos.y, pos.z)
    const block = bot.blockAt(structurePos)
    const blockId = getBlockId(mcData, structureName)
    
    if (block && block.type === blockId) {
        console.log(`[getStructure] Found ${structureName} at ${structurePos}`)
        return block
    }
    
    return throwOnMissing ? null : null
}

/**
 * Gets the crafting table block from memory, throws error if not found or invalid.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @return {Block} - The crafting table block.
 * @throws {Error} - If the crafting table is not found or invalid. 
 */
function getCraftingTable(bot, mcData) {
    return getStructure(bot, mcData, 'crafting_table', true) 
}

/**
 * Gets the furnace block from memory, returns null if not found or invalid.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @return {Block|null} - The furnace block.
 * @throws {Error} - If the furnace is not found or invalid. 
 */
function getFurnace(bot, mcData) {
    return getStructure(bot, mcData, 'furnace', false) 
}

export {
    getCraftingTable,
    getFurnace
}