import fs from 'fs'
import path from 'path'
import Vec3 from 'vec3'
import { getBlockId } from './inventory.js'

/**
 * Gets the memory file path for a bot
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {string} - The path to the memory file
 */
function getMemoryPath(bot) {
    return path.join('src', 'agents', 'memories', `${bot.username}_memory.json`)
}

/**
 * Saves a structure position to bot memory
 * @param {Bot} bot - The mineflayer bot instance
 * @param {string} structureName - Name of the structure
 * @param {Vec3} position - Position to save
 */
function saveStructureToMemory(bot, structureName, position) {
    const filePath = getMemoryPath(bot)
    const dir = path.dirname(filePath)
    
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
    }
    
    let memoryData = {}
    if (fs.existsSync(filePath)) {
        try {
            memoryData = JSON.parse(fs.readFileSync(filePath, 'utf8'))
        } catch (_e) {
            memoryData = {}
        }
    }
    
    memoryData[structureName] = { x: position.x, y: position.y, z: position.z }
    fs.writeFileSync(filePath, JSON.stringify(memoryData, null, 2))
    console.log(`[saveStructureToMemory] Saved ${structureName} at ${position}`)
}

/**
 * Clears a structure from bot memory
 * @param {Bot} bot - The mineflayer bot instance
 * @param {string} structureName - Name of the structure to remove
 */
function clearStructureFromMemory(bot, structureName) {
    const filePath = getMemoryPath(bot)
    
    if (!fs.existsSync(filePath)) {
        return
    }
    
    try {
        let memoryData = JSON.parse(fs.readFileSync(filePath, 'utf8'))
        if (memoryData[structureName]) {
            delete memoryData[structureName]
            fs.writeFileSync(filePath, JSON.stringify(memoryData, null, 2))
            console.log(`[clearStructureFromMemory] Cleared ${structureName} from memory`)
        }
    } catch (e) {
        console.warn(`[clearStructureFromMemory] Failed to clear ${structureName}: ${e.message}`)
    }
}

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
    
    if (throwOnMissing) {
        throw {
            type: `${structureName.toUpperCase()}_NOT_FOUND`,
            message: `${structureName} not found or invalid in memory`,
            reason: `${structureName}_missing`
        }
    }

    return null
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
 * Gets the furnace block from memory.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @return {Block|null} - The furnace block, or null if not in memory.
 */
function getFurnace(bot, mcData) {
    return getStructure(bot, mcData, 'furnace', false)
}

/**
 * Clears furnace from bot memory (useful when furnace is broken or not working)
 * @param {Bot} bot - The mineflayer bot instance
 */
function clearFurnaceMemory(bot) {
    clearStructureFromMemory(bot, 'furnace')
}

export {
    getCraftingTable,
    getFurnace,
    clearFurnaceMemory,
    saveStructureToMemory,
    clearStructureFromMemory
}