import fs from 'fs'
import path from 'path'
import Vec3 from 'vec3'

import { getItemId, hasItem } from '../primitives/inventory.js'
import { getCraftingTable } from '../primitives/structures.js'
import { moveToBlock } from '../primitives/movement.js'

/**
 * Places a block in front of the bot, checking for valid ground and updating memory if it's a crafting table or furnace.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - Block to place (e.g., "crafting_table" or "furnace").
 * @throws {Error} - If the block cannot be placed due to invalid ground or missing item.
 */
async function placeBlockFull(bot, mcData, blockName) {
    // Obtain the item data and check if we have it in inventory
    const itemId = getItemId(mcData, blockName)
    const item = bot.inventory.findInventoryItem(itemId)
    
    if (!item) throw new Error(`[ERROR] ${bot.username} I don't have a ${blockName} to place`)
    
    // Obtain the position of the blocks in memory
    const memoryData = loadMemory(bot)

    // Equip the item and place it
    await bot.equip(item, 'hand')
    let newBlockPos =
        bot.entity.position.offset(-Math.sin(bot.entity.yaw) * 2, 0, -Math.cos(bot.entity.yaw) * 2).floored()
    let groundBlockPos = newBlockPos.offset(0, -1, 0)
    
    // Avoid placing on top of existing structures in memory
    if (isPosInMemory(newBlockPos, memoryData)) {
        newBlockPos = newBlockPos.offset(1, 0, 0)
        groundBlockPos = newBlockPos.offset(0, -1, 0)
    }

    const groundBlock = bot.blockAt(groundBlockPos)
    if (!groundBlock) {
        throw new Error(`[ERROR] ${bot.username} No ground block found to place ${blockName}`)
    }

    try {
        await bot.placeBlock(groundBlock, new Vec3(0, 1, 0))
    } catch (e) {
        const expectedBlockId = mcData.blocksByName[blockName]?.id
        // First check the expected position
        const placedBlock = bot.blockAt(newBlockPos)
        if (expectedBlockId && placedBlock?.type === expectedBlockId) {
            console.warn(`[placeBlock] ${bot.username} Placement event timeout, but block exists at expected pos. Saving to memory.`)
        } else {
            // Scan nearby blocks — the block may have been placed at a slightly different position
            let found = false
            if (expectedBlockId) {
                const searchRadius = 3
                const botPos = bot.entity.position.floored()
                for (let dx = -searchRadius; dx <= searchRadius && !found; dx++) {
                    for (let dy = -1; dy <= 2 && !found; dy++) {
                        for (let dz = -searchRadius; dz <= searchRadius && !found; dz++) {
                            const checkPos = botPos.offset(dx, dy, dz)
                            const checkBlock = bot.blockAt(checkPos)
                            if (checkBlock?.type === expectedBlockId) {
                                console.warn(`[placeBlock] ${bot.username} Placement event timeout, but block found at ${checkPos}. Saving to memory.`)
                                newBlockPos = checkPos
                                found = true
                            }
                        }
                    }
                }
            }
            if (!found) throw e
        }
    }

    console.log(`[placeBlock] ${bot.username} Saving ${blockName} position to memory...`)
    console.log(`[placeBlock] ${bot.username} Position: ${JSON.stringify(newBlockPos)}`)
    
    addToMemory(bot, blockName, newBlockPos)
}

/**
 * Adds the position of a structure (crafting table or furnace) to the bot's memory file.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {string} structureName - The name of the structure ('crafting_table' or 'furnace').
 * @param {Vec3} position - The position of the structure to save.
 */
function addToMemory(bot, structureName, position) {
    const filePath = getMemoryPath(bot)
    const dir = path.dirname(filePath)
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
    }
    
    let positions = {}
    if (fs.existsSync(filePath)) {
        positions = JSON.parse(fs.readFileSync(filePath, 'utf8'))
    }

    positions[structureName] = toPlainPosition(position)
    console.log(`[placeBlock] ${structureName} saved`)

    fs.writeFileSync(filePath, JSON.stringify(positions, null, 2))
    console.log(`[placeBlock] File saved: ${filePath}`)
}

/**
 * ===============================================
 * ================ HELPERS ======================
 * ===============================================
 */
function getMemoryPath(bot) {
    return path.join('src', 'agents', 'memories', `${bot.username}_memory.json`)
}

function loadMemory(bot) {
    const filePath = getMemoryPath(bot)
    if (!fs.existsSync(filePath)) return {}
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

function isSamePos(pos, storedPos) {
    if (!pos || !storedPos) return false
    return pos.x === storedPos.x && pos.y === storedPos.y && pos.z === storedPos.z
}

function isPosInMemory(pos, memoryData) {
    if (!pos || !memoryData) return false
    return Object.values(memoryData).some((storedPos) => isSamePos(pos, storedPos))
}

function toPlainPosition(position) {
    return { x: position.x, y: position.y, z: position.z }
}

/**
 * Ensures a crafting table is placed and accessible.
 * 1. Checks memory for an existing table and verifies it
 * 2. If not found, crafts one (needs 4 planks in inventory) and places it
 * 
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @returns {Block} - The crafting table block.
 * @throws {Error} - If the crafting table cannot be created or placed.
 */
async function ensureStructure(bot, mcData, blockName) {
    // 1. Try to get existing structure from memory (if supported)
    if (blockName === 'crafting_table') {
        try {
            const table = getCraftingTable(bot, mcData)
            await moveToBlock(bot, table, 3)
            return table
        } catch (_e) {
            // Not found in memory or invalid, need to craft and place one
        }
    }

    // 2. Craft one if we don't have it in inventory
    if (!hasItem(bot, mcData, blockName, 1)) {
        const { craftItem } = await import('./crafting.js')
        await craftItem(bot, mcData, blockName, 1)
    }

    // 3. Place it and save to memory
    await placeBlockFull(bot, mcData, blockName)

    // 4. Return the structure if it's in memory and supported
    if (blockName === 'crafting_table') {
        return getCraftingTable(bot, mcData)
    }

    return null
}

async function ensureCraftingTable(bot, mcData) {
    return ensureStructure(bot, mcData, 'crafting_table')
}

export { placeBlockFull, ensureStructure, ensureCraftingTable }

