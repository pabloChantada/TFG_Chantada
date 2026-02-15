import { findNearestBlock, hasItem } from './inventory.js'
import { exploreRandom, moveToBlock } from './movement.js'

import { getItemNameFromBlock } from './helpers.js'

/**
 * Mines the specified block until the desired count of the corresponding item is obtained.
 * If the block is not found within the search radius, it explores the area and retries.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block to mine (e.g., "coal_ore").
 * @param {number} count - The number of items to obtain (e.g., 4 coal).
 * @param {number} searchRadius - The radius to search for the block.
 * @param {number} maxAttempts - The maximum number of attempts to find and mine the block.
 * @return {Promise<void>}
 */

async function mineBlock(bot, mcData, blockName, count, searchRadius = 32, maxAttempts = 3) {
    // Check if the block we want to mine is different from the item we want to obtain (e.g., mining "coal_ore" gives "coal")
    // If not, return the blockName as the itemName
    const itemName = getItemNameFromBlock(blockName)
    const startTime = Date.now()
    const MAX_TIME_MS = 120000 // 2 minutes
    
    let attempts = 0
    let lastOreLocation = null
    
    while (!hasItem(bot, mcData, itemName, count) && attempts < maxAttempts) {
        // Timeout
        if (Date.now() - startTime > MAX_TIME_MS) {
            throw {
                type: 'TIMEOUT',
                message: `Timeout after ${attempts} attempts`,
                reason: 'time_exceeded'
            }
        }

        // Increase search radius with each attempt to find more ores if they are not found nearby
        const currentRadius = searchRadius + (attempts * 16)
        // Search for the actual block in the world (e.g., 'stone'), not the item it drops (e.g., 'cobblestone')
        let ore = findNearestBlock(bot, mcData, blockName, currentRadius)
        
        if (ore) {
            lastOreLocation = ore.position
            try {
                await mine(bot, ore)
            } catch (e) {
                // Differentiate error types
                if (e.message.includes('No block provided')) {
                    throw { type: 'INVALID_BLOCK', reason: 'block_missing' }
                }
                attempts++
            }
        } else {
            // No ore visible
            if (attempts === 0) {
                // First attempt: explore
                await exploreRandom(bot, 50)
            } else if (attempts === 1) {
                // Second attempt: explore further
                await exploreRandom(bot, 100)
            } else {
                // Third attempt: the block probably doesn't exist
                throw {
                    type: 'ORE_NOT_FOUND',
                    searchedRadius: currentRadius,
                    reason: 'ore_exhausted_or_missing'
                }
            }
            attempts++
        }
    }
    
    // We dont have the required items after max attempts
    if (!hasItem(bot, mcData, itemName, count)) {
        throw {
            type: 'INSUFFICIENT_ITEMS',
            obtained: bot.inventory.count(mcData.itemsByName[itemName]?.id || 0),
            required: count,
            reason: 'max_attempts_reached'
        }
    }
    
    // Return final count obtained
    return bot.inventory.count(mcData.itemsByName[itemName]?.id || 0)
}

async function mine(bot, block) {
    const botLabel = bot?.name || bot?.username || 'bot'
    if (!block) throw new Error(`[ERROR] ${botLabel} No block provided for mining.`)
    
    try {
        await moveToBlock(bot, block, 4)
        // Check if the block is still there before trying to mine
        const currentBlock = bot.blockAt(block.position)
        if (!currentBlock || currentBlock.type === 0) {
            throw new Error(`Block disappeared before mining`)
        }
        
        await bot.collectBlock.collect(currentBlock)
    } catch (e) {
        console.error(`[ERROR] [${botLabel}] Failed to mine: ${e.message}`)
        throw e
    }
}

export {
    mineBlock,
    mine
}