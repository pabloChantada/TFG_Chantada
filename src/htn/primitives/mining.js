import { hasItem } from './inventory.js'
import { findNearestVisibleBlock } from './blocks.js'
import { exploreRandom, exploreDown, moveToBlock } from './movement.js'

import { getItemNameFromBlock } from './helpers.js'

const WATER_BLOCK_NAME = 'water'

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

function isWaterBlock(block) {
    if (!block) return false
    return block.name === WATER_BLOCK_NAME
}

function isUnderwaterTarget(bot, block) {
    if (!block) return false
    const above = bot.blockAt(block.position.offset(0, 1, 0))
    const same = bot.blockAt(block.position)
    return isWaterBlock(same) || isWaterBlock(above)
}

function isBotSubmerged(bot) {
    const feet = bot.blockAt(bot.entity.position.floored())
    const head = bot.blockAt(bot.entity.position.offset(0, 1, 0).floored())
    return isWaterBlock(feet) || isWaterBlock(head)
}

async function recoverFromWater(bot, timeout = 6000) {
    if (!isBotSubmerged(bot)) return true

    const start = Date.now()
    bot.setControlState('jump', true)

    try {
        while (Date.now() - start < timeout) {
            if (!isBotSubmerged(bot)) return true
            await sleep(150)
        }
        return !isBotSubmerged(bot)
    } finally {
        bot.setControlState('jump', false)
    }
}

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

async function mineBlock(bot, mcData, blockName, count, searchRadius = 32, maxAttempts = 5) {
    // Check if the block we want to mine is different from the item we want to obtain (e.g., mining "coal_ore" gives "coal")
    // If not, return the blockName as the itemName
    const itemName = getItemNameFromBlock(blockName)
    const startTime = Date.now()
    const MAX_TIME_MS = 180000 // 3 minutes (non-omniscient exploration needs more time)
    
    let attempts = 0
    let notFoundCount = 0
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
        // Only find blocks with exposed faces (visible to the bot, not buried underground)
        let ore = findNearestVisibleBlock(bot, mcData, blockName, currentRadius)
        
        if (ore) {
            if (isUnderwaterTarget(bot, ore)) {
                console.warn(`[mineBlock] Skipping ${blockName} at ${ore.position} (underwater target)`)
                await exploreRandom(bot, 24)
                attempts++
                continue
            }

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
            // No visible ore found — explore to expose new blocks
            console.log(`[mineBlock] No visible ${blockName} found (search ${notFoundCount + 1}, radius ${currentRadius})`)
            
            if (notFoundCount === 0) {
                // First: explore on surface, might find caves/ravines with exposed ores
                await exploreRandom(bot, 50)
            } else if (notFoundCount === 1) {
                // Second: dig underground where ores are more common, 
                // the tunnel we dig exposes ores in the walls
                await exploreDown(bot)
            } else if (notFoundCount === 2) {
                // Third: explore horizontally underground to expose more blocks
                await exploreRandom(bot, 40)
            } else {
                // Exhausted exploration strategies
                throw {
                    type: 'ORE_NOT_FOUND',
                    searchedRadius: currentRadius,
                    reason: 'ore_exhausted_or_missing'
                }
            }
            notFoundCount++
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
        const escapedWater = await recoverFromWater(bot)
        if (!escapedWater) {
            throw new Error(`Drowning risk detected before mining`)
        }

        await moveToBlock(bot, block, 4)

        const escapedWaterNearTarget = await recoverFromWater(bot)
        if (!escapedWaterNearTarget) {
            throw new Error(`Drowning risk detected near mining target`)
        }

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