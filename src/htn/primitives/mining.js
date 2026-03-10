import { hasItem } from './inventory.js'
import { findNearestVisibleBlock } from './blocks.js'
import { exploreRandom, exploreDown, moveToBlock } from './movement.js'

import { getItemNameFromBlock } from './helpers.js'

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Checks if the bot is underwater by monitoring oxygen level.
 * If oxygen level is below maximum (20), the bot is submerged.
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {boolean} - True if the bot is underwater
 */
function isBotSubmerged(bot) {
    return bot.oxygenLevel < 20
}

/**
 * Checks if a target block is underwater by attempting to approach it
 * and monitoring if oxygen level starts decreasing.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Block} block - The target block to check
 * @returns {boolean} - True if approaching the block causes oxygen loss
 */
function isUnderwaterTarget(bot, block) {
    if (!block) return false
    // If bot is already underwater, the target is likely underwater too
    return isBotSubmerged(bot)
}

/**
 * Attempts to recover from underwater situation by surfacing.
 * Monitors oxygen level and jumps until oxygen returns to maximum (20).
 * @param {Bot} bot - The mineflayer bot instance
 * @param {number} timeout - Maximum time to attempt recovery (default 6000ms)
 * @returns {Promise<boolean>} - True if successfully surfaced, false if timeout
 */
async function recoverFromWater(bot, timeout = 6000) {
    if (!isBotSubmerged(bot)) return true

    const start = Date.now()
    console.log(`[recoverFromWater] Oxygen level: ${bot.oxygenLevel}/20. Attempting to surface...`)
    
    bot.setControlState('jump', true)

    try {
        while (Date.now() - start < timeout) {
            // Check if we've reached the surface (oxygen back to maximum)
            if (bot.oxygenLevel === 20) {
                console.log(`[recoverFromWater] Successfully surfaced!`)
                return true
            }
            await sleep(150)
        }
        console.warn(`[recoverFromWater] Timeout - still underwater (oxygen: ${bot.oxygenLevel}/20)`)
        return false
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
    const MAX_TIME_MS = 180000 // 3 minutes (exploration needs more time)
    
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
                console.warn(`[mineBlock] Underwater detected (oxygen: ${bot.oxygenLevel}/20). Skipping ${blockName} and exploring elsewhere...`)
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

            // Cycle strategies across attempts instead of failing after 3 misses.
            // This gives maxAttempts real value and improves recovery in difficult terrain.
            const strategy = notFoundCount % 4
            if (strategy === 0) {
                //zSurface exploration, likely to find cave openings
                await exploreRandom(bot, 50)
            } else if (strategy === 1) {
                // Descend underground where stone/ores are more likely
                await exploreDown(bot)
            } else if (strategy === 2) {
                // Underground lateral exploration
                await exploreRandom(bot, 40)
            } else {
                // Short local reposition to break pathfinder deadlocks/chunk boundaries
                await exploreRandom(bot, 20)
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