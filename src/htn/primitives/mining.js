import pkg from 'mineflayer-pathfinder'
const { goals } = pkg

import { hasItem } from './inventory.js'
import { findNearestVisibleBlock } from './blocks.js'
import { exploreRandom, exploreDown, moveToBlock } from './movement.js'

import { getItemNameFromBlock } from './helpers.js'

const MIN_MINING_DIST = 1.5 // If closer than this, back up before mining

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Checks if the bot is underwater by testing the block at eye level.
 * oxygenLevel starts at 0 on spawn and syncs after a few ticks, so it
 * produces false positives on land — block-based check is reliable.
 * @param {Bot} bot - The mineflayer bot instance
 * @returns {boolean} - True if the bot's eyes are submerged in water
 */
function isBotSubmerged(bot) {
    const eyeBlock = bot.blockAt(bot.entity.position.offset(0, 1.62, 0))
    return eyeBlock != null && (eyeBlock.name === 'water' || eyeBlock.name === 'flowing_water')
}

/**
 * Checks if a target block is actually submerged — water at or above the block position.
 * Does NOT check if the bot itself is in water, only the target.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {Block} block - The target block to check
 * @returns {boolean} - True if the block itself is in/under water
 */
function isUnderwaterTarget(bot, block) {
    if (!block) return false
    const isWater = (b) => b && (b.name === 'water' || b.name === 'flowing_water')
    const blockAtPos = bot.blockAt(block.position)
    const blockAbove = bot.blockAt(block.position.offset(0, 1, 0))
    return isWater(blockAtPos) || isWater(blockAbove)
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

async function mineBlock(bot, mcData, blockName, count, searchRadius = 32, maxAttempts = 5, opts = {}) {
    const { useFovCone = true, allowUnderground = true } = opts
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
            throw new Error(`Timeout after ${attempts} attempts`)
        }

        // Increase search radius with each attempt to find more ores if they are not found nearby
        const currentRadius = searchRadius + (attempts * 16)
        // Only find blocks with exposed faces (visible to the bot, not buried underground)

        let ore = findNearestVisibleBlock(bot, mcData, blockName, currentRadius, useFovCone)

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
                    throw new Error('Invalid block: block missing')
                }
                attempts++
            }
        } else {
            // No visible ore found — explore to expose new blocks
            console.log(`[mineBlock] No visible ${blockName} found (search ${notFoundCount + 1}, radius ${currentRadius})`)
            const shouldInterruptExplore = () => {
                const visible = findNearestVisibleBlock(bot, mcData, blockName, currentRadius, useFovCone)
                return visible != null
            }

            if (!allowUnderground) {
                // Wood collection should stay on surface.
                // Reposition around the area without descending underground.
                const distances = [50, 35, 20]
                await exploreRandom(bot, distances[notFoundCount % distances.length], 10000, {
                    stopWhen: shouldInterruptExplore,
                    checkInterval: 500
                })
            } else {
                // Cycle strategies across attempts instead of failing after 3 misses.
                // This gives maxAttempts real value and improves recovery in difficult terrain.
                const strategy = notFoundCount % 4
                if (strategy === 0) {
                    // Surface exploration, likely to find cave openings
                    await exploreRandom(bot, 50, 10000, {
                        stopWhen: shouldInterruptExplore,
                        checkInterval: 500
                    })
                } else if (strategy === 1) {
                    // Descend underground where stone/ores are more likely
                    await exploreDown(bot)
                } else if (strategy === 2) {
                    // Underground lateral exploration
                    await exploreRandom(bot, 40, 10000, {
                        stopWhen: shouldInterruptExplore,
                        checkInterval: 500
                    })
                } else {
                    // Short local reposition to break pathfinder deadlocks/chunk boundaries
                    await exploreRandom(bot, 20, 10000, {
                        stopWhen: shouldInterruptExplore,
                        checkInterval: 500
                    })
                }
            }

            notFoundCount++
            attempts++
        }
    }
    
    // We dont have the required items after max attempts
    if (!hasItem(bot, mcData, itemName, count)) {
        const obtained = bot.inventory.count(mcData.itemsByName[itemName]?.id || 0)
        throw new Error(`Max attempts reached: got ${obtained}/${count} ${itemName}`)
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

        // If too close HORIZONTALLY, collectBlock's internal pathfinder (GoalLookAtBlock) gets
        // confused and loops doing movement instead of mining — back up to a safe distance.
        // Use horizontal-only distance: blocks directly above don't need a horizontal backup.
        const blockCenter = block.position.offset(0.5, 0, 0.5)
        const hDist = Math.sqrt(
            (bot.entity.position.x - blockCenter.x) ** 2 +
            (bot.entity.position.z - blockCenter.z) ** 2
        )
        if (hDist < MIN_MINING_DIST) {
            console.warn(`[mine] ${botLabel} Too close horizontally (${hDist.toFixed(2)} blocks), backing up...`)
            const dir = bot.entity.position.minus(blockCenter)
            const dirLen = dir.norm()
            const safeDir = dirLen > 0.01 ? dir.scaled(1 / dirLen) : { x: 1, y: 0, z: 0 }
            const backupX = blockCenter.x + safeDir.x * 2.5
            const backupZ = blockCenter.z + safeDir.z * 2.5
            try {
                const backupTimer = setTimeout(() => {
                    try { bot.pathfinder.stop() } catch (_) {}
                }, 5000)
                const backupPromise = bot.pathfinder.goto(new goals.GoalNear(backupX, block.position.y, backupZ, 0.5))
                const backupTimeoutPromise = new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('Backup movement timeout')), 5000)
                )
                try {
                    await Promise.race([backupPromise, backupTimeoutPromise])
                } finally {
                    clearTimeout(backupTimer)
                }
            } catch (e) {
                console.warn(`[mine] ${botLabel} Backup failed: ${e.message}`)
            }
        }

        const escapedWaterNearTarget = await recoverFromWater(bot)
        if (!escapedWaterNearTarget) {
            throw new Error(`Drowning risk detected near mining target`)
        }

        // Check if the block is still there before trying to mine
        const currentBlock = bot.blockAt(block.position)
        if (!currentBlock || currentBlock.type === 0) {
            throw new Error(`Block disappeared before mining`)
        }


        const minedPos = currentBlock.position.clone()
        const minedName = currentBlock.name
        // Timeout for collectBlock to prevent pathfinder hanging forever
        const collectTimeout = 30000
        const collectPromise = bot.collectBlock.collect(currentBlock)
        const collectTimer = setTimeout(() => {
            try { bot.pathfinder.stop() } catch (_) {}
        }, collectTimeout)
        const collectTimeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('collectBlock timeout')), collectTimeout)
        )
        try {
            await Promise.race([collectPromise, collectTimeoutPromise])
        } finally {
            clearTimeout(collectTimer)
        }
        // Mine the rest of the trunk straight up (non-omniscient: adjacent known positions)
        await mineTreeTrunk(bot, minedName, minedPos)
    } catch (e) {
        console.error(`[ERROR] [${botLabel}] Failed to mine: ${e.message}`)
        throw e
    }
}

/**
 * After mining the base of a tree, collects each log straight up the trunk
 * at (x, y+1, z), (x, y+2, z) ... until the block type changes.
 * Uses collectBlock.collect directly — no moveToBlock, no backup logic —
 * because the bot is already standing next to these blocks.
 * @param {Bot} bot - The mineflayer bot instance
 * @param {string} woodType - Log block name (e.g. 'oak_log')
 * @param {Object} basePos - Vec3 position of the log that was just mined
 */
async function mineTreeTrunk(bot, woodType, basePos) {
    // Check there is at least one log above before moving
    const firstAbove = bot.blockAt(basePos.offset(0, 1, 0))
    if (!firstAbove || firstAbove.name !== woodType) return

    // Step under the trunk column so the upward chopping looks natural
    try {
        const trunkTimer = setTimeout(() => {
            try { bot.pathfinder.stop() } catch (_) {}
        }, 5000)
        const trunkPromise = bot.pathfinder.goto(new goals.GoalNear(basePos.x, basePos.y, basePos.z, 0))
        const trunkTimeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Trunk positioning timeout')), 5000)
        )
        try {
            await Promise.race([trunkPromise, trunkTimeoutPromise])
        } finally {
            clearTimeout(trunkTimer)
        }
    } catch (_e) { /* continue even if positioning is imperfect */ }

    let pos = basePos.offset(0, 1, 0)
    while (true) {
        const block = bot.blockAt(pos)
        if (!block || block.name !== woodType) break
        try {
    
            await bot.lookAt(block.position.offset(0.5, 0.5, 0.5))
            await bot.dig(block)

            await bot.waitForTicks(2)
        } catch (_e) {
            break
        }
        pos = pos.offset(0, 1, 0)
    }
}

export {
    mineBlock,
    mineTreeTrunk,
    mine
}