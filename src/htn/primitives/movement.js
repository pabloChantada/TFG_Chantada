import pkg from 'mineflayer-pathfinder'
const { goals } = pkg

/**
 * Moves to a specified block within a certain range. 
 * If the block is out of range, it uses the pathfinder to navigate towards it. 
 * Includes timeout to prevent stuck situations.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Block} block - The target block to move towards.
 * @param {number} range - The distance within which the bot should be to consider it has arrived (default is 3).
 * @param {number|null} timeout - Maximum time to attempt movement in ms (null = auto by distance).
 * @returns {Promise<void>}
 */
async function moveToBlock(bot, block, range = 3, timeout = null) {
    if (!block) return
    
    const dist = bot.entity.position.distanceTo(block.position)
    if (dist <= range) return // Already in range

    const effectiveTimeout = timeout ?? Math.min(60000, Math.max(15000, Math.ceil(dist * 1000)))
    
    const startTime = Date.now()
    const startPos = bot.entity.position.clone()
    
    try {
        // Create a promise that rejects on timeout
        const movePromise = bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, range))
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Movement timeout')), effectiveTimeout)
        )
        
        await Promise.race([movePromise, timeoutPromise])
    } catch (e) {
        const movedDistance = startPos.distanceTo(bot.entity.position)
        const elapsed = Date.now() - startTime
        
        // If we moved some distance, might be partial success
        if (movedDistance > 2) {
            console.warn(`[moveToBlock] Partial move: ${movedDistance.toFixed(1)} blocks in ${(elapsed/1000).toFixed(1)}s - ${e.message}`)
            // Don't throw, partial progress is acceptable
            return
        }
        
        console.warn(`[moveToBlock] Failed after ${(elapsed/1000).toFixed(1)}s: ${e.message}`)
        throw new Error(`Movement failed: ${e.message}`)
    }
}

/**
 * Explores in a random direction for a certain distance. Used as a fallback when pathfinding fails.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {number} distance - The distance to explore in a random direction (default is 30).
 * @param {number} timeout - Maximum time for exploration in ms (default 10000ms).
 * @returns {Promise<void>}
 */
async function exploreRandom(bot, distance = 30, timeout = 10000) {
    const angle = Math.random() * 2 * Math.PI
    const dx = Math.cos(angle) * distance
    const dz = Math.sin(angle) * distance
    const target = bot.entity.position.offset(dx, 0, dz)
    const startPos = bot.entity.position.clone()
    
    try {
        const movePromise = bot.pathfinder.goto(new goals.GoalNear(target.x, target.y, target.z, 5))
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Exploration timeout')), timeout)
        )
        
        await Promise.race([movePromise, timeoutPromise])
    } catch (e) {
        const movedDistance = startPos.distanceTo(bot.entity.position)
        
        // If we moved at least a bit, consider it partial success
        if (movedDistance > 3) {
            console.warn(`[exploreRandom] Partial exploration: ${movedDistance.toFixed(1)} blocks`)
            return
        }
        
        console.warn(`[exploreRandom] Exploration failed: ${e.message}`)
        throw new Error(`Exploration failed: ${e.message}`)
    }
}

/**
 * Explores underground by navigating to a lower Y level.
 * With canDig=true on the pathfinder, the bot will mine through blocks to create a path,
 * which naturally exposes ores in the walls of the tunnels it digs.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {number} targetY - Target Y level to reach (default 16, good for iron/coal).
 * @param {number} timeout - Maximum time in ms (default 30000).
 * @returns {Promise<void>}
 */
async function exploreDown(bot, targetY = 16, timeout = 30000) {
    const currentY = Math.floor(bot.entity.position.y)
    if (currentY <= targetY + 5) {
        // Already near the target depth, explore horizontally to expose more blocks
        await exploreRandom(bot, 30)
        return
    }

    console.log(`[exploreDown] Digging from Y=${currentY} to Y=${targetY}...`)
    const startPos = bot.entity.position.clone()

    try {
        const movePromise = bot.pathfinder.goto(new goals.GoalY(targetY))
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Underground exploration timeout')), timeout)
        )

        await Promise.race([movePromise, timeoutPromise])
        console.log(`[exploreDown] Reached Y=${Math.floor(bot.entity.position.y)}`)
    } catch (e) {
        const movedDown = startPos.y - bot.entity.position.y
        if (movedDown > 5) {
            console.warn(`[exploreDown] Partial descent: ${movedDown.toFixed(1)} blocks down (now Y=${Math.floor(bot.entity.position.y)})`)
            return
        }
        console.warn(`[exploreDown] Failed to dig down: ${e.message}`)
        // Fall back to horizontal exploration
        await exploreRandom(bot, 20)
    }
}

export {
    moveToBlock,
    exploreRandom,
    exploreDown
}