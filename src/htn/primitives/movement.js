import pkg from 'mineflayer-pathfinder'
const { goals } = pkg

/**
 * Moves to a specified block within a certain range. 
 * If the block is out of range, it uses the pathfinder to navigate towards it. 
 * Includes timeout to prevent stuck situations.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Block} block - The target block to move towards.
 * @param {number} range - The distance within which the bot should be to consider it has arrived (default is 3).
 * @param {number} timeout - Maximum time to attempt movement in ms (default 15000ms).
 * @returns {Promise<void>}
 */
async function moveToBlock(bot, block, range = 3, timeout = 15000) {
    if (!block) return
    
    const dist = bot.entity.position.distanceTo(block.position)
    if (dist <= range) return // Already in range
    
    const startTime = Date.now()
    const startPos = bot.entity.position.clone()
    
    try {
        // Create a promise that rejects on timeout
        const movePromise = bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, range))
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Movement timeout')), timeout)
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

export {
    moveToBlock,
    exploreRandom
}