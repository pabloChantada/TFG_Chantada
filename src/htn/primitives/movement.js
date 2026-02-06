import pkg from 'mineflayer-pathfinder'
const { goals } = pkg

/**
 * Moves to a specified block within a certain range. 
 * If the block is out of range, it uses the pathfinder to navigate towards it. 
 * If the pathfinder fails, it attempts to nudge forward in the direction of the block.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Block} block - The target block to move towards.
 * @param {number} range - The distance within which the bot should be to consider it has arrived (default is 3).
 * @param {MetricsCollector} metricsCollector - Optional metrics collector to track action performance.
 * @returns {Promise<void>}
 */
async function moveToBlock(bot, block, range = 3) {
    if (!block) return
    
    const dist = bot.entity.position.distanceTo(block.position)
    if (dist > range) {
        try {
            await bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, range))
        } catch (e) {
            // Fallback: explore randomly
            try {
                bot.lookAt(block.position)
                await exploreRandom(bot, 30)
            } catch (exploreError) {
                console.warn(`[moveToBlock] Failed: ${exploreError.message}`)
                throw exploreError
            }
        }
    }
}

/**
 * Explores in a random direction for a certain distance. Used as a fallback when pathfinding fails.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {number} distance - The distance to explore in a random direction (default is 30).
 * @returns {Promise<void>}
 */
async function exploreRandom(bot, distance = 30) {
    const angle = Math.random() * 2 * Math.PI
    const dx = Math.cos(angle) * distance
    const dz = Math.sin(angle) * distance
    const target = bot.entity.position.offset(dx, 0, dz)
    
    try {
        await bot.pathfinder.goto(new goals.GoalNear(target.x, target.y, target.z, 5))
    } catch (e) {
        console.warn(`[Movement.js] Bot's stuck. Disconnecting.`);
        bot.end();
    }
}

export {
    moveToBlock,
    exploreRandom
}