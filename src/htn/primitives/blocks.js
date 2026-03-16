/** 
 * Gets the block ID for a given block name using the minecraft data.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block (e.g., "coal_ore").
 * @returns {number|null} - The block ID if found, or null if not found.
 * Note: This function is used in mining tasks to find ores in the world.
 */
function getBlockId(mcData, blockName) {
    return mcData.blocksByName[blockName]?.id
}

// Approximate player-like field of view (degrees)
const HORIZONTAL_FOV_DEG = 90
const VERTICAL_FOV_DEG = 60

function normalizeAngleRad(angle) {
    let a = angle
    while (a > Math.PI) a -= 2 * Math.PI
    while (a < -Math.PI) a += 2 * Math.PI
    return a
}

/**
 * Finds the nearest block of a given type within a specified distance.
 * WARNING: This is an omniscient search — it finds blocks even if buried underground.
 * DEPRECATED: Use findNearestVisibleBlock for realistic, non-omniscient search.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block to find (e.g., "coal_ore").
 * @param {number} maxDistance - The maximum distance to search for the block (default is 32).
 * @returns {Block|null} - The nearest block if found, or null if not found.
 */
function findNearestBlock(bot, mcData, blockName, maxDistance = 32) {
    const blockId = getBlockId(mcData, blockName)
    if (!blockId) return null
    return bot.findBlock({ matching: blockId, maxDistance })
}

/**
 * Checks if a block has at least one exposed face (adjacent to air or non-solid block).
 * A block with an exposed face is "visible" — a player could actually see it in-game.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Vec3} blockPos - The position of the block to check.
 * @returns {boolean} - True if the block has at least one exposed face.
 */
function isBlockExposed(bot, blockPos) {
    // Check the 6 adjacent blocks (up, down, north, south, east, west)
    const offsets = [
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
    ] 
    for (const [dx, dy, dz] of offsets) {
        // Check neightbours blocks
        const neighbor = bot.blockAt(blockPos.offset(dx, dy, dz))
        // If any is air or non-solid, the block is exposed
        if (!neighbor || neighbor.boundingBox === 'empty') {
            return true
        }
    }
    return false
}

/**
 * Checks if the bot has line-of-sight to a block position.
 * Uses raycast from the bot's eye position toward the block — the same non-omniscient
 * approach as the RL agent's bot.blockAtCursor().
 * 
 * The bot can only "see" a block if there is NO opaque block between its eyes and the target.
 * This prevents finding blocks in unvisited caves, behind walls, etc.
 * 
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Vec3} blockPos - The block position to check visibility for.
 * @returns {boolean} - True if the bot has clear line-of-sight to the block.
 */
function hasLineOfSight(bot, blockPos) {
    const eyePos = bot.entity.position.offset(0, 1.62, 0) // player eye height
    const blockCenter = blockPos.offset(0.5, 0.5, 0.5)
    const direction = blockCenter.minus(eyePos)
    const distance = direction.norm()

    // Very close blocks are always visible (same block or adjacent)
    if (distance < 1.5) return true
    // Beyond reasonable visual range, skip the expensive raycast
    if (distance > 64) return false

    const normalized = direction.scaled(1 / distance)

    try {
        // bot.world.raycast — same engine that powers bot.blockAtCursor()
        const hit = bot.world.raycast(eyePos, normalized, distance + 1)
        if (!hit) return false

        // The raycast result can be a block or {position: Vec3}
        const hitPos = hit.position || hit
        return Math.floor(hitPos.x) === Math.floor(blockPos.x) &&
               Math.floor(hitPos.y) === Math.floor(blockPos.y) &&
               Math.floor(hitPos.z) === Math.floor(blockPos.z)
    } catch (_e) {
        return false
    }
}

/**
 * Checks if a block lies inside the bot's camera vision cone.
 *
 * We model a realistic player-like FOV using current yaw/pitch:
 * - Horizontal cone: ±45° (90° total)
 * - Vertical cone:   ±30° (60° total)
 *
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Vec3} blockPos - The block position to check.
 * @param {number} horizontalFovDeg - Horizontal FOV in degrees.
 * @param {number} verticalFovDeg - Vertical FOV in degrees.
 * @returns {boolean} - True if the block is inside the current vision cone.
 */
function isInVisionCone(
    bot,
    blockPos,
    horizontalFovDeg = HORIZONTAL_FOV_DEG,
    verticalFovDeg = VERTICAL_FOV_DEG
) {
    const eyePos = bot.entity.position.offset(0, 1.62, 0)
    const blockCenter = blockPos.offset(0.5, 0.5, 0.5)
    const direction = blockCenter.minus(eyePos)
    const distance = direction.norm()

    // Very close blocks are considered visible regardless of camera cone
    if (distance < 1.5) return true

    // Target camera angles from eye -> block center
    const targetYaw = Math.atan2(-direction.x, -direction.z)
    const targetPitch = Math.asin(Math.max(-1, Math.min(1, direction.y / distance)))

    const yawDiff = Math.abs(normalizeAngleRad(targetYaw - bot.entity.yaw))
    const pitchDiff = Math.abs(targetPitch - bot.entity.pitch)

    const maxYaw = (horizontalFovDeg * Math.PI / 180) / 2
    const maxPitch = (verticalFovDeg * Math.PI / 180) / 2

    return yawDiff <= maxYaw && pitchDiff <= maxPitch
}

/**
 * Finds the nearest block that the bot can actually SEE (non-omniscient).
 * 
 * Three-stage filter:
 *   1. isBlockExposed — block has at least one face adjacent to air (cheap check)
 *   2. isInVisionCone — block must be inside camera FOV cone (player-like perception)
 *   3. hasLineOfSight — unobstructed raycast from bot's eyes to the block (same as RL)
 * 
 * This means the bot CANNOT find blocks in:
 *   - Unvisited caves (wall between bot and block)
 *   - Behind hills or structures
 *   - Buried underground with no exposure
 * 
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} blockName - The name of the block to find (e.g., "iron_ore").
 * @param {number} maxDistance - Maximum search distance (default 32).
 * @returns {Block|null} - The nearest truly visible block, or null if none found.
 */
function findNearestVisibleBlock(bot, mcData, blockName, maxDistance = 32) {
    const blockId = getBlockId(mcData, blockName)
    if (!blockId) return null

    const positions = bot.findBlocks({
        matching: blockId,
        maxDistance,
        count: 256
    })

    for (const pos of positions) {
        // Stage 1: cheap check — does the block have an exposed face?
        if (!isBlockExposed(bot, pos)) continue
        // Stage 2: block must be inside current camera vision cone
        if (!isInVisionCone(bot, pos)) continue
        // Stage 3: expensive check — can the bot actually see it from here?
        if (!hasLineOfSight(bot, pos)) continue

        return bot.blockAt(pos)
    }

    return null
}

export {
    getBlockId,
    findNearestBlock,
    findNearestVisibleBlock,
    isBlockExposed,
    hasLineOfSight,
    isInVisionCone
}
