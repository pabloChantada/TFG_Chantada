/**
 * State extraction for the RL wood-chopping agent.
 * 
 * IMPORTANT: No omniscience. We only use information the bot can "see":
 *   - Its own position, orientation
 *   - Inventory counts
 *   - Block at cursor (raycast)
 *   - Nearby visible blocks (limited by distance + line of sight)
 * 
 * This produces a flat numeric vector for the model, plus optional screenshot path.
 */

const LOG_BLOCK_NAMES = [
    'oak_log', 'spruce_log', 'birch_log', 'jungle_log',
    'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log'
]

/**
 * Check if a block name is a log type
 * @param {string} name 
 * @returns {boolean}
 */
export function isLog(name) {
    return LOG_BLOCK_NAMES.includes(name)
}

/**
 * Get the number of logs in the bot's inventory (all wood types combined).
 * @param {import('mineflayer').Bot} bot 
 * @returns {number}
 */
export function getLogCount(bot) {
    return bot.inventory.items()
        .filter(item => isLog(item.name))
        .reduce((sum, item) => sum + item.count, 0)
}

/**
 * Extract a numeric state vector from the bot's current situation.
 * No findBlock (omniscient) — only raycast and entity position.
 * 
 * State vector layout (13 dimensions):
 *   [0]  x position (normalized)
 *   [1]  y position (normalized)
 *   [2]  z position (normalized)
 *   [3]  yaw  (normalized to [-1, 1])
 *   [4]  pitch (normalized to [-1, 1])
 *   [5]  log_count in inventory
 *   [6]  is_looking_at_log (0 or 1)
 *   [7]  looking_at_block_hardness (0 if air/nothing), 2 for logs
 *   [8]  is_on_ground (0 or 1)
 *   [9]  health
 *   [10] food_level
 *   [11] cursor_block_distance (0-4.5, 0 if no block)
 *   [12] time_of_day (0-1)
 * 
 * @param {import('mineflayer').Bot} bot 
 * @returns {{ vector: number[], meta: object }}
 */
export function extractState(bot) {
    const pos = bot.entity.position
    const yaw = bot.entity.yaw
    const pitch = bot.entity.pitch

    // Raycast: what block is the bot looking at? (NON-OMNISCIENT)
    const cursorBlock = bot.blockAtCursor(4.5)
    const lookingAtLog = cursorBlock ? isLog(cursorBlock.name) : false
    const blockHardness = cursorBlock ? (cursorBlock.hardness || 0) : 0
    const cursorDist = cursorBlock
        ? pos.distanceTo(cursorBlock.position.offset(0.5, 0.5, 0.5))
        : 0

    const logCount = getLogCount(bot)

    const vector = [
        round(pos.x / 100, 4),                     // [0] normalized x
        round(pos.y / 100, 4),                     // [1] normalized y
        round(pos.z / 100, 4),                     // [2] normalized z
        round(yaw / Math.PI, 4),                   // [3] yaw normalized [-1, 1]
        round(pitch / (Math.PI / 2), 4),           // [4] pitch normalized [-1, 1]
        logCount,                                   // [5] log count
        lookingAtLog ? 1 : 0,                       // [6] looking at log?
        round(blockHardness, 2),                    // [7] block hardness
        bot.entity.onGround ? 1 : 0,               // [8] on ground
        round((bot.health || 20) / 20, 2),         // [9] health normalized
        round((bot.food || 20) / 20, 2),           // [10] food normalized
        round(cursorDist / 4.5, 3),                // [11] cursor distance normalized
        round((bot.time?.timeOfDay || 0) / 24000, 3) // [12] time of day
    ]

    const meta = {
        position: { x: pos.x, y: pos.y, z: pos.z },
        yaw, pitch,
        logCount,
        lookingAtLog,
        cursorBlockName: cursorBlock?.name || null,
        cursorDist
    }

    return { vector, meta }
}

export const STATE_DIM = 13

function round(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals)
}