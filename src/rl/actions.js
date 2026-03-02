/**
 * Discrete action space for wood chopping RL agent.
 * Maps action indices to bot control commands.
 * 
 * Aligned with control_tracker.js control states:
 *   forward, back, left, right, jump, sprint, sneak, mine, place
 * 
 * We use a simplified discrete action set for the wood task.
 */

export const ACTION_NAMES = [
    'noop',             // 0: do nothing
    'forward',          // 1: move forward
    'back',             // 2: move backward
    'left',             // 3: strafe left
    'right',            // 4: strafe right
    'jump',             // 5: jump
    'mine',             // 6: start mining (dig)
    'forward_mine',     // 7: move forward + mine
    'look_up',          // 8: pitch camera up
    'look_down',        // 9: pitch camera down
    'turn_left',        // 10: yaw camera left
    'turn_right',       // 11: yaw camera right
    'forward_jump',     // 12: move forward + jump (for repositioning)
    'sprint_forward',   // 13: sprint forward (for approaching trees)
    'stop_all',         // 14: release all controls
]

export const NUM_ACTIONS = ACTION_NAMES.length

// Camera rotation increments (radians)
const YAW_STEP = Math.PI / 8      // 22.5 degrees
const PITCH_STEP = Math.PI / 12   // 15 degrees

/**
 * Apply a discrete action to the bot.
 * Returns a cleanup function to release controls after the action tick.
 * 
 * @param {import('mineflayer').Bot} bot 
 * @param {number} actionIndex 
 * @returns {Promise<void>}
 */
export async function applyAction(bot, actionIndex) {
    const name = ACTION_NAMES[actionIndex]
    if (!name) throw new Error(`Invalid action index: ${actionIndex}`)

    // First, release all movement controls to avoid stacking
    await releaseAllControls(bot)

    switch (name) {
        case 'noop':
            break

        case 'forward':
            bot.setControlState('forward', true)
            break

        case 'back':
            bot.setControlState('back', true)
            break

        case 'left':
            bot.setControlState('left', true)
            break

        case 'right':
            bot.setControlState('right', true)
            break

        case 'jump':
            bot.setControlState('jump', true)
            break

        case 'mine':
            await startMining(bot)
            break

        case 'forward_mine':
            bot.setControlState('forward', true)
            await startMining(bot)
            break

        case 'look_up':
            await bot.look(bot.entity.yaw, bot.entity.pitch - PITCH_STEP, false)
            break

        case 'look_down':
            await bot.look(bot.entity.yaw, bot.entity.pitch + PITCH_STEP, false)
            break

        case 'turn_left':
            await bot.look(bot.entity.yaw + YAW_STEP, bot.entity.pitch, false)
            break

        case 'turn_right':
            await bot.look(bot.entity.yaw - YAW_STEP, bot.entity.pitch, false)
            break

        case 'forward_jump':
            bot.setControlState('forward', true)
            bot.setControlState('jump', true)
            break

        case 'sprint_forward':
            bot.setControlState('forward', true)
            bot.setControlState('sprint', true)
            break

        case 'stop_all':
            // Already released above
            break
    }
}

/**
 * Try to mine the block the bot is currently looking at.
 * Non-omniscient: uses bot.blockAtCursor (raycast from camera).
 * @param {import('mineflayer').Bot} bot 
 */
async function startMining(bot) {
    const block = bot.blockAtCursor(4.5) // max reach distance
    if (block && block.type !== 0 && bot.canDigBlock(block)) {
        try {
            await bot.dig(block, 'ignore')
        } catch (_e) {
            // Block may have been destroyed or moved out of range
        }
    }
}

/**
 * Release all movement/action controls.
 * @param {import('mineflayer').Bot} bot 
 */
export async function releaseAllControls(bot) {
    const controls = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']
    for (const c of controls) {
        bot.setControlState(c, false)
    }
}