/**
 * MultiDiscrete action space for RL agent.
 * 
 * Matches EXACTLY the action format from the metrics tracker (rl_action_tracker.js).
 * This ensures recorded demos and RL actions use the same representation.
 * 
 * Action Space: MultiDiscrete([3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5])
 * 
 *   [0] move_forward   Discrete(3)  0=still, 1=walk, 2=sprint
 *   [1] move_backward  Discrete(2)  0=still, 1=walk
 *   [2] move_lateral   Discrete(3)  0=still, 1=left, 2=right
 *   [3] move_vertical  Discrete(3)  0=still, 1=jump, 2=sneak
 *   [4] camera_yaw     Discrete(5)  0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
 *   [5] camera_pitch   Discrete(5)  0=none, 1=+15°, 2=-15°, 3=+45°, 4=-45°
 *   [6] attack         Discrete(2)  0=no, 1=yes
 *   [7] craft          Discrete(7)  0=none,1=planks,2=stick,3=crafting_table,4=wpick,5=spick,6=ipick
 *   [8] smelt          Discrete(2)  0=none, 1=iron_ingot
 *   [9] place          Discrete(4)  0=none, 1=crafting_table, 2=furnace, 3=torch
 *  [10] equip          Discrete(5)  0=none, 1=wpick, 2=spick, 3=ipick, 4=axe
 */

/** Number of discrete choices per dimension */
export const ACTION_SPACE = [3, 2, 3, 3, 5, 5, 2, 7, 2, 4, 5]

// Camera step sizes (degrees → radians)
const DEG15 = (15 * Math.PI) / 180
const DEG45 = (45 * Math.PI) / 180

// Craft recipes (index → item name)
const CRAFT_ITEMS = [null, 'oak_planks', 'stick', 'crafting_table', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe']
const SMELT_ITEMS = [null, 'iron_ingot']
const PLACE_ITEMS = [null, 'crafting_table', 'furnace', 'torch']
const EQUIP_ITEMS = [null, 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wooden_axe']
const PRE_ACTION_WAIT_MS = 180

/**
 * Apply a MultiDiscrete action vector to the bot.
 * 
 * @param {import('mineflayer').Bot} bot
 * @param {number[]} action - Array of 11 integers, one per dimension
 */
export async function applyMultiDiscreteAction(bot, action) {
    const [fwd, back, lateral, vertical, yaw, pitch, attack, craft, smelt, place, equip] = action

    // ── Release all controls first ──
    await releaseAllControls(bot)

    // ── Movement ──
    if (fwd !== 0 || back !== 0 || lateral !== 0 || vertical !== 0) {
        await waitBeforeAction()
    }

    if (fwd === 1) {
        bot.setControlState('forward', true)
    } else if (fwd === 2) {
        bot.setControlState('forward', true)
        bot.setControlState('sprint', true)
    }

    if (back === 1) {
        bot.setControlState('back', true)
    }

    if (lateral === 1) {
        bot.setControlState('left', true)
    } else if (lateral === 2) {
        bot.setControlState('right', true)
    }

    if (vertical === 1) {
        bot.setControlState('jump', true)
    } else if (vertical === 2) {
        bot.setControlState('sneak', true)
    }

    // ── Camera ──
    let yawDelta = 0
    let pitchDelta = 0

    if (yaw === 1) yawDelta = DEG15
    else if (yaw === 2) yawDelta = -DEG15
    else if (yaw === 3) yawDelta = DEG45
    else if (yaw === 4) yawDelta = -DEG45

    if (pitch === 1) pitchDelta = DEG15
    else if (pitch === 2) pitchDelta = -DEG15
    else if (pitch === 3) pitchDelta = DEG45
    else if (pitch === 4) pitchDelta = -DEG45

    if (yawDelta !== 0 || pitchDelta !== 0) {
        await waitBeforeAction()
        const newYaw = bot.entity.yaw + yawDelta
        const newPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, bot.entity.pitch + pitchDelta))
        await bot.look(newYaw, newPitch, false)
    }

    // ── Attack (mine block at cursor — non-omniscient raycast) ──
    if (attack === 1) {
        await waitBeforeAction()
        await tryMine(bot)
    }

    // ── Craft ──
    if (craft > 0 && craft < CRAFT_ITEMS.length) {
        await waitBeforeAction()
        await tryCraft(bot, CRAFT_ITEMS[craft])
    }

    // ── Smelt ──
    if (smelt > 0 && smelt < SMELT_ITEMS.length) {
        await waitBeforeAction()
        await trySmelt(bot, SMELT_ITEMS[smelt])
    }

    // ── Place ──
    if (place > 0 && place < PLACE_ITEMS.length) {
        await waitBeforeAction()
        await tryPlace(bot, PLACE_ITEMS[place])
    }

    // ── Equip ──
    if (equip > 0 && equip < EQUIP_ITEMS.length) {
        await waitBeforeAction()
        await tryEquip(bot, EQUIP_ITEMS[equip])
    }
}

function waitBeforeAction() {
    return new Promise(resolve => setTimeout(resolve, PRE_ACTION_WAIT_MS))
}

/**
 * Try to mine the block at cursor (non-omniscient raycast).
 */
async function tryMine(bot) {
    const block = bot.blockAtCursor(4.5)
    if (block && block.type !== 0 && bot.canDigBlock(block)) {
        try {
            await bot.dig(block, 'ignore')
        } catch (_) {}
    }
}

/**
 * Try to craft an item. Requires crafting table nearby for table recipes.
 */
async function tryCraft(bot, itemName) {
    try {
        const mcData = (await import('minecraft-data')).default(bot.version)
        const item = mcData.itemsByName[itemName]
        if (!item) return

        const recipes = bot.recipesFor(item.id)
        if (!recipes || recipes.length === 0) return

        const recipe = recipes[0]
        let craftingTable = null

        if (recipe.requiresTable) {
            craftingTable = bot.findBlock({
                matching: mcData.blocksByName['crafting_table']?.id,
                maxDistance: 4,
            })
            if (!craftingTable) return
        }

        await bot.craft(recipe, 1, craftingTable)
    } catch (_) {}
}

async function trySmelt(bot, _itemName) {
    try {
        const mcData = (await import('minecraft-data')).default(bot.version)
        const furnaceBlock = bot.findBlock({
            matching: mcData.blocksByName['furnace']?.id,
            maxDistance: 4,
        })
        if (!furnaceBlock) return
        // Placeholder — full furnace interaction is complex
    } catch (_) {}
}

async function tryPlace(bot, blockName) {
    try {
        const item = bot.inventory.items().find(i => i.name === blockName)
        if (!item) return
        await bot.equip(item, 'hand')
        const ref = bot.blockAtCursor(4.5)
        if (ref) await bot.placeBlock(ref, { x: 0, y: 1, z: 0 })
    } catch (_) {}
}

async function tryEquip(bot, itemName) {
    try {
        const item = bot.inventory.items().find(i => i.name === itemName)
        if (item) await bot.equip(item, 'hand')
    } catch (_) {}
}

/**
 * Release all movement/action controls.
 * @param {import('mineflayer').Bot} bot
 */
export async function releaseAllControls(bot) {
    for (const c of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
        bot.setControlState(c, false)
    }
}