// Imports for Primitives (Low Level)
import { mineBlock } from './primitives/mining.js'
import { exploreRandom, moveToBlock } from './primitives/movement.js'
import { hasItem, getItemId } from './primitives/inventory.js'
import { 
    chop, 
    obtainWoodType, 
    obtainPlankType 
} from './primitives/wood.js'

// Imports for Composite Tasks (High Level)
import { craftItem } from './tasks/crafting.js'
import { ensureCraftingTable, placeBlockFull } from './tasks/block_placement.js'
import { smeltItem } from './tasks/smelting.js'

/**
 * =========================================================
 * ERROR HANDLING AND REPLANNING SYSTEM
 * =========================================================
 */

/**
 * Executes a task with retry and recovery logic.
 * We define a recovery task, applying real recursion in replanning, not just an alternative step.
 * @param {Bot} bot - Bot instance
 * @param {string} taskName - Name for logs
 * @param {Function} conditionFn - Function that returns true if task is done (post-condition)
 * @param {Function} actionFn - Main action to execute
 * @param {Function} recoveryFn - (Optional) Action to execute if actionFn fails (e.g., craft new pickaxe)
 */
async function runSmartTask(bot, taskName, conditionFn, actionFn, recoveryFn = null) {
    if (await conditionFn()) {
        console.log(`[SKIP] [${bot.name}] ${taskName} completed. Skipping...`)
        return
    }

    bot.chat(`[START] [${bot.name}] ${taskName}`)
    let attempts = 0
    const maxAttempts = 5

    while (!await conditionFn() && attempts < maxAttempts) {
        try {
            await actionFn()
            await bot.waitForTicks(10)
        } catch (err) {
            console.error(`[ERROR] [${bot.name}] ${taskName}: ${err.message}`)
            bot.chat(`[FAIL] [${bot.name}] ${taskName} - Try ${attempts + 1}/${maxAttempts}`)

            // Replan
            if (recoveryFn) {
                console.log(`[REPLAN] Executing recovery for ${taskName}...`)
                try {
                    await recoveryFn(err)
                    console.log(`[REPLAN] Recovery finished. Retrying main task.`)
                    continue 
                } catch (recoveryErr) {
                    console.error(`[FATAL] Recovery failed: ${recoveryErr.message}`)
                }
            }
            
            // If no recovery or it failed, increment attempts and explore to get unstuck
            attempts++
            await exploreRandom(bot, 10) 
        }
    }

    if (!await conditionFn()) {
        throw new Error(`[ABORT] [${bot.name}] Could not complete ${taskName} after ${maxAttempts} attempts.`)
    }
    console.log(`[COMPLETED] [${bot.name}] ${taskName}`)
}

/**
 * =========================================================
 * TOOL LOGIC (REAL REPLANNING)
 * =========================================================
 */

/**
 * Ensures the bot has the specified item equipped, with internal logic to recover if it's missing (e.g., craft a new one).
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {string} itemName - The name of the item to ensure is equipped (e.g., "wooden_pickaxe").
 * @param {Function} materialSourceFn - (Optional) An async function that provides necessary materials if crafting is needed (e.g., mining stone for a new pickaxe).
 * @returns {Promise<void>}
 */
async function ensureEquipped(bot, mcData, itemName, materialSourceFn = null) {
    const item = bot.inventory.findInventoryItem(getItemId(mcData, itemName), null)
    
    if (item) {
        await bot.equip(item, 'hand')
        return
    }

    console.log(`[REPLAN] I don't have ${itemName}. Crafting a new one...`)
    
    // Prerequisite: To craft tools we need wood/sticks and a crafting table
    // This is a mini-dependency chain
    if (!hasItem(bot, mcData, 'stick', 2)) {
        await ensureWoodAndPlanks(bot, mcData, 2) // We need wood for sticks
        await craftItem(bot, mcData, 'stick', 1) // Craft sticks
    }

    // Ensure nearby crafting table
    await ensureCraftingTable(bot, mcData)

    // If we need specific materials (e.g., cobblestone for stone pickaxe)
    if (materialSourceFn) {
        await materialSourceFn()
    }

    await craftItem(bot, mcData, itemName, 1)
    
    // Equip the new item
    const newItem = bot.inventory.findInventoryItem(getItemId(mcData, itemName), null)
    if (newItem) await bot.equip(newItem, 'hand')
    else throw new Error(`Could not equip ${itemName} even after crafting it.`)
}

// Specific recovery helpers
async function ensureWoodAndPlanks(bot, mcData, amountLogs = 3) {
    // 1. Get logs
    const woodType = await obtainWoodType(bot, mcData)
    if (!hasItem(bot, mcData, woodType, 1)) {
        await chop(bot, mcData, amountLogs)
    }
    // 2. Convert to planks
    const plankType = await obtainPlankType(bot, mcData, woodType)
    if (!hasItem(bot, mcData, plankType, 4)) {
        await craftItem(bot, mcData, plankType, 1) // 1 log = 4 planks
    }
}

/**
 * =========================================================
 * PROGRESSION PHASES
 * =========================================================
 */

async function phaseWood(bot, mcData) {
    const woodType = await obtainWoodType(bot, mcData)
    const plankType = await obtainPlankType(bot, mcData, woodType)

    // Task 1: Collect wood
    // Doesn't need complex recovery, if chop fails it's usually because no trees are found -> exploreRandom
    await runSmartTask(
        bot, 
        'Collect Wood',
        async () => hasItem(bot, mcData, woodType, 5), // Condition
        async () => chop(bot, mcData, 5)           // Action
    )

    // Task 2: Craft planks
    await runSmartTask(
        bot, 'Craft Planks',
        async () => hasItem(bot, mcData, plankType, 12),
        async () => craftItem(bot, mcData, plankType, 4)
    )

    // Task 3: Build crafting table
    await runSmartTask(
        bot, 'Place Crafting Table',
        async () => hasItem(bot, mcData, 'crafting_table', 1) || bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 10 }),
        async () => ensureCraftingTable(bot, mcData) // This already handles crafting and placement
    )

    // Task 4: Craft wooden pickaxe
    // Recovery: If it fails (e.g., no sticks), the ensureEquipped function has internal logic,
    // but we define it explicitly as a task here
    await runSmartTask(
        bot, 'Craft Wooden Pickaxe',
        async () => hasItem(bot, mcData, 'wooden_pickaxe'),
        async () => {
            if (!hasItem(bot, mcData, 'stick', 2)) await craftItem(bot, mcData, 'stick', 1)
            await craftItem(bot, mcData, 'wooden_pickaxe', 1)
        }
    )
}

async function phaseStone(bot, mcData) {
    // Task: Get stone (Cobblestone)
    // REPLANNING: If mineBlock fails, we assume it might be because of missing pickaxe.
    await runSmartTask(
        bot,
        'Mine Stone',
        async () => hasItem(bot, mcData, 'cobblestone', 14), // 3 for pickaxe, 8 for furnace, 3 extra
        async () => {
            // Before mining, ensure tool
            await ensureEquipped(bot, mcData, 'wooden_pickaxe', async () => {
                // If we need to recover the wooden pickaxe, what do we need?
                // Wood. (Defined in generic ensureEquipped)
            })
            await mineBlock(bot, mcData, 'stone', 14) 
        },
        // Recovery Fn for the main runSmartTask loop
        async (err) => {
            if (err.message.includes('No block provided') || err.message.includes('TIMEOUT')) {
                 await exploreRandom(bot, 20) // Move if we can't find stone
            } else {
                // If the error was about the tool, ensureEquipped will try to fix it in the next cycle,
                // but we can force a check here.
                console.log(`[REPLAN] Checking if I need to recover my wooden pickaxe...`)
                const hasPickaxe = bot.inventory.findInventoryItem(getItemId(mcData, 'wooden_pickaxe'), null)
                if (!hasPickaxe) {
                    console.log(`[REPLAN] I don't have a wooden pickaxe. Crafting one before retrying...`)
                    await ensureEquipped(bot, mcData, 'wooden_pickaxe') // This will handle full recovery
                }
            }
        }
    )

    // Task: Craft stone pickaxe
    await runSmartTask(
        bot, 'Craft Stone Pickaxe',
        async () => hasItem(bot, mcData, 'stone_pickaxe'),
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 2)) await craftItem(bot, mcData, 'stick', 1)
            await craftItem(bot, mcData, 'stone_pickaxe', 1)
        }
    )
}

async function phaseIron(bot, mcData) {
    // Task: Get iron
    await runSmartTask(
        bot, 'Mine Iron',
        async () => hasItem(bot, mcData, 'raw_iron', 3) || hasItem(bot, mcData, 'iron_ingot', 3),
        async () => {
            // Ensure stone pickaxe (required tier for iron)
            await ensureEquipped(bot, mcData, 'stone_pickaxe', async () => {
                // If we need to recover the stone pickaxe, we need cobblestone
                // If we don't have cobblestone, go mine stone
                if (!hasItem(bot, mcData, 'cobblestone', 3)) {
                    console.log('[REPLAN] I need stone for the new pickaxe.')
                    await ensureEquipped(bot, mcData, 'wooden_pickaxe') // Recursion: ensure wooden one to mine stone
                    await mineBlock(bot, mcData, 'stone', 3)
                }
            })
            // Mine iron
            await mineBlock(bot, mcData, 'iron_ore', 3)
        }
    )

    // Task: Get coal (for furnace)
    await runSmartTask(
        bot, 'Mine Coal',
        async () => hasItem(bot, mcData, 'coal', 3),
        async () => {
            await ensureEquipped(bot, mcData, 'stone_pickaxe') // Use the stone one, it's faster
            await mineBlock(bot, mcData, 'coal_ore', 3)
        }
    )
    
    // Task: Furnace and smelting
    await runSmartTask(
        bot, 'Smelt Iron',
        async () => hasItem(bot, mcData, 'iron_ingot', 3),
        async () => {
            // 1. Ensure furnace
            if (!hasItem(bot, mcData, 'furnace') && !bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 16 })) {
                await ensureCraftingTable(bot, mcData)
                await craftItem(bot, mcData, 'furnace', 1)
                await placeBlockFull(bot, mcData, 'furnace')
            }
            // 2. Smelt
            await smeltItem(bot, mcData, 'iron_ingot', 3)
        }
    )

    // Final task: Iron pickaxe
    await runSmartTask(
        bot, 'Craft Iron Pickaxe',
        async () => hasItem(bot, mcData, 'iron_pickaxe'),
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 2)) await craftItem(bot, mcData, 'stick', 1)
            await craftItem(bot, mcData, 'iron_pickaxe', 1)
        }
    )
}

/**
 * =========================================================
 * MAIN ORCHESTRATOR
 * =========================================================
 */

async function runFullProgression(bot, mcData, metricsCollector = null) {
    bot.chat('Starting Progression with Replanning System v2')
    
    try {
        await phaseWood(bot, mcData)
        bot.chat('--- Wood Phase Complete ---')
        
        await phaseStone(bot, mcData)
        bot.chat('--- Stone Phase Complete ---')
        
        await phaseIron(bot, mcData)
        bot.chat('✓ Mission Complete! I have iron tools.')
        
    } catch (error) {
        bot.chat(`[FATAL] Progression stopped: ${error.message}`)
        console.error(error)
    }
}

export {
    runFullProgression
}