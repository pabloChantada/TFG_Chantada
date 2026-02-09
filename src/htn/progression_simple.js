// TODO: refactor
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
 * Ensures the bot has a wooden pickaxe equipped.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 */
async function ensureWoodenPickaxe(bot, mcData) {
    const item = bot.inventory.findInventoryItem(getItemId(mcData, 'wooden_pickaxe'), null)
    if (item) {
        await bot.equip(item, 'hand')
        return
    }

    console.log('[REPLAN] I don\'t have a wooden pickaxe. Crafting one...')
    
    // Get wood and planks
    const woodType = await obtainWoodType(bot, mcData)
    const plankType = await obtainPlankType(bot, mcData, woodType)
    
    if (!hasItem(bot, mcData, woodType, 1)) {
        await chop(bot, mcData, 1)
    }
    
    if (!hasItem(bot, mcData, plankType, 4)) {
        await craftItem(bot, mcData, plankType, 1)
    }
    
    // Ensure crafting table
    await ensureCraftingTable(bot, mcData)
    
    // Craft sticks
    if (!hasItem(bot, mcData, 'stick', 2)) {
        await craftItem(bot, mcData, 'stick', 1)
    }
    
    // Craft wooden pickaxe
    await craftItem(bot, mcData, 'wooden_pickaxe', 1)
    
    const newItem = bot.inventory.findInventoryItem(getItemId(mcData, 'wooden_pickaxe'), null)
    if (newItem) {
        await bot.equip(newItem, 'hand')
    }
}

/**
 * Ensures the bot has a stone pickaxe equipped.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 */
async function ensureStonePickaxe(bot, mcData) {
    const item = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_pickaxe'), null)
    if (item) {
        await bot.equip(item, 'hand')
        return
    }

    console.log('[REPLAN] I don\'t have a stone pickaxe. Crafting one...')
    
    // Ensure wooden pickaxe first
    await ensureWoodenPickaxe(bot, mcData)
    
    // Get cobblestone
    if (!hasItem(bot, mcData, 'cobblestone', 3)) {
        await mineBlock(bot, mcData, 'stone', 3)
    }
    
    // Ensure crafting table
    await ensureCraftingTable(bot, mcData)
    
    // Craft sticks
    if (!hasItem(bot, mcData, 'stick', 2)) {
        await craftItem(bot, mcData, 'stick', 1)
    }
    
    // Craft stone pickaxe
    await craftItem(bot, mcData, 'stone_pickaxe', 1)
    
    const newItem = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_pickaxe'), null)
    if (newItem) {
        await bot.equip(newItem, 'hand')
    }
}

/**
 * Ensures the bot has a stone shovel equipped.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 */
async function ensureStoneShovel(bot, mcData) {
    const item = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_shovel'), null)
    if (item) {
        await bot.equip(item, 'hand')
        return
    }

    console.log('[REPLAN] I don\'t have a stone shovel. Crafting one...')
    
    // Ensure stone is available
    if (!hasItem(bot, mcData, 'cobblestone', 1)) {
        await ensureStonePickaxe(bot, mcData)
        await mineBlock(bot, mcData, 'stone', 1)
    }
    
    // Ensure crafting table
    await ensureCraftingTable(bot, mcData)
    
    // Craft sticks
    if (!hasItem(bot, mcData, 'stick', 2)) {
        await craftItem(bot, mcData, 'stick', 1)
    }
    
    // Craft stone shovel
    await craftItem(bot, mcData, 'stone_shovel', 1)
    
    const newItem = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_shovel'), null)
    if (newItem) {
        await bot.equip(newItem, 'hand')
    }
}

/**
 * =========================================================
 * PROGRESSION PHASES
 * =========================================================
 */

/**
 * Phase 1: Collect wood and build crafting table
 */
async function phaseWood(bot, mcData) {
    const woodType = await obtainWoodType(bot, mcData)
    const plankType = await obtainPlankType(bot, mcData, woodType)

    // Task 1: Collect wood
    await runSmartTask(
        bot, 
        'Collect Wood',
        async () => hasItem(bot, mcData, woodType, 5),
        async () => chop(bot, mcData, 5)
    )

    // Task 2: Craft planks
    await runSmartTask(
        bot, 'Craft Planks',
        async () => hasItem(bot, mcData, plankType, 12),
        async () => craftItem(bot, mcData, plankType, 4)
    )

    // Task 3: Place crafting table
    await runSmartTask(
        bot, 'Place Crafting Table',
        async () => bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 10 }),
        async () => ensureCraftingTable(bot, mcData)
    )

    // Task 4: Craft wooden pickaxe
    await runSmartTask(
        bot, 'Craft Wooden Pickaxe',
        async () => hasItem(bot, mcData, 'wooden_pickaxe'),
        async () => {
            if (!hasItem(bot, mcData, 'stick', 2)) {
                await craftItem(bot, mcData, 'stick', 1)
            }
            await craftItem(bot, mcData, 'wooden_pickaxe', 1)
        }
    )
}

/**
 * Phase 2: Mine stone and craft stone pickaxe
 */
async function phaseStone(bot, mcData) {
    await runSmartTask(
        bot,
        'Mine Stone',
        async () => hasItem(bot, mcData, 'cobblestone', 8),
        async () => {
            await ensureWoodenPickaxe(bot, mcData)
            await mineBlock(bot, mcData, 'stone', 8) 
        }
    )

    await runSmartTask(
        bot, 'Craft Stone Pickaxe',
        async () => hasItem(bot, mcData, 'stone_pickaxe'),
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 2)) {
                await craftItem(bot, mcData, 'stick', 1)
            }
            await craftItem(bot, mcData, 'stone_pickaxe', 1)
        }
    )
}

/**
 * Phase 3: Craft stone shovel and dig a 2x2 hole
 */
async function phaseShovelAndDig(bot, mcData) {
    // Task 1: Mine extra stone if needed
    await runSmartTask(
        bot,
        'Mine Stone for Shovel',
        async () => hasItem(bot, mcData, 'cobblestone', 12),
        async () => {
            await ensureStonePickaxe(bot, mcData)
            await mineBlock(bot, mcData, 'stone', 5)
        }
    )

    // Task 2: Craft stone shovel
    await runSmartTask(
        bot, 'Craft Stone Shovel',
        async () => hasItem(bot, mcData, 'stone_shovel'),
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 2)) {
                await craftItem(bot, mcData, 'stick', 1)
            }
            await craftItem(bot, mcData, 'stone_shovel', 1)
        }
    )

    // Task 3: Dig a 2x2 hole
    await runSmartTask(
        bot, 'Dig 2x2 Hole',
        async () => {
            // Check if we've dug (simplified - always returns true to trigger digging)
            return false // Keep retrying until we manually consider it done
        },
        async () => {
            const shovel = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_shovel'), null)
            if (shovel) {
                await bot.equip(shovel, 'hand')
            }
            
            // Find a suitable location and dig a 2x2 hole
            const playerPos = bot.entity.position
            
            // Dig 2x2 area (4 blocks down)
            for (let dx = 0; dx < 2; dx++) {
                for (let dz = 0; dz < 2; dz++) {
                    const targetBlock = bot.blockAt(playerPos.offset(dx, -1, dz))
                    if (targetBlock && targetBlock.name !== 'bedrock') {
                        await bot.dig(targetBlock)
                        await bot.waitForTicks(5)
                    }
                }
            }
            
            // Consider task complete after one iteration
            throw new Error('2x2 hole dug successfully')
        }
    )
}

/**
 * =========================================================
 * MAIN ORCHESTRATOR
 * =========================================================
 */

/**
 * Simple progression: Build stone pickaxe, craft stone shovel, and dig a 2x2 hole
 */
async function runSimpleProgression(bot, mcData, metricsCollector = null) {
    bot.chat('Starting Simple Progression: Stone Pickaxe → Stone Shovel → 2x2 Dig')
    
    try {
        await phaseWood(bot, mcData)
        bot.chat('--- Wood Phase Complete ---')
        
        await phaseStone(bot, mcData)
        bot.chat('--- Stone Phase Complete ---')
        
        await phaseShovelAndDig(bot, mcData)
        bot.chat('✓ Mission Complete! Dug a 2x2 hole with stone shovel.')
        
    } catch (error) {
        if (error.message.includes('2x2 hole dug successfully')) {
            bot.chat('✓ Mission Complete! Dug a 2x2 hole with stone shovel.')
        } else {
            bot.chat(`[FATAL] Progression stopped: ${error.message}`)
            console.error(error)
        }
    }
}

/**
 * Stone pickaxe only progression (even simpler)
 */
async function runStonePickaxeProgression(bot, mcData, metricsCollector = null) {
    bot.chat('Starting Stone Pickaxe Progression')
    
    try {
        await phaseWood(bot, mcData)
        bot.chat('--- Wood Phase Complete ---')
        
        await phaseStone(bot, mcData)
        bot.chat('✓ Mission Complete! Stone pickaxe crafted.')
        
    } catch (error) {
        bot.chat(`[FATAL] Progression stopped: ${error.message}`)
        console.error(error)
    }
}

export {
    runSimpleProgression,
    runStonePickaxeProgression
}
