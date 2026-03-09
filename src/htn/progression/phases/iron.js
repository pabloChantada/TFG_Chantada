import { getBotLabel } from '../utils.js'
import { runSmartTask } from '../smart_task.js'
import { hasItem } from '../../primitives/inventory.js'
import { getBlockId } from '../../primitives/blocks.js'
import { mineBlock } from '../../primitives/mining.js'
import { craftItem } from '../../tasks/crafting.js'
import { ensureCraftingTable } from '../../tasks/block_placement.js'
import { placeBlockFull } from '../../tasks/block_placement.js'
import { smeltItem } from '../../tasks/smelting.js'
import { ensureEquipped } from '../recovery.js'
import { exploreRandom } from '../../primitives/movement.js'

async function phaseIron(bot, mcData) {
    // Task 1: Mine iron ore
    await runSmartTask(
        bot, 
        'Mine Iron',
        async () => hasItem(bot, mcData, 'raw_iron', 3) || hasItem(bot, mcData, 'iron_ingot', 3),
        async () => {
            await ensureEquipped(bot, mcData, 'stone_pickaxe', async () => {
                if (!hasItem(bot, mcData, 'cobblestone', 3)) {
                    console.log(`[REPLAN] [${getBotLabel(bot)}] I need stone for the new pickaxe.`)
                    await ensureEquipped(bot, mcData, 'wooden_pickaxe')
                    await mineBlock(bot, mcData, 'stone', 3)
                }
            })
            await mineBlock(bot, mcData, 'iron_ore', 3)
        }
    )

    // Task 2: Mine coal
    await runSmartTask(
        bot, 
        'Mine Coal',
        async () => hasItem(bot, mcData, 'coal', 3),
        async () => {
            await ensureEquipped(bot, mcData, 'stone_pickaxe', async () => {
                if (!hasItem(bot, mcData, 'cobblestone', 3)) {
                    console.log(`[REPLAN] [${getBotLabel(bot)}] I need stone for a new stone pickaxe.`)
                    await ensureEquipped(bot, mcData, 'wooden_pickaxe')
                    await mineBlock(bot, mcData, 'stone', 3)
                }
            })
            await mineBlock(bot, mcData, 'coal_ore', 3)
        }
    )
    
    // Task 3: Smelt iron
    await runSmartTask(
        bot, 
        'Smelt Iron',
        async () => hasItem(bot, mcData, 'iron_ingot', 3),
        async () => {
            const { getFurnace } = await import('../../primitives/structures.js')
            
            let furnace = getFurnace(bot, mcData)
            if (!furnace) {
                if (!hasItem(bot, mcData, 'furnace', 1)) {
                    await ensureCraftingTable(bot, mcData)
                    await craftItem(bot, mcData, 'furnace', 1)
                }
                await placeBlockFull(bot, mcData, 'furnace')
            }
            await smeltItem(bot, mcData, 'iron_ingot', 3)
        },
        async (err) => {
            console.log(`[REPLAN] [${getBotLabel(bot)}] Smelting failed. Breaking old furnace and placing new one...`)
            try {
                const { getFurnace, clearFurnaceMemory } = await import('../../primitives/structures.js')
                
                clearFurnaceMemory(bot)
                
                const oldFurnace = bot.findBlock({
                    matching: getBlockId(mcData, 'furnace'),
                    maxDistance: 32
                })
                
                if (oldFurnace) {
                    console.log(`[REPLAN] Breaking old furnace at (${oldFurnace.position.x}, ${oldFurnace.position.y}, ${oldFurnace.position.z})`)
                    await mineBlock(bot, mcData, 'furnace', 1)
                }
                
                await exploreRandom(bot, 10)
                
                if (!hasItem(bot, mcData, 'furnace', 1)) {
                    await ensureCraftingTable(bot, mcData)
                    await craftItem(bot, mcData, 'furnace', 1)
                }
                
            } catch (recoveryErr) {
                console.warn(`[REPLAN] Furnace recovery had issues: ${recoveryErr.message}`)
                await exploreRandom(bot, 5)
            }
        }
    )

    // Task 4: Craft iron pickaxe
    await runSmartTask(
        bot, 
        'Craft Iron Pickaxe',
        async () => hasItem(bot, mcData, 'iron_pickaxe'),
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 2)) await craftItem(bot, mcData, 'stick', 1)
            await craftItem(bot, mcData, 'iron_pickaxe', 1)
        },
        null,
        { exploreOnFail: false }
    )
}

export { phaseIron }