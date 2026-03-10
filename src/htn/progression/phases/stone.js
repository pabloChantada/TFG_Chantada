import { getBotLabel } from '../utils.js'
import { runSmartTask } from '../smart_task.js'
import { hasItem, getItemId } from '../../primitives/inventory.js'
import { mineBlock } from '../../primitives/mining.js'
import { craftItem } from '../../tasks/crafting.js'
import { ensureCraftingTable } from '../../tasks/block_placement.js'
import { ensureEquipped } from '../recovery.js'
import { exploreRandom } from '../../primitives/movement.js'

async function phaseStone(bot, mcData) {
    // Task 1: Mine stone (extra cobblestone for 2 stone pickaxes: 6 cobble + 8 for furnace = 14)
    await runSmartTask(
        bot,
        'Mine Stone',
        async () => hasItem(bot, mcData, 'cobblestone', 14),
        async () => {
            await ensureEquipped(bot, mcData, 'wooden_pickaxe', async () => {
                // Obtain a wooden pickaxe if we don't have one
                await ensureEquipped(bot, mcData, 'wooden_pickaxe')
            })
            await mineBlock(bot, mcData, 'stone', 14) 
        },
        async (err) => {
            const errorMessage = err?.message ? String(err.message) : String(err)
            if (errorMessage.includes('No block provided') || errorMessage.includes('TIMEOUT')) {
                 await exploreRandom(bot, 20)
            } else {
                console.log(`[REPLAN] [${getBotLabel(bot)}] Checking if I need to recover my wooden pickaxe...`)
                const hasPickaxe = bot.inventory.findInventoryItem(getItemId(mcData, 'wooden_pickaxe'), null)
                if (!hasPickaxe) {
                    console.log(`[REPLAN] [${getBotLabel(bot)}] I don't have a wooden pickaxe. Crafting one before retrying...`)
                    await ensureEquipped(bot, mcData, 'wooden_pickaxe')
                }
            }
        }
    )

    // Task 2: Craft 2 stone pickaxes (extra durability for exploration)
    await runSmartTask(
        bot, 
        'Craft Stone Pickaxes',
        async () => {
            const pickaxeId = getItemId(mcData, 'stone_pickaxe')
            return pickaxeId && bot.inventory.count(pickaxeId) >= 2
        },
        async () => {
            await ensureCraftingTable(bot, mcData)
            if (!hasItem(bot, mcData, 'stick', 4)) await craftItem(bot, mcData, 'stick', 2)
            await craftItem(bot, mcData, 'stone_pickaxe', 2)
        },
        null,
        { exploreOnFail: false }
    )
}

export { phaseStone }