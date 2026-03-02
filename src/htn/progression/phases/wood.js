import { getBotLabel } from '../utils.js'
import { runSmartTask } from '../smart_task.js'
import { hasItem, getItemId } from '../../primitives/inventory.js'
import { chop, obtainWoodType, obtainPlankType, getWoodTypeFromInventory } from '../../primitives/wood.js'
import { craftItem } from '../../tasks/crafting.js'
import { ensureCraftingTable } from '../../tasks/block_placement.js'
import { getCraftingTable } from '../../primitives/structures.js'
import { exploreRandom } from '../../primitives/movement.js'

async function phaseWood(bot, mcData) {
    const inventoryWoodType = getWoodTypeFromInventory(bot)
    const woodType = inventoryWoodType || await obtainWoodType(bot, mcData)
    const plankType = await obtainPlankType(bot, mcData, woodType)

    // Task 1: Collect wood
    await runSmartTask(
        bot, 
        'Collect Wood',
        async () => hasItem(bot, mcData, woodType, 5) || hasItem(bot, mcData, plankType, 12),
        async () => chop(bot, mcData, 5)
    )

    // Task 2: Craft planks
    await runSmartTask(
        bot, 
        'Craft Planks',
        async () => hasItem(bot, mcData, plankType, 12),
        async () => craftItem(bot, mcData, plankType, 4),
        null,
        { exploreOnFail: false }
    )

    // Task 3: Build crafting table
    await runSmartTask(
        bot, 
        'Place Crafting Table',
        async () => {
            try {
                return Boolean(getCraftingTable(bot, mcData))
            } catch (_e) {
                return false
            }
        },
        async () => ensureCraftingTable(bot, mcData),
        async (err) => {
            console.log(`[REPLAN] [${getBotLabel(bot)}] Crafting table placement failed. Moving to new location...`)
            await exploreRandom(bot, 5)
        }
    )

    // Task 4: Craft wooden pickaxe
    await runSmartTask(
        bot, 
        'Craft Wooden Pickaxe',
        async () => hasItem(bot, mcData, 'wooden_pickaxe'),
        async () => {
            if (!hasItem(bot, mcData, 'stick', 2)) await craftItem(bot, mcData, 'stick', 1)
            await craftItem(bot, mcData, 'wooden_pickaxe', 1)
        },
        null,
        { exploreOnFail: false }
    )
}

export { phaseWood }