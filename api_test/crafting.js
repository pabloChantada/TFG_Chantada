const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder'). pathfinder
const Movements = require('mineflayer-pathfinder').Movements
const { GoalNear } = require('mineflayer-pathfinder').goals
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')

/**
 * We use async/await to execute crafting steps sequentially.
 * The bot will craft a wooden pickaxe when a player sends "craft" in chat.
 * WE ASSUME THE BOT HAS LOGS AND A CRAFTING TABLE IN ITS INVENTORY.
 */

const bot = mineflayer.createBot({
    host: 'localhost',
    port: 64183,
    username: 'crafterBot',
    version: '1.20.1',
    auth: 'offline',
})

inventoryViewer(bot)
bot.loadPlugin(pathfinder)

async function craftWoodenPickaxe() {
    const mcData = require('minecraft-data')(bot.version)
    
    // Get item IDs using minecraft-data
    const planksId = mcData.itemsByName.oak_planks.id
    const sticksId = mcData.itemsByName.stick.id
    const woodenPickaxeId = mcData.itemsByName.wooden_pickaxe.id

    const craftingTablePosition = bot.entity.position.offset(1, 0, 0)

    // Set up pathfinder
    const movements = new Movements(bot, mcData)
    bot.pathfinder.setMovements(movements)

    try {
        // Move near the crafting table position
        await bot.pathfinder.goto(new GoalNear(craftingTablePosition. x, craftingTablePosition.y, craftingTablePosition.z, 1))

        // Check if crafting table exists or place it
        let craftingTableBlock = bot.blockAt(craftingTablePosition)
        
        if (!craftingTableBlock || craftingTableBlock.name !== 'crafting_table') {
            const craftingTableItem = bot.inventory.items().find(item => item.name === 'crafting_table')
            if (! craftingTableItem) {
                bot.chat('No crafting table in inventory.')
                return
            }

            await bot.equip(craftingTableItem, 'hand')
            const blockBelow = bot.blockAt(craftingTablePosition. offset(0, -1, 0))
            await bot.placeBlock(blockBelow, new Vec3(0, 1, 0))
            bot.chat('Crafting table placed.')
        } else {
            bot.chat('Crafting table already placed.')
        }

        // Update the crafting table block reference
        craftingTableBlock = bot.blockAt(craftingTablePosition)

        // Step 1: Craft planks (can be done without crafting table)
        // recipesFor -> (resultItemID, resultMetadata, resultCount, craftingTableBlock)
        // Returns a list of Recipe instances that you could use to craft itemType with metadata.
        // This list is all the ways you could craft that item
        const planksRecipes = bot.recipesFor(planksId, null, 1, null)
        if (planksRecipes.length === 0) {
            bot. chat('No recipe found for planks.  Do you have logs?')
            return
        }
        await bot.craft(planksRecipes[0], 4, null) // Craft 4 times to get 16 planks
        bot.chat('Crafted planks.')

        // Step 2: Craft sticks (can be done without crafting table)
        const sticksRecipes = bot.recipesFor(sticksId, null, 1, null)
        if (sticksRecipes.length === 0) {
            bot.chat('No recipe found for sticks.')
            return
        }
        await bot.craft(sticksRecipes[0], 1, null) // Craft sticks
        bot.chat('Crafted sticks.')

        // Step 3: Craft wooden pickaxe (requires crafting table)
        const pickaxeRecipes = bot.recipesFor(woodenPickaxeId, null, 1, craftingTableBlock)
        if (pickaxeRecipes.length === 0) {
            bot. chat('No recipe found for wooden pickaxe.  Check materials.')
            return
        }
        await bot.craft(pickaxeRecipes[0], 1, craftingTableBlock)
        bot.chat('Successfully crafted wooden pickaxe!')

    } catch (err) {
        bot.chat(`Error: ${err.message}`)
        console.error(err)
    }
}

bot.once('spawn', () => {
    bot. on('chat', (username, message) => {
        if (username === bot.username) return // Ignore own messages
        if (message === 'craft') {
            craftWoodenPickaxe()
        }
    })
})