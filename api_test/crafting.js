const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const Movements = require('mineflayer-pathfinder').Movements
const { GoalNear } = require('mineflayer-pathfinder').goals
const Vec3 = require('vec3')

const bot = mineflayer.createBot({
    host: 'localhost',
    port: 63021,
    username: 'crafterBot',
    version: '1.20.1',
    auth: 'offline',
})

bot.loadPlugin(pathfinder)

function craftWoodenPickaxe() {
    // The crafting table is set 1 block east of the bot's spawn position
    const craftingTablePosition = bot.entity.position.offset(1, 0, 0)

    // Check if a crafting table already exists at the target position
    const craftingTableBlock = bot.blockAt(craftingTablePosition)

    function placeAndActivateCraftingTable(callback) {
        if (craftingTableBlock && craftingTableBlock.name === 'crafting_table') {
            // Crafting table already placed
            bot.chat('Crafting table already placed at the position.')
            callback() // Proceed to crafting steps
        } else {
            // Find the crafting table in the bot's inventory
            const craftingTable = bot.inventory.items().find(item => item.name === 'crafting_table')
            if (!craftingTable) {
                bot.chat('No crafting table in inventory.')
                return
            }

            // Equip and place the crafting table
            bot.equip(craftingTable, 'hand', (equipErr) => {
                if (equipErr) {
                    bot.chat('Failed to equip crafting table.')
                    return
                }

                // Place the crafting table on the ground
                bot.placeBlock(bot.blockAt(craftingTablePosition.offset(0, -1, 0)), new Vec3(0, 1, 0), (placeErr) => {
                    if (placeErr) {
                        bot.chat('Failed to place crafting table.')
                        return
                    }

                    // Proceed to crafting after successful placement
                    callback()
                })
            })
        }
    }

    function craftItems() {
        // Wait and activate crafting table
        const craftingTableBlock = bot.blockAt(craftingTablePosition)
        bot.activateBlock(craftingTableBlock)

        // Obtain recipes for planks, sticks, and wooden pickaxe
        const planks = bot.recipesFor(5, null, 4, null)
        const sticks = bot.recipesFor(280, null, 4, null)
        const woodenPickaxe = bot.recipesFor(270, null, 1, craftingTableBlock)

        if (planks.length === 0 || sticks.length === 0 || woodenPickaxe.length === 0) {
            bot.chat('Missing recipes for crafting.')
            return
        }

        // Craft planks -> sticks -> pickaxe in sequence
        bot.craft(planks[0], 4, null, (plankErr) => {
            if (plankErr) {
                bot.chat('Failed to craft planks.')
                return
            }

            bot.craft(sticks[0], 4, null, (stickErr) => {
                if (stickErr) {
                    bot.chat('Failed to craft sticks.')
                    return
                }

                bot.craft(woodenPickaxe[0], 1, craftingTableBlock, (pickErr) => {
                    if (pickErr) {
                        bot.chat('Failed to craft wooden pickaxe.')
                    } else {
                        bot.chat('Successfully crafted wooden pickaxe.')
                    }
                })
            })
        })
    }

    // Move to the crafting table position before placing/activating it
    const mcData = require('minecraft-data')(bot.version)
    const movements = new Movements(bot, mcData)
    bot.pathfinder.setMovements(movements)

    bot.pathfinder.goto(new GoalNear(craftingTablePosition.x, craftingTablePosition.y, craftingTablePosition.z, 1), () => {
        placeAndActivateCraftingTable(craftItems)
    })
}

bot.once('spawn', () => {
    bot.on('chat', (username, message) => {
        if (message === 'craft') {
            craftWoodenPickaxe()
        }
    })
})