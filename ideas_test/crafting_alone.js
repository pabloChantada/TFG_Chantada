const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')
const collectBlock = require('mineflayer-collectblock').plugin
const mineflayerViewer = require('prismarine-viewer').mineflayer

/**
 * The bot will craft a wooden pickaxe when a player sends "craft" in chat.
 * This time the bot will do it alone, without player interaction.
 */

/** 
 * The bot will need to collect:
 * - 3 Oak Logs (to make 12 planks)
 * - 1 Sticks (2 planks = 4 sticks, need 2 sticks for pickaxe)
 * - 1 Crafting Table (to craft the pickaxe)
 */

/**
 * The process would be:
 * 1. Collect 3 Oak Logs from nearby trees
 * 2. Craft 12 Oak Planks from the logs
 * 3. Craft 4 Sticks from the planks
 * 4. Place the crafting table if not already placed
 * 5. Craft the Wooden Pickaxe using the crafting table
 * (Extra) 6. Collect 3 Stone from nearby stone blocks
 * (Extra) 7. Craft a Stone Pickaxe using the crafting table
 */


const bot = mineflayer.createBot({
    host: 'localhost',
    port: parseInt(process.argv[2]) || 25565, // Pass port as command line argument, or use default 25565
    username: 'collecterBot',
    version: '1.20.1',
    auth: 'offline',
})

const mcData = require('minecraft-data')(bot.version)

function runBot() {
    bot.loadPlugin(pathfinder)
    bot.loadPlugin(collectBlock)
    inventoryViewer(bot)

    bot.once('spawn', () => {
        bot.chat('Bot spawned')  
        mineflayerViewer(bot, { port: 3001, firstPerson: true }) // Start the viewing server on port 3001, 3000 is reserved for inventory

        // Draw the path followed by the bot
        const path = [bot.entity.position.clone()]
        bot.on('move', () => {
            if (path[path.length - 1].distanceTo(bot.entity.position) > 1) {
            path.push(bot.entity.position.clone())
            bot.viewer.drawLine('path', path)
            }
        })

        bot.on('chat', (username, message) => {
            if (username === bot.username) return
            
            if (message === 'clear') {
                clearInventory()
            }
            
            if (message === 'craft') {
                (async () => {
                    try {
                        await collectLogs(3)      
                        await craftPlanks()
                        await craftSticks()
                        await craftCraftingTable()
                        await craftWoodenPickaxe()
                        await collectStone(3)      // EXTRA
                        await craftStonePickaxe()  // EXTRA
                        bot.chat('Process completed!')
                    } catch (err) {
                        bot.chat(`Error in crafting process: ${err.message}`)
                        console.error(err)
                    }
                })()
            }
        })
    })
}

let oakLogItemId;
let oakLogBlockId;
async function collectLogs(requiredLogs) {
    // In the world, oak logs are represented by the block name 'oak_log'
    // While in the inventory, they are represented by the item name 'oak_log'
    // There might be a better way to handle this ID's but for now we use this version
    oakLogItemId = mcData.itemsByName.oak_log.id
    oakLogBlockId = mcData.blocksByName.oak_log.id
    
    console.log(`Oak log item ID: ${oakLogItemId}, block ID: ${oakLogBlockId}`)
    
    while (bot.inventory.count(oakLogItemId) < requiredLogs) {  
        // Find nearest oak log block
        const logBlock = bot.findBlock({
            matching: oakLogBlockId,  
            maxDistance: 32,
        })
        
        if (logBlock) {
            try {
                console.log(`Found log at ${logBlock.position}`)
                await bot.collectBlock.collect(logBlock)
                
                const currentCount = bot.inventory.count(oakLogItemId) 
                console.log(`Inventory count after collection: ${currentCount}`)
                console.log(`Inventory items:`, bot.inventory.items().map(i => `${i.name}x${i.count}`))
                
                bot.chat(`Collected an oak log. Total: ${currentCount}/${requiredLogs}`)
            } catch (err) {
                bot.chat(`Error collecting oak log: ${err.message}`)
                console.error('Collection error:', err)
                return
            }
        } else {
            bot.chat('No oak logs found nearby!')
            console.log('No oak logs found in range')
            return
        }
    }
    
    bot.chat(`Successfully collected ${requiredLogs} oak logs!`)
}

async function craftPlanks() {
    const planksId = mcData.itemsByName.oak_planks.id
    const planksRecipes = bot.recipesFor(planksId, null, 1, null)
    
    if (planksRecipes.length === 0) {
        bot.chat('No recipe found for planks.')
        return
    }
    
    const craftTimes = bot.inventory.count(oakLogItemId) 
    
    // We use all our logs to craft planks
    for (let i = 0; i < craftTimes; i++) {
        try {
            await bot.craft(planksRecipes[0], 1, null)
            bot.chat(`Crafted planks (${(i + 1) * 4})`)
        } catch (err) {
            bot.chat(`Error crafting planks: ${err.message}`)
            return
        }
    }
    
    bot.chat('Successfully crafted planks!')
}

async function craftSticks() {
    bot.chat('Starting to craft sticks...')
    const sticksId = mcData.itemsByName.stick.id  
    const sticksRecipes = bot.recipesFor(sticksId, null, 1, null)
    
    if (sticksRecipes.length === 0) {
        bot.chat('No recipe found for sticks.')
        return
    }
    
    try {
        await bot.craft(sticksRecipes[0], 1, null)
        const sticksCount = bot.inventory.count(sticksId)
        bot.chat(`Crafted sticks. Total: ${sticksCount}`)
    } catch (err) {
        bot.chat(`Error crafting sticks: ${err.message}`)
        return
    }
}

async function craftCraftingTable() {
    bot.chat('Starting to craft crafting table...')
    const craftingTableId = mcData.itemsByName.crafting_table.id  
    const craftingTableRecipes = bot.recipesFor(craftingTableId, null, 1, null)

    if (craftingTableRecipes.length === 0) {
        bot.chat('No recipe found for crafting table.')
        return
    }

    try {
        // Generate one crafting table
        await bot.craft(craftingTableRecipes[0], 1, null)
        bot.chat('Crafted crafting table.')
        
        // Set up the crafting table 
        const craftingTablePosition = bot.entity.position.offset(1, 0, 0)
        let craftingTableBlock = bot.blockAt(craftingTablePosition)
        
        if (!craftingTableBlock || craftingTableBlock.name !== 'crafting_table') {
            const craftingTableItem = bot.inventory.items().find(item => item.name === 'crafting_table')
            
            if (!craftingTableItem) {
                bot.chat('No crafting table in inventory.')
                return
            }

            // Equip and place the crafting table
            await bot.equip(craftingTableItem, 'hand')
            const blockBelow = bot.blockAt(craftingTablePosition.offset(0, -1, 0)) 
            await bot.placeBlock(blockBelow, new Vec3(0, 1, 0))
            bot.chat('Crafting table placed.')
        } else {
            bot.chat('Crafting table already placed.')
        }

    } catch (err) {
        bot.chat(`Error with crafting table: ${err.message}`)
        console.error(err)
        return
    }
}

async function craftWoodenPickaxe() {
    bot.chat('Starting to craft wooden pickaxe...')
    const woodenPickaxeId = mcData.itemsByName.wooden_pickaxe.id
    
    const craftingTableBlock = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 8
    })
    
    if (!craftingTableBlock) {
        bot.chat('Crafting table not found nearby!')
        return
    }
    
    bot.chat(`Found crafting table at ${craftingTableBlock.position}`)

    const woodenPickaxeRecipes = bot.recipesFor(woodenPickaxeId, null, 1, craftingTableBlock) 

    if (woodenPickaxeRecipes.length === 0) {
        bot.chat('No recipe found for wooden pickaxe.')
        return
    }
    
    try {
        await bot.craft(woodenPickaxeRecipes[0], 1, craftingTableBlock)
        const pickaxeCount = bot.inventory.count(woodenPickaxeId)
        bot.chat(`Crafted wooden pickaxe! Total: ${pickaxeCount}`)
    } catch (err) {
        bot.chat(`Error crafting wooden pickaxe: ${err.message}`)
        console.error(err)
        return
    }
}

runBot()

/**
 * TO BE IMPLEMENTED FUNCTIONS
 */

let cobblestoneItemId;
let stoneBlockId;

async function collectStone(requiredStone) {
    // FIX: Use itemsByName for the item you receive in inventory
    cobblestoneItemId = mcData.itemsByName.cobblestone.id  // CORRECTO
    stoneBlockId = mcData.blocksByName.stone.id
    
    console.log(`Cobblestone item ID: ${cobblestoneItemId}, Stone block ID: ${stoneBlockId}`)
    
    while (bot.inventory.count(cobblestoneItemId) < requiredStone) {  
        const stoneBlock = bot.findBlock({
            matching: stoneBlockId,
            maxDistance: 64,
        })
        
        if (stoneBlock) {
            try {
                console.log(`Found stone at ${stoneBlock.position}`)
                await bot.collectBlock.collect(stoneBlock)
                
                const currentCount = bot.inventory.count(cobblestoneItemId) 
                console.log(`Inventory count after collection: ${currentCount}`)
                console.log(`Inventory items:`, bot.inventory.items().map(i => `${i.name}x${i.count}`))
                
                bot.chat(`Collected stone. Total: ${currentCount}/${requiredStone}`)
            } catch (err) {
                bot.chat(`Error collecting stone: ${err.message}`)
                console.error('Collection error:', err)
                return
            }
        } else {
            bot.chat('No stone found nearby!')
            console.log('No stone found in range')
            return
        }
    }
    
    bot.chat(`Successfully collected ${requiredStone} cobblestone!`)
}

async function craftStonePickaxe() {
    bot.chat('Starting to craft stone pickaxe...')
    const stonePickaxeId = mcData.itemsByName.stone_pickaxe.id
    
    const craftingTableBlock = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 12
    })
    
    if (!craftingTableBlock) {
        bot.chat('Crafting table not found nearby!')
        return
    }
    
    bot.chat(`Found crafting table at ${craftingTableBlock.position}`)

    const stonePickaxeRecipe = bot.recipesFor(stonePickaxeId, null, 1, craftingTableBlock) 

    if (stonePickaxeRecipe.length === 0) {
        bot.chat('No recipe found for wooden pickaxe.')
        return
    }
    
    try {
        await bot.craft(stonePickaxeRecipe[0], 1, craftingTableBlock)
        // The variables are define in the function scope, so we can reuse them here
        const pickaxeCount = bot.inventory.count(stoneBlockId)
        bot.chat(`Crafted stone  pickaxe! Total: ${pickaxeCount}`)
    } catch (err) {
        bot.chat(`Error crafting stone pickaxe: ${err.message}`)
        console.error(err)
        return
    }
}

/**
 * AUXILIARY FUNCTION TO CLEAR INVENTORY
 */

async function clearInventory() {
    bot.chat('Clearing inventory')
    const items = bot.inventory.items()
    
    if (items.length === 0) {
        bot.chat('Inventory already empty')
        return
    }
    // Drop each item in the inventory
    for (const item of items) {
        try {
            await bot.toss(item.type, null, item.count)
            await bot.waitForTicks(2)  // Pequeña pausa entre cada toss
        } catch (err) {
            console.error(`Error dropping ${item.name}:`, err.message)
        }
    }
    
    bot.chat(`Cleared ${items.length} item types from inventory`)
}

