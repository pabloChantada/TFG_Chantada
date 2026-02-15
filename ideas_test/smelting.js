
const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const { Movements, goals } = require('mineflayer-pathfinder')
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')
const collectBlock = require('mineflayer-collectblock').plugin
const mineflayerViewer = require('prismarine-viewer').mineflayer

// In this version we'll use "cheating", but in the next version we 
// need to simulate vision correctly.

const bot = mineflayer.createBot({
    host: 'localhost',
    port: parseInt(process.argv[2]) || 25565,
    username: 'SmeltingBot',
    version: '1.20.1',
    auth: 'offline',
})

let mcData

bot.loadPlugin(pathfinder)
bot.loadPlugin(collectBlock)

bot.once('spawn', () => {
    mcData = require('minecraft-data')(bot.version)
    inventoryViewer(bot)
    mineflayerViewer(bot, { port: 3001, firstPerson: true })
    bot.chat('Bot listo para trabajar')
})

// --- Lógica Principal ---

bot.on('chat', async (username, message) => {
    if (username === bot.username) return

    try {
        if (message === 'smelt') {
            await startFullProgression()
        } else if (message === 'clear') {
            await clearInventory()
        }
    } catch (err) {
        bot.chat(`Error: ${err.message}`)
        console.error(err)
    }
})

async function startFullProgression() {
    // 1. Preparar básicos (Planks y Sticks no requieren mesa)
    await collectResource('oak_log', 3)
    await smartCraft('oak_planks', 3) // 3 logs -> 12 planks
    await smartCraft('stick', 1)      // 2 planks -> 4 sticks
    
    // 2. Mesa de trabajo
    await smartCraft('crafting_table', 1)
    await placeCraftingTable()
    
    await smartCraft('wooden_pickaxe', 1)
    
    // 2. Recolectar piedra
    await collectResource('stone', 8)    

    // 3. Horno
    await smartCraft('furnace', 1)
    
    // 4. Fundir hierro (Se asume que ya tenemos hierro y carbón en el inventario)
    const furnaceBlock = bot.findBlock({
        matching: mcData.blocksByName.furnace.id,
        maxDistance: 4
    })
    if (!furnaceBlock) throw new Error('No se encontró un horno cercano.')

    const ironOre = bot.inventory.findInventoryItem(mcData.itemsByName.iron_ore.id)
    const coal = bot.inventory.findInventoryItem(mcData.itemsByName.coal.id)

    if (!ironOre || !coal) throw new Error('Faltan materiales para fundir.')

    await bot.placeBlock(bot.blockAt(bot.entity.position.offset(0, 0, 0)), new Vec3(1, 0, 0))
    await bot.smelt(ironOre, coal, furnaceBlock)

    bot.chat('¡Hierro fundido con éxito!')
}

// --- Funciones Reutilizables ---

/**
 * Recolecta cualquier bloque por nombre
 */
async function collectResource(blockName, count) {
    const blockId = mcData.blocksByName[blockName].id
    const itemId = mcData.itemsByName[blockName]?.id || mcData.blocksByName[blockName].id

    bot.chat(`Buscando ${blockName}...`)

    while (bot.inventory.count(itemId) < count) {
        const block = bot.findBlock({
            matching: blockId,
            maxDistance: 32
        })

        if (!block) throw new Error(`No encontré ${blockName} cerca.`)
        await bot.collectBlock.collect(block)
    }
}

/**
 * Craftea cualquier item de forma inteligente
 */
async function smartCraft(itemName, amount) {
    const item = mcData.itemsByName[itemName]
    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 4
    })

    const recipe = bot.recipesFor(item.id, null, amount, craftingTable)[0]
    
    if (!recipe) {
        throw new Error(`No tengo materiales para ${itemName} o falta mesa cercana.`)
    }

    await bot.craft(recipe, amount, craftingTable)
    bot.chat(`Crafteado: ${itemName}`)
}

/**
 * Coloca la mesa de trabajo frente al bot
 */
async function placeCraftingTable() {
    const tableItem = bot.inventory.findInventoryItem(mcData.itemsByName.crafting_table.id)
    if (!tableItem) return

    // Evitar colocarla si ya hay una cerca
    const existingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 4
    })
    if (existingTable) return

    await bot.equip(tableItem, 'hand')
    const pos = bot.entity.position.offset(0, 0, 0) // Bloque de suelo adyacente
    await bot.placeBlock(bot.blockAt(pos), new Vec3(1, 0, 0))
    bot.chat('Mesa colocada')
}

async function clearInventory() {
    for (const item of bot.inventory.items()) {
        await bot.toss(item.type, null, item.count)
    }
    bot.chat('Inventario limpio')
}