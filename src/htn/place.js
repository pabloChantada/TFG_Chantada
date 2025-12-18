const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const { Movements } = require('mineflayer-pathfinder')
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')
const collectBlock = require('mineflayer-collectblock').plugin
const mineflayerViewer = require('prismarine-viewer').mineflayer

const bot = mineflayer.createBot({
    host: 'localhost',
    port: parseInt(process.argv[2]) || 25565,
    username: 'TestBot',
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
    
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    bot.pathfinder.setMovements(defaultMove)

    bot.chat('Listo. Escribe "test" para probar SOLO la mesa.')
})

bot.on('chat', async (username, message) => {
    if (username === bot.username) return

    if (message === 'test') {
        await runTableTest()
    } else if (message === 'clear') {
        const items = bot.inventory.items()
        for (const item of items) await bot.toss(item.type, null, item.count)
    }
})

async function runTableTest() {
    try {
        bot.chat('--- INICIANDO TEST DE MESA ---')
        
        // 1. Recolectar 1 tronco (suficiente para 1 mesa)
        if (bot.inventory.count(mcData.itemsByName.oak_log.id) < 1) {
            bot.chat('1. Recolectando 1 tronco...')
            await collectResource('oak_log', 1)
        }

        // 2. Craftear Tablones
        bot.chat('2. Crafteando tablones...')
        await smartCraft('oak_planks', 1) // 1 log = 4 planks

        // 3. Craftear Mesa
        bot.chat('3. Crafteando mesa...')
        await smartCraft('crafting_table', 1)

        // 4. Colocar Mesa (Versión Robusta)
        bot.chat('4. Colocando mesa...')
        await placeCraftingTable()

        bot.chat('--- TEST FINALIZADO CON ÉXITO ---')

    } catch (err) {
        bot.chat(`ERROR: ${err.message}`)
        console.error(err)
    }
}

// --- FUNCIONES ---

async function collectResource(blockName, count) {
    const blockId = mcData.blocksByName[blockName].id
    const itemId = mcData.itemsByName[blockName]?.id || blockId

    while (bot.inventory.count(itemId) < count) {
        const block = bot.findBlock({ matching: blockId, maxDistance: 32 })
        if (!block) throw new Error(`No encuentro ${blockName}`)
        await bot.collectBlock.collect(block)
    }
}

async function smartCraft(itemName, amount) {
    const item = mcData.itemsByName[itemName]
    // Para tablones y mesa, NO necesitamos mesa de crafteo para fabricarlos
    // Pasamos null como block de crafteo
    const recipe = bot.recipesFor(item.id, null, amount, null)[0]
    
    if (!recipe) throw new Error(`No puedo craftear ${itemName} (quizás falta material)`)
    await bot.craft(recipe, amount, null)
}

// ESTA ES LA FUNCIÓN QUE ARREGLA TU ERROR
// No usa offset fijo, usa la mirada del bot.
async function placeCraftingTable() {
    const tableId = mcData.itemsByName.crafting_table.id
    const tableItem = bot.inventory.findInventoryItem(tableId)
    
    if (!tableItem) throw new Error('No tengo la mesa en el inventario.')

    // Detener pathfinder para que el bot no se mueva mientras construye
    bot.pathfinder.setGoal(null)
    await bot.waitForTicks(2)

    // --- CÁLCULO DE POSICIÓN ---
    // Usamos Yaw para calcular el bloque enfrente
    const yaw = bot.entity.yaw
    const distance = 2.0 // 2 bloques de distancia para evitar colisión
    
    const p = bot.entity.position
    const dx = -Math.sin(yaw) * distance
    const dz = -Math.cos(yaw) * distance
    
    const targetPos = p.offset(dx, 0, dz).floored()
    
    // Bloque donde se apoyará (suelo)
    const groundBlock = bot.blockAt(targetPos.offset(0, -1, 0))
    // Bloque donde irá la mesa (aire/hierba)
    const targetBlock = bot.blockAt(targetPos)

    console.log(`Intentando poner en: ${targetPos} (Suelo: ${groundBlock?.name})`)

    // --- VALIDACIÓN Y LIMPIEZA ---
    if (!groundBlock || groundBlock.boundingBox !== 'block') {
        throw new Error('No hay suelo sólido enfrente.')
    }

    // Si hay hierba o flores, limpiamos
    if (targetBlock && targetBlock.boundingBox !== 'empty') {
        bot.chat(`Limpiando ${targetBlock.name}...`)
        await bot.dig(targetBlock)
        await bot.waitForTicks(5)
    }

    // --- COLOCACIÓN ---
    await bot.equip(tableItem, 'hand')
    
    // Mirar al suelo ayuda al servidor a validar la colocación
    await bot.lookAt(groundBlock.position.offset(0.5, 1, 0.5), true)
    
    // Intentar colocar
    try {
        await bot.placeBlock(groundBlock, new Vec3(0, 1, 0))
        bot.chat('¡Mesa colocada!')
    } catch (err) {
        throw new Error(`Fallo al colocar: ${err.message}`)
    }
}