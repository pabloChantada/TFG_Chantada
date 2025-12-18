const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const { Movements, goals } = require('mineflayer-pathfinder')
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')
const collectBlock = require('mineflayer-collectblock').plugin
const mineflayerViewer = require('prismarine-viewer').mineflayer

// Import correcto de las tareas
const task = require('./task')
const collectResource = task.collectResource
const smartCraft = task.smartCraft
const placeBlock = task.placeBlock
const smeltItem = task.smeltItem
const clearInventory = task.clearInventory

// ========================================================
// --- CONFIGURACIÓN DEL BOT ---
// ========================================================

const bot = mineflayer.createBot({
    host: 'localhost',
    port: parseInt(process.argv[2]) || 25565,
    username: 'IronBot',
    version: '1.20.1',
    auth: 'offline',
})

// ========================================================
// --- CONSTANTES Y CONFIGURACIONES ---
// ========================================================

let mcData

const DROPS = {
    stone: 'cobblestone',
    grass_block: 'dirt',
    iron_ore: 'raw_iron',
    gold_ore: 'raw_gold',
    copper_ore: 'raw_copper',
    coal_ore: 'coal',
    diamond_ore: 'diamond',
    oak_log: 'oak_log'
}

bot.loadPlugin(pathfinder)
bot.loadPlugin(collectBlock)

bot.on('error', (err) => console.log('Error general:', err))
bot.on('kicked', console.log)

// =========================================================
// --- HELPERS DE REPLANIFICACIÓN ---
// =========================================================

async function exploreArea(steps = 2) {
    const dirs = ['forward', 'back', 'left', 'right']
    for (let i = 0; i < steps; i++) {
        const dir = dirs[Math.floor(Math.random() * dirs.length)]
        bot.setControlState(dir, true)
        await bot.waitForTicks(20)
        bot.setControlState(dir, false)
        await bot.waitForTicks(5)
    }
}

/**
 * runTask admite un replanFn(err, attempt) que, si devuelve true,
 * considera manejado el error y reintenta sin fallar la tarea.
 */
async function runTask(name, checkFn, actionFn, retries = 1, replanFn = null) {
    if (checkFn()) {
        bot.chat(`[SKIP] ${name}`)
        return
    }

    bot.chat(`[START] ${name}`)
    
    for (let i = 0; i <= retries; i++) {
        try {
            await actionFn()
            await bot.waitForTicks(10)
            
            if (checkFn()) {
                bot.chat(`[OK] ${name}`)
                return
            }
            throw new Error(`Acción finalizada pero condición no cumplida.`)
        } catch (err) {
            console.error(`Error en ${name}: ${err.message}`)

            if (replanFn) {
                const handled = await replanFn(err, i)
                if (handled) {
                    bot.chat(`[REPLAN] ${name}: nuevo intento (${i + 1}/${retries})`)
                    continue
                }
            }

            if (i < retries) {
                bot.chat(`Reintentando ${name}...`)
                await bot.waitForTicks(40)
            } else {
                bot.chat(`[FAIL] ${name}`)
                throw err
            }
        }
    }
}

// Replanificador
const replan = (stepsBase = 2) => async (_err, attempt = 0) => {
  await exploreArea(stepsBase + attempt)
  return true
}

// =========================================================
// --- LÓGICA DEL BOT ---
// =========================================================

bot.once('spawn', () => {
    mcData = require('minecraft-data')(bot.version)
    inventoryViewer(bot)
    mineflayerViewer(bot, { port: 3001, firstPerson: true })
    
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)

    bot.chat('Bot listo. Escribe "craft" para ir a por el hierro.')
})

bot.on('chat', async (username, message) => {
    if (username === bot.username) return
    if (message === 'craft') await startFullProgression()
    if (message === 'clear') await clearInventory()
})

// =========================================================
// --- TAREA PRINCIPAL: PROGRESIÓN COMPLETA HASTA HIERRO ---
// =========================================================

async function startFullProgression() {
    try {
        const has = (name, count = 1) => {
            const finalName = DROPS[name] || name
            const id = mcData.itemsByName[finalName]?.id
            return id ? bot.inventory.count(id) >= count : false
        }

        const tableId = () => mcData.blocksByName.crafting_table.id
        const isTablePlaced = () => bot.findBlock({ matching: tableId(), maxDistance: 16 }) !== null
        const isFurnacePlaced = () => bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 16 }) !== null

        // Guardas de fase para no repetir trabajo si reinicias
        const phase = {
            atStone: () => has('stone_pickaxe', 1) || has('iron_pickaxe', 1),
            atFurnace: () => has('furnace', 1) || isFurnacePlaced() || has('iron_pickaxe', 1),
            atIron: () => has('iron_pickaxe', 1)
        }

        // --- Madera y básicos
        if (!phase.atStone()) {
            await runTask('Madera', () => has('oak_log', 4), async () => await collectResource('oak_log', 4), 3, replan('oak_log'))
            await runTask('Tablones', () => has('oak_planks', 12), async () => await smartCraft('oak_planks', 3))
            await runTask('Palos', () => has('stick', 4), async () => await smartCraft('stick', 1))

            await runTask('Craftear Mesa', () => has('crafting_table', 1) || isTablePlaced(), async () => await smartCraft('crafting_table', 1))
            await runTask('Colocar Mesa', isTablePlaced, async () => await placeBlock('crafting_table'), 2, replan())

            await runTask('Pico Madera', () => has('wooden_pickaxe', 1) || has('stone_pickaxe', 1), async () => await smartCraft('wooden_pickaxe', 1))
            const woodPick = bot.inventory.findInventoryItem(mcData.itemsByName.wooden_pickaxe.id)
            if (woodPick) await bot.equip(woodPick, 'hand')

            await runTask('Piedra Total', () => has('stone', 11) || phase.atFurnace(), async () => await collectResource('stone', 11), 3, replan('stone'))

            await runTask('Pico Piedra', () => has('stone_pickaxe', 1) || has('iron_pickaxe', 1), async () => await smartCraft('stone_pickaxe', 1))
            const stonePick = bot.inventory.findInventoryItem(mcData.itemsByName.stone_pickaxe.id)
            if (stonePick) await bot.equip(stonePick, 'hand')
        }

        // --- Horno
        if (!phase.atFurnace()) {
            await runTask('Craftear Horno', () => has('furnace', 1) || isFurnacePlaced(), async () => await smartCraft('furnace', 1))
            await runTask('Colocar Horno', isFurnacePlaced, async () => await placeBlock('furnace'), 2, replan())
        }

        // --- Minería y fundición de hierro
        if (!phase.atIron()) {
            await runTask('Carbón', () => has('coal_ore', 3) || has('coal', 3), async () => await collectResource('coal_ore', 3), 3, replan('coal_ore'))
            await runTask('Hierro', () => has('iron_ore', 3) || has('raw_iron', 3) || has('iron_ingot', 3), async () => await collectResource('iron_ore', 3), 3, replan('iron_ore'))

            await runTask('Lingote de Hierro',
                () => has('iron_ingot', 3), 
                async () => await smeltItem('iron_ingot', 3),
                2, replan()
            )

            await runTask('Recargar Palos', 
                () => has('stick', 2), 
                async () => await smartCraft('stick', 1))

            await runTask('Pico de Hierro', 
                () => has('iron_pickaxe', 1), 
                async () => await smartCraft('iron_pickaxe', 1))
        }

        bot.chat('¡Misión Completa: Tengo el Pico de Hierro!')

    } catch (err) {
        bot.chat(`Proceso detenido: ${err.message}`)
        console.error(err)
    }
}   