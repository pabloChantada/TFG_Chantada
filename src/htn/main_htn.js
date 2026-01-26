import pkg from 'mineflayer-pathfinder'
const { Movements, goals } = pkg
import Vec3 from 'vec3'
import inventoryViewer from 'mineflayer-web-inventory'
import prismarineViewer from 'prismarine-viewer'
const mineflayerViewer = prismarineViewer.mineflayer
import minecraftData from 'minecraft-data'

// Import correcto de las tareas
import { collectResource, smartCraft, placeBlock, smeltItem, clearInventory } from './task.js'

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

// =========================================================
// --- HELPERS DE REPLANIFICACIÓN ---
// =========================================================

async function exploreArea(bot, steps = 2) {
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
async function runTask(bot, name, checkFn, actionFn, retries = 1, replanFn = null) {
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
const replan = (bot, stepsBase = 2) => async (_err, attempt = 0) => {
  await exploreArea(bot, stepsBase + attempt)
  return true
}

// =========================================================
// --- LÓGICA DEL BOT ---
// =========================================================

export async function startHTN(bot) {
    mcData = minecraftData(bot.version)
    
    // inventoryViewer(bot) // Optional: might conflict with existing viewer
    // mineflayerViewer(bot, { port: 3001, firstPerson: true }) // Already handled by Mindcraft
    
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)

    bot.chat('Bot listo. Escribe "craft" para ir a por el hierro.')

    bot.on('chat', async (username, message) => {
        if (username === bot.username) return
        if (message === 'craft') await startFullProgression(bot)
        if (message === 'clear') await clearInventory(bot)
    })
}

// =========================================================
// --- TAREA PRINCIPAL: PROGRESIÓN COMPLETA HASTA HIERRO ---
// =========================================================

async function startFullProgression(bot) {
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
            await runTask(bot, 'Madera', () => has('oak_log', 4), async () => await collectResource(bot, mcData, 'oak_log', 4), 3, replan(bot, 'oak_log'))
            await runTask(bot, 'Tablones', () => has('oak_planks', 12), async () => await smartCraft(bot, mcData, 'oak_planks', 3))
            await runTask(bot, 'Palos', () => has('stick', 4), async () => await smartCraft(bot, mcData, 'stick', 1))

            await runTask(bot, 'Craftear Mesa', () => has('crafting_table', 1) || isTablePlaced(), async () => await smartCraft(bot, mcData, 'crafting_table', 1))
            await runTask(bot, 'Colocar Mesa', isTablePlaced, async () => await placeBlock(bot, mcData, 'crafting_table'), 2, replan(bot))

            await runTask(bot, 'Pico Madera', () => has('wooden_pickaxe', 1) || has('stone_pickaxe', 1), async () => await smartCraft(bot, mcData, 'wooden_pickaxe', 1))
            const woodPick = bot.inventory.findInventoryItem(mcData.itemsByName.wooden_pickaxe.id)
            if (woodPick) await bot.equip(woodPick, 'hand')

            await runTask(bot, 'Piedra Total', () => has('stone', 11) || phase.atFurnace(), async () => await collectResource(bot, mcData, 'stone', 11), 3, replan(bot, 'stone'))

            await runTask(bot, 'Pico Piedra', () => has('stone_pickaxe', 1) || has('iron_pickaxe', 1), async () => await smartCraft(bot, mcData, 'stone_pickaxe', 1))
            const stonePick = bot.inventory.findInventoryItem(mcData.itemsByName.stone_pickaxe.id)
            if (stonePick) await bot.equip(stonePick, 'hand')
        }

        // --- Horno
        if (!phase.atFurnace()) {
            await runTask(bot, 'Craftear Horno', () => has('furnace', 1) || isFurnacePlaced(), async () => await smartCraft(bot, mcData, 'furnace', 1))
            await runTask(bot, 'Colocar Horno', isFurnacePlaced, async () => await placeBlock(bot, mcData, 'furnace'), 2, replan(bot))
        }

        // --- Minería y fundición de hierro
        if (!phase.atIron()) {
            await runTask(bot, 'Carbón', () => has('coal_ore', 3) || has('coal', 3), async () => await collectResource(bot, mcData, 'coal_ore', 3), 3, replan(bot, 'coal_ore'))
            await runTask(bot, 'Hierro', () => has('iron_ore', 3) || has('raw_iron', 3) || has('iron_ingot', 3), async () => await collectResource(bot, mcData, 'iron_ore', 3), 3, replan(bot, 'iron_ore'))

            // Si no tenemos un horno cerca, lo creamos
            try {
                const fs = await import('fs')
                const path = 'my_agent/memory.json'
                let found = false

                if (fs.existsSync(path)) {
                    const positions = JSON.parse(fs.readFileSync(path, 'utf8'))
                    if (positions.furnace) {
                        const pos = positions.furnace
                        const target = new Vec3(pos.x, pos.y, pos.z)
                        
                        if (bot.entity.position.distanceTo(target) <= 32) {
                            bot.chat('Yendo al horno conocido en ' + target)
                            await bot.pathfinder.goto(new goals.GoalNear(target.x, target.y, target.z, 2))
                            found = true
                        } else {
                            bot.chat('El horno conocido está muy lejos. Ignorando memoria.')
                        }
                    }
                }

                if (!found) {
                    bot.chat('No tengo un horno cercano utilizable. Creando uno nuevo.')
                    await runTask(bot, 'Craftear Horno', () => has('furnace', 1) || isFurnacePlaced(), async () => await smartCraft(bot, mcData, 'furnace', 1))
                    await runTask(bot, 'Colocar Horno', isFurnacePlaced, async () => await placeBlock(bot, mcData, 'furnace'), 2, replan(bot))
                }
            } catch (e) {
                console.error('Error reading memory:', e)
            }

            await runTask(bot, 'Lingote de Hierro',
                () => has('iron_ingot', 3), 
                async () => await smeltItem(bot, mcData, 'iron_ingot', 3),
                2, replan(bot)
            )

            await runTask(bot, 'Recargar Palos', 
                () => has('stick', 2), 
                async () => await smartCraft(bot, mcData, 'stick', 1))

            await runTask(bot, 'Pico de Hierro', 
                () => has('iron_pickaxe', 1), 
                async () => await smartCraft(bot, mcData, 'iron_pickaxe', 1))
        }

        bot.chat('¡Misión Completa: Tengo el Pico de Hierro!')

    } catch (err) {
        bot.chat(`Proceso detenido: ${err.message}`)
        console.error(err)
    }
}