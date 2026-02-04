import pkg from 'mineflayer-pathfinder'
const { goals } = pkg
import Vec3 from 'vec3'

import {
    collectResource,
    smartCraft,
    ensureCraftingTable,
    placeBlock,
    smeltItem,
    getItemNameFromBlock,
    exploreRandom
} from '../primitive_task.js'
import { chopTree, obtainWoodType, obtainPlankType } from './wood.js'
import { getCoal, getIron } from './mining.js'

// =========================================================
// --- HELPERS DE REPLANIFICACIÓN ---
// =========================================================

async function exploreArea(bot, steps = 1) {
    // Usa exploreRandom de primitives para exploración más efectiva
    for (let i = 0; i < steps; i++) {
        await exploreRandom(bot, 10 + i * 10)
        await bot.waitForTicks(10)
    }
}

/**
 * runTask admite un replanFn(err, attempt) que, si devuelve true,
 * considera manejado el error y reintenta sin fallar la tarea.
 */
async function runTask(bot, name, checkFn, actionFn, retries = 1, replanFn = null) {
    if (checkFn()) {
        bot.chat(`[SKIP] ${name}`)
        return null
    }

    bot.chat(`[START] ${name}`)
    await bot.trackAction(name)

    for (let i = 0; i <= retries; i++) {
        try {
            const result = await actionFn()
            await bot.waitForTicks(10)

            if (checkFn()) {
                bot.chat(`[OK] ${name}`)
                return result
            }
            throw new Error('Acción finalizada pero condición no cumplida.')
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

const replan = (bot, stepsBase = 2) => async (_err, attempt = 0) => {
    await exploreArea(bot, stepsBase + attempt)
    return true
}

// =========================================================
// --- FASES DE PROGRESIÓN ---
// =========================================================

function createHas(bot, mcData) {
    return (name, count = 1) => {
        const finalName = getItemNameFromBlock(name)
        const id = mcData.itemsByName[finalName]?.id
        return id ? bot.inventory.count(id) >= count : false
    }
}

function createPhaseGuards(bot, mcData, has) {
    const tableId = () => mcData.blocksByName.crafting_table.id
    const isTablePlaced = () => bot.findBlock({ matching: tableId(), maxDistance: 16 }) !== null
    const isFurnacePlaced = () => bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 16 }) !== null

    return {
        isTablePlaced,
        isFurnacePlaced,
        atStone: () => has('stone_pickaxe', 1) || has('iron_pickaxe', 1),
        atFurnace: () => has('furnace', 1) || isFurnacePlaced() || has('iron_pickaxe', 1),
        atIron: () => has('iron_pickaxe', 1)
    }
}

async function woodPhase(bot, mcData, has, guards, metricsCollector = null) {
    if (guards.atStone()) return

    // Madera: tarea crítica, más reintentos
    const woodType = await obtainWoodType(bot, mcData)
    const plankType = obtainPlankType(woodType)
    
    await runTask(
        bot,
        'Madera',
        () => has(woodType, 4),
        async () => chopTree(bot, mcData, 4, metricsCollector),
        5,
        replan(bot, 3)
    )
    
    bot.chat(`Craftear ${plankType} desde ${woodType}`)
    await runTask(bot, 'Tablones', () => has(plankType, 12), async () => smartCraft(bot, mcData, plankType, 4))
    await runTask(bot, 'Palos', () => has('stick', 4), async () => smartCraft(bot, mcData, 'stick', 2))

    await runTask(bot, 'Asegurar Mesa', guards.isTablePlaced, async () => ensureCraftingTable(bot, mcData, 32), 3, replan(bot, 2))

    await runTask(bot, 'Pico Madera', () => has('wooden_pickaxe', 1) || has('stone_pickaxe', 1), async () => smartCraft(bot, mcData, 'wooden_pickaxe', 1), 2)
    const woodPick = bot.inventory.findInventoryItem(mcData.itemsByName.wooden_pickaxe.id)
    if (woodPick) await bot.equip(woodPick, 'hand')

    // Piedra: tarea crítica, más reintentos y exploración más agresiva
    await runTask(
        bot,
        'Piedra Total',
        () => has('stone', 11) || guards.atFurnace(),
        async () => collectResource(bot, mcData, 'stone', 11, metricsCollector),
        5,
        replan(bot, 3)
    )

    await runTask(bot, 'Pico Piedra', () => has('stone_pickaxe', 1) || has('iron_pickaxe', 1), async () => smartCraft(bot, mcData, 'stone_pickaxe', 1), 2)
    const stonePick = bot.inventory.findInventoryItem(mcData.itemsByName.stone_pickaxe.id)
    if (stonePick) await bot.equip(stonePick, 'hand')
}

async function furnacePhase(bot, mcData, has, guards, metricsCollector = null) {
    // Solo se llama cuando ya tenemos materiales para fundir
    if (guards.isFurnacePlaced()) return

    await runTask(bot, 'Craftear Horno', () => has('furnace', 1) || guards.isFurnacePlaced(), async () => smartCraft(bot, mcData, 'furnace', 1))
    await runTask(bot, 'Colocar Horno', guards.isFurnacePlaced, async () => placeBlock(bot, mcData, 'furnace'), 2, replan(bot))
}

async function ensureNearbyFurnace(bot, mcData, has, guards) {
    // Si no tenemos un horno cerca, lo creamos
    try {
        const fs = await import('fs')
        const path = `src/agents/memories/${bot.username}_memory.json`
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
            await runTask(bot, 'Craftear Horno', () => has('furnace', 1) || guards.isFurnacePlaced(), async () => smartCraft(bot, mcData, 'furnace', 1))
            await runTask(bot, 'Colocar Horno', guards.isFurnacePlaced, async () => placeBlock(bot, mcData, 'furnace'), 2, replan(bot))
        }
    } catch (e) {
        console.error('Error reading memory:', e)
    }
}

async function ironPhase(bot, mcData, has, guards, metricsCollector = null) {
    if (guards.atIron()) return

    // 1. Primero minamos carbón y hierro
    await runTask(bot, 'Carbón', () => has('coal', 3), async () => getCoal(bot, mcData, 3), 3, replan(bot, 4))
    await runTask(bot, 'Hierro', () => has('raw_iron', 3) || has('iron_ingot', 3), async () => getIron(bot, mcData, 3), 3, replan(bot, 4))

    // 2. Ahora que tenemos materiales, ponemos el horno
    await ensureNearbyFurnace(bot, mcData, has, guards)

    // 3. Fundimos el hierro
    await runTask(
        bot,
        'Lingote de Hierro',
        () => has('iron_ingot', 3),
        async () => smeltItem(bot, mcData, 'iron_ingot', 3),
        2,
        replan(bot)
    )

    // 4. Crafteamos el pico
    await runTask(bot, 'Recargar Palos', () => has('stick', 2), async () => smartCraft(bot, mcData, 'stick', 1))
    await runTask(bot, 'Pico de Hierro', () => has('iron_pickaxe', 1), async () => smartCraft(bot, mcData, 'iron_pickaxe', 1))
}

// =========================================================
// --- ORQUESTADOR DE PROGRESIÓN ---
// =========================================================

async function runFullProgression(bot, mcData, metricsCollector = null) {
    const has = createHas(bot, mcData)
    const guards = createPhaseGuards(bot, mcData, has)

    await woodPhase(bot, mcData, has, guards, metricsCollector)
    // furnacePhase se llama desde ironPhase cuando ya tenemos materiales
    await ironPhase(bot, mcData, has, guards, metricsCollector)

    bot.chat('¡Misión Completa: Tengo el Pico de Hierro!')
}

export {
    runFullProgression
}
