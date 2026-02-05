import {
    findNearestBlock,
    mineBlock,
    hasItem,
    getItemNameFromBlock,
    exploreRandom,
} from '../primitive_task.js'

// import { ensureCraftingTable } from './wood.js'

// =========================================================
// --- MINERÍA OMNISCIENTE ---
// Usa findBlock con rangos amplios para localizar minerales
// directamente sin necesidad de heurísticas de excavación.
// =========================================================

const ORE_SEARCH_RADIUS = 128  // Radio de búsqueda para minerales

/**
 * Busca y mina un mineral específico usando findBlock.
 * Explora si no encuentra el mineral en el rango inicial.
 */
async function mineOre(bot, mcData, oreName, count) {
    const itemName = getItemNameFromBlock(oreName)
    
    if (hasItem(bot, mcData, itemName, count)) {
        bot.chat(`[SKIP] Ya tengo suficiente ${itemName}`)
        return
    }

    bot.chat(`Buscando ${count}x ${oreName}...`)
    
    let attempts = 0
    const MAX_ATTEMPTS = 50
    
    while (!hasItem(bot, mcData, itemName, count) && attempts < MAX_ATTEMPTS) {
        // Buscar mineral en rango creciente
        let ore = findNearestBlock(bot, mcData, oreName, 32)
        if (!ore) ore = findNearestBlock(bot, mcData, oreName, 64)
        if (!ore) ore = findNearestBlock(bot, mcData, oreName, ORE_SEARCH_RADIUS)
        
        if (ore) {
            try {
                await mineBlock(bot, ore)
            } catch (e) {
                // Ignorar errores de minado, buscar otro
                attempts++
                continue
            }
        } else {
            // No hay mineral visible, explorar
            bot.chat(`No veo ${oreName}, explorando...`)
            await exploreRandom(bot, 50)
        }
        
        attempts++
    }
    
    if (!hasItem(bot, mcData, itemName, count)) {
        throw new Error(`No consigo encontrar suficiente ${oreName}`)
    }
    
    const finalCount = bot.inventory.count(mcData.itemsByName[itemName]?.id || 0)
    bot.chat(`Conseguidos ${finalCount}x ${itemName}`)
}

/**
 * Obtiene carbón (coal_ore → coal)
 */
async function getCoal(bot, mcData, count = 3) {
    await mineOre(bot, mcData, 'coal_ore', count)
}

/**
 * Obtiene hierro crudo (iron_ore → raw_iron)
 */
async function getIron(bot, mcData, count = 3) {
    await mineOre(bot, mcData, 'iron_ore', count)
}

export {
    mineOre,
    getCoal,
    getIron
}


// =========================================================
// --- CÓDIGO LEGACY (STRAIGHT MINING) - COMENTADO ---
// =========================================================
/*

const DEFAULT_LINE_LENGTH = 48

const ORE_DEPTHS = {
    coal_ore: 35,
    iron_ore: 16
}

function getTargetDepth(blockName) {
    return ORE_DEPTHS[blockName] ?? 16
}

async function descendToDepth(bot, targetY) {
    const pos = bot.entity.position
    const goal = new goals.GoalNear(pos.x, targetY, pos.z, 1)
    await bot.pathfinder.goto(goal)
}

/**
 * Devuelve un paso en una dirección cardinal (N, S, E u O) según la orientación del bot.
 * Redondea el ángulo para obtener un vector simple y, si no hay dirección válida, usa (1, 0) por defecto.
 * Si el bot mira al norte (yaw ≈ 0), dx=0, dz=-1 y el bot avanza en Z negativa.
 * @param {object} bot - El bot que contiene su orientación actual.
 * @returns {{dx: number, dz: number}} Un vector de paso con componentes en X y Z.
 */

/*
function getCardinalStep(bot) {
    const yaw = bot.entity.yaw
    const dx = Math.round(-Math.sin(yaw))
    const dz = Math.round(-Math.cos(yaw))
    return { dx: dx === 0 && dz === 0 ? 1 : dx, dz: dx === 0 && dz === 0 ? 0 : dz }
}

async function mineForwardStep(bot, mcData) {
    const { dx, dz } = getCardinalStep(bot)
    const nextPos = bot.entity.position.offset(dx, 0, dz).floored()

    const targetBlock = bot.blockAt(nextPos)
    const headBlock = bot.blockAt(nextPos.offset(0, 1, 0))

    if (headBlock && headBlock.type !== 0) await bot.dig(headBlock)
    if (targetBlock && targetBlock.type !== 0) await bot.dig(targetBlock)

    await bot.pathfinder.goto(new goals.GoalBlock(nextPos.x, nextPos.y, nextPos.z))
}

async function mineStraightLine(bot, mcData, length = DEFAULT_LINE_LENGTH, watchBlockName = null) {
    for (let i = 0; i < length; i++) {
        if (watchBlockName) {
            const ore = findNearestBlock(bot, mcData, watchBlockName, 4)
            if (ore) await mineBlock(bot, ore)
        }

        await mineForwardStep(bot, mcData)
    }
}

async function rotateYaw(bot, radians) {
    await bot.look(bot.entity.yaw + radians, bot.entity.pitch, true)
}

async function straightMineForResource(bot, mcData, blockName, count, options = {}) {
    const lineLength = options.lineLength ?? DEFAULT_LINE_LENGTH
    const targetY = options.targetY ?? getTargetDepth(blockName)
    const maxPasses = options.maxPasses ?? 3

    const itemName = getItemNameFromBlock(blockName)
    if (hasItem(bot, mcData, itemName, count)) return

    await ensureCraftingTable(bot, mcData, 32)
    await descendToDepth(bot, targetY)

    let passes = 0
    while (!hasItem(bot, mcData, itemName, count) && passes < maxPasses) {
        await mineStraightLine(bot, mcData, lineLength, blockName)
        await nudgeForward(bot, 5)
        await rotateYaw(bot, Math.PI / 2)
        passes++
    }
}

async function getCoal(bot, mcData, count = 3, options = {}) {
    await straightMineForResource(bot, mcData, 'coal_ore', count, {
        lineLength: options.lineLength ?? DEFAULT_LINE_LENGTH,
        targetY: options.targetY ?? ORE_DEPTHS.coal_ore,
        maxPasses: options.maxPasses ?? 3
    })
}

async function getIron(bot, mcData, count = 3, options = {}) {
    await straightMineForResource(bot, mcData, 'iron_ore', count, {
        lineLength: options.lineLength ?? DEFAULT_LINE_LENGTH,
        targetY: options.targetY ?? ORE_DEPTHS.iron_ore,
        maxPasses: options.maxPasses ?? 3
    })
}

export {
    descendToDepth,
    mineStraightLine,
    straightMineForResource,
    getCoal,
    getIron
}
*/