import pkg from 'mineflayer-pathfinder'
const { goals } = pkg
import Vec3 from 'vec3'
import fs from 'fs'

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

const SMELT_INPUTS = {
    iron_ingot: 'raw_iron',
    gold_ingot: 'raw_gold',
    copper_ingot: 'raw_copper'
}

// =========================================================
// --- PRIMITIVAS ATÓMICAS (HTN) ---
// =========================================================

function getItemNameFromBlock(blockName) {
    // Buscamos el item correspondiente al bloque.
    // Por ejemplo, buscamos "stone" pero miramos cuanto tenemos de "cobblestone".
    return DROPS[blockName] || blockName
}

function getItemId(mcData, itemName) {
    return mcData.itemsByName[itemName]?.id
}

function getBlockId(mcData, blockName) {
    return mcData.blocksByName[blockName]?.id
}

function hasItem(bot, mcData, itemName, count = 1) {
    const id = getItemId(mcData, itemName)
    return id ? bot.inventory.count(id) >= count : false
}

function findNearestBlock(bot, mcData, blockName, maxDistance = 32) {
    const blockId = getBlockId(mcData, blockName)
    if (!blockId) return null
    return bot.findBlock({ matching: blockId, maxDistance })
}

async function moveToBlock(bot, block, range = 3, metricsCollector = null) {
    if (!block) return
    
    metricsCollector?.trackActionStart('move', bot)
    
    const dist = bot.entity.position.distanceTo(block.position)
    if (dist > range) {
        try {
            await bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, range))
            metricsCollector?.trackActionEnd(true, bot)
        } catch (e) {
            // Si el pathfinder falla, intentar acercarse manualmente
            try {
                bot.lookAt(block.position)
                await nudgeForward(bot, 30)
                metricsCollector?.trackActionEnd(true, bot)
            } catch (nudgeError) {
                metricsCollector?.trackActionEnd(false, bot)
            }
        }
    } else {
        metricsCollector?.trackActionEnd(true, bot)
    }
}

async function mineBlock(bot, block, metricsCollector = null) {
    if (!block) return
    
    metricsCollector?.trackActionStart('mine', bot)
    
    try {
        // Primero nos movemos al bloque
        await moveToBlock(bot, block, 4)
        
        // Verificar que el bloque sigue existiendo
        const currentBlock = bot.blockAt(block.position)
        if (!currentBlock || currentBlock.type === 0) {
            metricsCollector?.trackActionEnd(false, bot)
            return
        }
        
        await bot.collectBlock.collect(currentBlock)
        metricsCollector?.trackActionEnd(true, bot)
    } catch (e) {
        // Fallback: cavar manualmente
        try {
            const currentBlock = bot.blockAt(block.position)
            if (currentBlock && bot.canDigBlock(currentBlock)) {
                await bot.dig(currentBlock)
                metricsCollector?.trackActionEnd(true, bot)
            } else {
                metricsCollector?.trackActionEnd(false, bot)
            }
        } catch (digError) {
            metricsCollector?.trackActionEnd(false, bot)
        }
    }
}

async function nudgeForward(bot, ticks = 20) {
    bot.setControlState('forward', true)
    await bot.waitForTicks(ticks)
    bot.setControlState('forward', false)
}

function getCraftingTable(bot, mcData, maxDistance = null) {
    const path = `src/agents/memories/${bot.username}_memory.json`

    if (fs.existsSync(path)) {
        const memory = JSON.parse(fs.readFileSync(path, 'utf8'))
        const pos = memory?.crafting_table
        if (pos) {
            const tablePos = new Vec3(pos.x, pos.y, pos.z)
            if (maxDistance !== null && bot.entity.position.distanceTo(tablePos) > maxDistance) {
                return null
            }

            const block = bot.blockAt(tablePos)
            if (block && block.type === getBlockId(mcData, 'crafting_table')) return block
        }
    }

    const tableId = getBlockId(mcData, 'crafting_table')
    if (!tableId) return null
    return bot.findBlock({ matching: tableId, maxDistance: maxDistance ?? 32 })
}

async function craftItem(bot, item, amount, table = null) {
    // TODO: optimizar para crafteos grandes
    const recipe = bot.recipesFor(item.id, null, amount, table)[0]
    if (!recipe) throw new Error(`No puedo craftear ${item.name} (Faltan materiales?)`)
    await bot.craft(recipe, amount, recipe.requiresTable ? table : null)
}

async function ensureCraftingTable(bot, mcData, maxDistance = 32) {
    let table = getCraftingTable(bot, mcData, maxDistance)
    if (table) return table

    if (!hasItem(bot, mcData, 'crafting_table', 1)) {
        await smartCraft(bot, mcData, 'crafting_table', 1)
    }

    await placeBlock(bot, mcData, 'crafting_table')

    table = getCraftingTable(bot, mcData, maxDistance)
    if (!table) throw new Error('No pude localizar la mesa tras colocarla')
    return table
}

function getSmeltInputName(outputName) {
    return SMELT_INPUTS[outputName] || null
}

function getFurnace(bot, mcData, maxDistance = null) {
    const path = `src/agents/memories/${bot.username}_memory.json`

    if (!fs.existsSync(path)) return null

    const memory = JSON.parse(fs.readFileSync(path, 'utf8'))
    const pos = memory?.furnace
    if (!pos) return null

    const furnacePos = new Vec3(pos.x, pos.y, pos.z)
    // De momento vamos a hacer que no sea por distancia
    if (maxDistance !== null && bot.entity.position.distanceTo(furnacePos) > maxDistance) return null

    const block = bot.blockAt(furnacePos)
    return block && block.type === getBlockId(mcData, 'furnace') ? block : null
}

async function exploreRandom(bot, distance = 30) {
    // Exploración en dirección aleatoria
    const angle = Math.random() * 2 * Math.PI
    const dx = Math.cos(angle) * distance
    const dz = Math.sin(angle) * distance
    const target = bot.entity.position.offset(dx, 0, dz)
    
    try {
        await bot.pathfinder.goto(new goals.GoalNear(target.x, target.y, target.z, 5))
    } catch (e) {
        // Si falla el pathfinding, caminar en esa dirección
        await bot.lookAt(target)
        await nudgeForward(bot, 40)
    }
}

async function collectResource(bot, mcData, blockName, count, metricsCollector = null) {
    const itemName = getItemNameFromBlock(blockName)
    const itemId = getItemId(mcData, itemName)
    if (!itemId) throw new Error(`Item inválido: ${itemName}`)

    if (hasItem(bot, mcData, itemName, count)) {
        bot.chat(`[SKIP] Ya tengo suficiente ${blockName}`)
        return
    }

    bot.chat(`Buscando ${count}x ${blockName}...`)
    await bot.waitForTicks(10)

    let attempts = 0
    let noBlockStreak = 0
    const MAX_ATTEMPTS = 200
    const EXPLORE_AFTER = 5 // Explorar después de 5 fallos consecutivos

    while (!hasItem(bot, mcData, itemName, count)) {
        if (attempts > MAX_ATTEMPTS) {
            throw new Error(`No consigo encontrar suficiente ${blockName}`)
        }

        // Buscar bloque en rango creciente
        let block = findNearestBlock(bot, mcData, blockName, 32)
        if (!block) block = findNearestBlock(bot, mcData, blockName, 64)
        if (!block) block = findNearestBlock(bot, mcData, blockName, 128)
        
        if (!block) {
            noBlockStreak++
            if (noBlockStreak >= EXPLORE_AFTER) {
                bot.chat(`No veo ${blockName}, explorando...`)
                await exploreRandom(bot, 40)
                noBlockStreak = 0
            } else {
                await nudgeForward(bot, 25)
            }
            attempts++
            continue
        }

        noBlockStreak = 0
        
        try {
            await mineBlock(bot, block, metricsCollector)
        } catch (e) {
            // Ignorar errores de minado, intentar con otro bloque
            attempts++
            continue
        }
        
        attempts++
    }
    
    const finalCount = bot.inventory.count(itemId)
    bot.chat(`Conseguidos ${finalCount}x ${itemName}`)
}

async function smartCraft(bot, mcData, itemName, amount) {
    const item = mcData.itemsByName[itemName]
    if (!item) throw new Error(`Item inválido: ${itemName}`)

    if (hasItem(bot, mcData, itemName, amount)) {
        bot.chat(`[SKIP] Ya tengo suficiente ${itemName}`)
        return
    }

    // Primero verificamos si necesita mesa
    let table = getCraftingTable(bot, mcData, 64)
    
    // La receta tendria que ser recursiva para crafteos grandes
    const recipe = bot.recipesFor(item.id, null, amount, table)[0]
    if (!recipe) throw new Error(`No puedo craftear ${itemName} (Faltan materiales?)`)

    if (recipe.requiresTable) {
        if (!table) {
            // Crear mesa si no existe
            const tableItem = mcData.itemsByName.crafting_table
            if (!tableItem) throw new Error('No existe crafting_table en mcData')
            await craftItem(bot, tableItem, 1, null)
            await placeBlock(bot, mcData, 'crafting_table')
            table = getCraftingTable(bot, mcData, 32)
            if (!table) throw new Error('No pude localizar la mesa tras colocarla')
        }
        
        // IMPORTANTE: Moverse CERCA de la mesa (rango 3 para poder interactuar)
        bot.chat(`Yendo a la mesa de crafteo...`)
        await moveToBlock(bot, table, 3)
        await bot.waitForTicks(10)
        
        // Re-obtener la mesa después de moverse (por si cambió el chunk)
        table = getCraftingTable(bot, mcData, 8)
        if (!table) throw new Error('Perdí la mesa de crafteo')
    }

    await craftItem(bot, item, amount, table)
}

async function placeBlock(bot, mcData, blockName) {
    const itemId = getItemId(mcData, blockName)
    const item = bot.inventory.findInventoryItem(itemId)
    
    if (!item) throw new Error(`No tengo ${blockName} para colocar.`)

    bot.pathfinder.setGoal(null)
    await bot.waitForTicks(5)

    const yaw = bot.entity.yaw
    const targetPos = bot.entity.position.offset(-Math.sin(yaw) * 2, 0, -Math.cos(yaw) * 2).floored()
    const groundPos = targetPos.offset(0, -1, 0)
    
    let memoryPositions = fs.existsSync(`src/agents/memories/${bot.username}_memory.json`) ?
        JSON.parse(fs.readFileSync(`src/agents/memories/${bot.username}_memory.json`, 'utf8')) : {}

    const targetInMemory = Object.values(memoryPositions || {}).some(pos =>
        pos && pos.x === targetPos.x && pos.y === targetPos.y && pos.z === targetPos.z
    )

    if (targetInMemory) {
        console.log(`Ya hay un bloque en la posición objetivo: ${JSON.stringify(targetPos)}`)
        await nudgeForward(bot, 10)
    }
    const groundBlock = bot.blockAt(groundPos)
    const targetBlock = bot.blockAt(targetPos)

    if (!groundBlock || groundBlock.boundingBox !== 'block') {
         // Fix: Moverse para encontrar suelo
        await nudgeForward(bot, 10)
        throw new Error('Suelo inválido, me he movido.')
    }
    
    if (targetBlock.type !== 0) await bot.dig(targetBlock)

    await bot.equip(item, 'hand')
    let lookBlock = await bot.lookAt(groundBlock.position.offset(0.5, 1, 0.5), true)
    
    // Esto deberia evitar que rompa mesas/hornos para colocar mesas/horno en su posicion
    if (lookBlock === 'crafting_table' || lookBlock === 'furnace') {
        await bot.nudgeForward(5)
    }

    await bot.placeBlock(groundBlock, new Vec3(0, 1, 0))

    // Si es una mesa o un horno, guardar posición
    const path = `src/agents/memories/${bot.username}_memory.json`
    
    console.log(`[placeBlock] Guardando posición de ${blockName} para usuario: ${bot.username}`)
    console.log(`[placeBlock] Ruta: ${path}`)
    console.log(`[placeBlock] Posición: ${JSON.stringify(targetPos)}`)
    
    // Crear directorio si no existe
    const dir = path.substring(0, path.lastIndexOf('/'))
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
    }
    
    let positions = {}
    if (fs.existsSync(path)) {
        positions = JSON.parse(fs.readFileSync(path, 'utf8'))
    }

    if (blockName === 'crafting_table') {
        positions.crafting_table = targetPos
        console.log(`[placeBlock] Mesa guardada`)
    }
    if (blockName === 'furnace') {
        positions.furnace = targetPos
        console.log(`[placeBlock] Horno guardado`)
    }

    fs.writeFileSync(path, JSON.stringify(positions, null, 2))
    console.log(`[placeBlock] Archivo guardado: ${path}`)
}

/**
 * Función de Fundición (Batch Processing)
 */
async function smeltItem(bot, mcData, itemName, amountNeeded) {
    const item = mcData.itemsByName[itemName]
    if (!item) throw new Error(`El item ${itemName} no existe en los datos.`)

    if (hasItem(bot, mcData, itemName, amountNeeded)) {
        bot.chat(`[SKIP] Ya tengo suficiente ${itemName}`)
        return
    }

    const furnaceBlock = getFurnace(bot, mcData, null)
    if (!furnaceBlock) throw new Error('No encuentro el horno.')

    const inputName = getSmeltInputName(itemName)
    if (!inputName) throw new Error(`No sé qué material fundir para ${itemName}`)

    const inputItem = mcData.itemsByName[inputName]
    if (!inputItem) throw new Error(`El material ${inputName} no existe en los datos.`)

    const rawMaterial = bot.inventory.findInventoryItem(inputItem.id)
    const fuel = bot.inventory.findInventoryItem(mcData.itemsByName.coal.id) // TODO: generalizar combustibles

    if (!rawMaterial || !fuel) throw new Error('Faltan materiales para fundir.')

    await moveToBlock(bot, furnaceBlock, 32)

    const furnace = await bot.openFurnace(furnaceBlock)

    await furnace.putFuel(fuel.type, null, fuel.count)
    await furnace.putInput(rawMaterial.type, null, rawMaterial.count)

    bot.chat('Horno encendido, esperando fundición...')

    let retries = 0
    while (retries < 5) { // Max 400 segundos
        await new Promise(resolve => setTimeout(resolve, 10000))

        try {
            await furnace.takeOutput()
        } catch (e) {
            // A veces falla si está vacío, ignoramos
        }

        const currentOutput = bot.inventory.count(item.id)
        if (currentOutput >= amountNeeded) {
            bot.chat(`¡Conseguidos ${currentOutput} ${itemName}!`)
            break
        }
        retries++
    }

    furnace.close()
}

async function clearInventory(bot) {
    const items = bot.inventory.items()
    for (const item of items) await bot.toss(item.type, null, item.count)
}

export {
    getItemNameFromBlock,
    getItemId,
    getBlockId,
    hasItem,
    findNearestBlock,
    moveToBlock,
    mineBlock,
    nudgeForward,
    exploreRandom,
    getCraftingTable,
    craftItem,
    ensureCraftingTable,
    getSmeltInputName,
    getFurnace,
    collectResource,
    smartCraft,
    placeBlock,
    smeltItem,
    clearInventory
}