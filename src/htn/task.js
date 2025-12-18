// ==========================================================
// --- FUNCIONES AUXILIARES ---
// =========================================================


async function collectResource(blockName, count) {
    const blockId = mcData.blocksByName[blockName].id
    const itemName = DROPS[blockName] || blockName
    const itemId = mcData.itemsByName[itemName].id

    if (bot.inventory.count(itemId) >= count) {
        bot.chat(`[SKIP] Ya tengo suficiente ${blockName}`)
        return
    }

    // Pequeña espera inicial para asegurar estado
    await bot.waitForTicks(10)

    let safety = 0
    while (bot.inventory.count(itemId) < count) {
        if (safety > 100) throw new Error(`No consigo encontrar suficiente ${blockName}`)
        
        const block = bot.findBlock({ matching: blockId, maxDistance: 32 })
        if (!block) {
            // Si no encuentra, moverse un poco al azar puede ayudar a cargar chunks
            bot.setControlState('forward', true)
            await bot.waitForTicks(20)
            bot.setControlState('forward', false)
            throw new Error(`No encuentro ${blockName} cerca.`)
        }
        
        await bot.collectBlock.collect(block)
        safety++
    }
}

async function smartCraft(itemName, amount) {
    const item = mcData.itemsByName[itemName]
    const craftingTable = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 32 })

    if (bot.inventory.count(item.id) >= amount) {
        bot.chat(`[SKIP] Ya tengo suficiente ${itemName}`)
        return
    }

    const recipe = bot.recipesFor(item.id, null, amount, craftingTable)[0]
    if (!recipe) throw new Error(`No puedo craftear ${itemName} (Faltan materiales?)`)

    if (recipe.requiresTable) {
        if (!craftingTable) throw new Error(`Necesito mesa para ${itemName}`)
        if (bot.entity.position.distanceTo(craftingTable.position) > 4) {
            await bot.pathfinder.goto(new goals.GoalLookAtBlock(craftingTable.position, bot.world))
        }
    }

    await bot.craft(recipe, amount, craftingTable)
}

async function placeBlock(blockName) {
    const itemId = mcData.itemsByName[blockName].id
    const item = bot.inventory.findInventoryItem(itemId)
    
    if (!item) throw new Error(`No tengo ${blockName} para colocar.`)

    bot.pathfinder.setGoal(null)
    await bot.waitForTicks(5)

    const yaw = bot.entity.yaw
    const targetPos = bot.entity.position.offset(-Math.sin(yaw) * 2, 0, -Math.cos(yaw) * 2).floored()
    const groundPos = targetPos.offset(0, -1, 0)
    
    const groundBlock = bot.blockAt(groundPos)
    const targetBlock = bot.blockAt(targetPos)

    if (!groundBlock || groundBlock.boundingBox !== 'block') {
         // Fix: Moverse para encontrar suelo
        bot.setControlState('forward', true)
        await bot.waitForTicks(10)
        bot.setControlState('forward', false)
        throw new Error('Suelo inválido, me he movido.')
    }
    
    if (targetBlock.type !== 0) await bot.dig(targetBlock)

    await bot.equip(item, 'hand')
    await bot.lookAt(groundBlock.position.offset(0.5, 1, 0.5), true)
    await bot.placeBlock(groundBlock, new Vec3(0, 1, 0))
}

/**
 * Función de Fundición (Batch Processing)
 */
async function smeltItem(itemName, amountNeeded) {
    const item = mcData.itemsByName[itemName];
    if (!item) throw new Error(`El item ${itemName} no existe en los datos.`);

    if (bot.inventory.count(item.id) >= amountNeeded) {
        bot.chat(`[SKIP] Ya tengo suficiente ${itemName}`);
        return;
    }

    const furnaceBlock = bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 16 });
    if (!furnaceBlock) throw new Error('No encuentro el horno.');

    const rawMaterial = bot.inventory.findInventoryItem(item.id);
    const fuel = bot.inventory.findInventoryItem(mcData.itemsByName.coal.id); // Puedes hacer esto más general si usas otros combustibles

    if (!rawMaterial || !fuel) throw new Error('Faltan materiales para fundir.');

    if (bot.entity.position.distanceTo(furnaceBlock.position) > 3) {
        await bot.pathfinder.goto(new goals.GoalLookAtBlock(furnaceBlock.position, bot.world));
    }

    const furnace = await bot.openFurnace(furnaceBlock);

    // Poner TODO el combustible y TODO el material
    await furnace.putFuel(fuel.type, null, fuel.count);
    await furnace.putInput(rawMaterial.type, null, rawMaterial.count);

    bot.chat('Horno encendido, esperando fundición...');

    // Bucle de espera activa hasta tener la cantidad deseada
    // Chequeamos cada 10 segundos
    let retries = 0;
    while (retries < 40) { // Max 400 segundos
        await new Promise(resolve => setTimeout(resolve, 10000));

        // Intentar sacar lo que haya
        try {
            await furnace.takeOutput();
        } catch (e) {
            // A veces falla si está vacío, ignoramos
        }

        const currentOutput = bot.inventory.count(item.id);
        if (currentOutput >= amountNeeded) {
            bot.chat(`¡Conseguidos ${currentOutput} ${itemName}!`);
            break;
        }
        retries++;
    }

    furnace.close();
}

async function clearInventory() {
    const items = bot.inventory.items()
    for (const item of items) await bot.toss(item.type, null, item.count)
}

module.exports = {
    collectResource,
    smartCraft,
    placeBlock,
    smeltItem,
    clearInventory
}