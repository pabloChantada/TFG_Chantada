import {
    collectResource,
    getItemNameFromBlock,
    getItemId
} from './primitive_task.js'
import { craftItem } from './tasks/crafting.js'
import { ensureCraftingTable } from './tasks/block_placement.js'
import { obtainWoodType, obtainPlankType } from './tasks/wood.js'

// =========================================================
// --- PROGRESIÓN SIMPLE PARA TESTING ---
// Objetivo: Pico de Piedra
// =========================================================

/**
 * Progresión simple: Madera -> Piedra -> Pico de Piedra
 */
async function runSimpleProgression(bot, mcData, metricsCollector = null) {
    const has = (itemName, count = 1) => {
        const finalName = getItemNameFromBlock(itemName);
        const id = getItemId(mcData, finalName);
        return id ? bot.inventory.count(id) >= count : false;
    };

    const ensurePlanksAndSticks = async (plankType, woodType, minPlanks, minSticks) => {
        const plankId = getItemId(mcData, plankType)
        const currentPlanks = plankId ? bot.inventory.count(plankId) : 0
        const neededPlanks = Math.max(minPlanks - currentPlanks, 0)

        if (neededPlanks > 0) {
            const neededLogs = Math.ceil(neededPlanks / 4)
            if (!has(woodType, neededLogs)) {
                bot.chat(`[MATERIAL] Faltan troncos (${neededLogs}). Recolectando...`)
                await collectResource(bot, mcData, woodType, neededLogs, metricsCollector)
            }

            const craftPlanksAmount = Math.ceil(neededPlanks / 4) * 4
            await craftItem(bot, mcData, plankType, craftPlanksAmount)
        }

        if (!has('stick', minSticks)) {
            // Craftear en múltiplos de 4
            const neededSticks = Math.max(minSticks - (getItemId(mcData, 'stick') ? bot.inventory.count(getItemId(mcData, 'stick')) : 0), 0)
            const craftSticksAmount = Math.ceil(neededSticks / 4) * 4
            await craftItem(bot, mcData, 'stick', craftSticksAmount || 4)
        }
    }

    try {
        bot.chat('[WAIT] Esperando inicio del prismarine viewer...');
        await new Promise(resolve => setTimeout(resolve, 1000));

        // === FASE 1: MADERA ===
        bot.chat('[FASE 1] Iniciando progresión simple...');
        // Obtener madera
        const woodType = await obtainWoodType(bot, mcData);
        const plankType = obtainPlankType(woodType);
        
        if (!has(woodType, 3)) {
            bot.chat(`[1.1] Cortando ${woodType}...`);
            await collectResource(bot, mcData, woodType, 3, metricsCollector);
        } else {
            bot.chat(`[SKIP] Ya tengo ${woodType}`);
        }

        // Craftear tablones si es necesario
        if (!has(plankType, 12)) {
            bot.chat(`[1.2] Crafteando ${plankType}...`);
            const currentPlanks = bot.inventory.count(getItemId(mcData, plankType))
            const neededPlanks = 12 - currentPlanks
            const neededLogs = Math.ceil(neededPlanks / 4)

            if (!has(woodType, neededLogs)) {
                bot.chat(`[1.2] Faltan troncos (${neededLogs}). Recolectando...`)
                await collectResource(bot, mcData, woodType, neededLogs, metricsCollector)
            }

            // Intentar craftear con reintentos simples
            let crafted = false
            for (let i = 0; i < 3 && !crafted; i++) {
                try {
                    await craftItem(bot, mcData, plankType, 12)
                    crafted = true
                } catch (err) {
                    bot.chat(`[1.2] Reintentando tablones (${i + 1}/3)...`)
                    await collectResource(bot, mcData, woodType, 1, metricsCollector)
                }
            }
        }

        // Craftar palos si es necesario
        if (!has('stick', 4)) {
            bot.chat(`[1.3] Crafteando palos...`);
            await craftItem(bot, mcData, 'stick', 4);
        }

        // === FASE 2: MESA DE CRAFTEO ===
        bot.chat('[FASE 2] Colocando mesa de crafteo...');
        
        const tableId = mcData.blocksByName.crafting_table.id;
        const isTablePlaced = () => bot.findBlock({ matching: tableId, maxDistance: 16 }) !== null;
        
        if (!isTablePlaced()) {
            await craftItem(bot, mcData, 'crafting_table', 1);
            const { placeBlock } = await import('./primitive_task.js');
            await placeBlock(bot, mcData, 'crafting_table');
            bot.chat('[OK] Mesa colocada');
        } else {
            bot.chat('[SKIP] Mesa ya existe');
        }

        // === FASE 3: PICO DE MADERA (opcional, para minar piedra) ===
        bot.chat('[FASE 3] Crafteando pico de madera...');

        if (!has('wooden_pickaxe', 1)) {
            // Asegurar mesa cerca y materiales mínimos
            await ensureCraftingTable(bot, mcData, 32)

            // Para pico de madera: 3 tablones + 2 palos
            await ensurePlanksAndSticks(plankType, woodType, 3, 2)

            // Reintentar crafteo de pico si falla por materiales
            let craftedPick = false
            for (let i = 0; i < 3 && !craftedPick; i++) {
                try {
                    await craftItem(bot, mcData, 'wooden_pickaxe', 1)
                    craftedPick = true
                } catch (_err) {
                    bot.chat(`[3.3] Reintentando pico (${i + 1}/3)...`)
                    await ensurePlanksAndSticks(plankType, woodType, 3, 2)
                }
            }

            const woodPick = bot.inventory.findInventoryItem(getItemId(mcData, 'wooden_pickaxe'));
            if (woodPick) await bot.equip(woodPick, 'hand');
        }

        // === FASE 4: PIEDRA ===
        bot.chat('[FASE 4] Minando piedra...');
        
        if (!has('stone', 3)) {
            await collectResource(bot, mcData, 'stone', 3, metricsCollector);
        } else {
            bot.chat('[SKIP] Ya tengo piedra');
        }

        // === FASE 5: PICO DE PIEDRA ===
        bot.chat('[FASE 5] Crafteando pico de piedra...');
        
        if (!has('stone_pickaxe', 1)) {
            await craftItem(bot, mcData, 'stone_pickaxe', 1);
            const stonePick = bot.inventory.findInventoryItem(getItemId(mcData, 'stone_pickaxe'));
            if (stonePick) await bot.equip(stonePick, 'hand');
        }

        // === ÉXITO ===
        if (has('stone_pickaxe', 1)) {
            bot.chat('✓ ¡Progresión completa! Tengo el pico de piedra.');
            return { success: true };
        } else {
            bot.chat('✗ Error: No tengo el pico de piedra');
            return { success: false };
        }

    } catch (err) {
        bot.chat(`[ERROR] ${err.message}`);
        console.error('[SimpleProgression]', err);
        return { success: false };
    }
}

export {
    runSimpleProgression
}
