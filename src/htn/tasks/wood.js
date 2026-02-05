import { collectResource, ensureCraftingTable } from '../primitive_task.js'

// =========================================================
// --- TAREAS DE NIVEL MEDIO: MADERA DINÁMICA ---
// =========================================================

// Mapeo extendido y más preciso
const biomeToWoodType = {
    'forest': 'oak_log',
    'birch_forest': 'birch_log',
    'dark_forest': 'dark_oak_log',
    'jungle': 'jungle_log',
    'spruce_taiga': 'spruce_log',
    'taiga': 'spruce_log',
    'old_growth_pine_taiga': 'spruce_log',
    'old_growth_spruce_taiga': 'spruce_log',
    'acacia_savanna': 'acacia_log',
    'savanna': 'acacia_log',
    'warped_forest': 'warped_stem',
    'crimson_forest': 'crimson_stem',
    'mangrove_swamp': 'mangrove_log',
    'cherry_grove': 'cherry_log',
    'snowy_taiga': 'spruce_log',
    'flower_forest': 'oak_log',
    'meadow': 'oak_log'
}

// Lista de sufijos de bloques de madera válidos
const WOOD_SUFFIXES = ['_log', '_stem', '_hyphae', '_wood'];

async function chopTree(bot, mcData, logs = 4, metricsCollector = null) {
    const woodType = await obtainWoodType(bot, mcData);
    console.log(`[Wood.js] Bioma detectado. Buscando: ${woodType}`);
    await collectResource(bot, mcData, woodType, logs, metricsCollector);
}

function getBiomeName(bot, mcData) {
    const biomeId = bot.world.getBiome(bot.entity.position);
    return mcData.biomes[biomeId]?.name || 'plains';
}

/**
 * Determina qué madera recolectar basándose en el bioma 
 * o en lo que hay alrededor si el bioma es desconocido.
 */
async function obtainWoodType(bot, mcData) {
    const biome = getBiomeName(bot, mcData);
    let woodType = biomeToWoodType[biome];

    // Si el bioma no está en el mapa, intentamos detectar qué madera hay cerca físicamente
    if (!woodType) {
        console.warn(`[Wood.js] Bioma "${biome}" no mapeado. Escaneando alrededores...`);
        
        // Buscamos bloques que coincidan con IDs de logs en mcData
        const nearbyWood = bot.findBlock({
            matching: (block) => {
                const name = block.name;
                return WOOD_SUFFIXES.some(suffix => name.endsWith(suffix));
            },
            maxDistance: 32
        });

        if (nearbyWood) {
            woodType = nearbyWood.name;
        } else {
            // Fallback final: roble (el más común)
            console.warn(`[Wood.js] No se encontró madera cerca. Usando roble por defecto.`);
            woodType = 'oak_log';
        }
    }

    return woodType;
}

/**
 * Convierte un tipo de madera (log/stem) a su tipo de planks correspondiente
 */
function obtainPlankType(woodType) {
    // Mapeo de log/stem a planks
    const logToPlank = {
        'oak_log': 'oak_planks',
        'birch_log': 'birch_planks',
        'dark_oak_log': 'dark_oak_planks',
        'spruce_log': 'spruce_planks',
        'jungle_log': 'jungle_planks',
        'acacia_log': 'acacia_planks',
        'mangrove_log': 'mangrove_planks',
        'cherry_log': 'cherry_planks',
        'warped_stem': 'warped_planks',
        'crimson_stem': 'crimson_planks'
    };

    return logToPlank[woodType] || 'oak_planks';
}

export {
    chopTree,
    obtainWoodType,
    obtainPlankType
}