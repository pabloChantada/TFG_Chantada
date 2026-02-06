import { collectResource } from '../primitive_task.js'
import { getBiomeName } from '../utils.js'

// Some other types may be needed, for now we use these
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

// In overworld we only need logs. But also include stem, hyphae and wood for nether biomes
const WOOD_SUFFIXES = ['_log', '_stem', '_hyphae', '_wood'];

// Log -> Planks
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

/**
 * =========================================================
 * ============== WOOD FUNCTIONS ===========================
 * =========================================================
 */


/**
 * Obtains the wood type based on the biome or nearby blocks.
 * If the biome is not mapped, it scans nearby blocks for any wood type.
 * If no wood is found, disconnect.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @returns {Promise<string>} - The wood type to use (e.g., "oak_log").
 */
async function obtainWoodType(bot, mcData) {
    // Obtain biome name and map to wood type
    const biome = getBiomeName(bot, mcData);
    let woodType = biomeToWoodType[biome];

    if (!woodType) {
        console.warn(`[Wood.js] Biome "${biome}" not mapped. Scanning surroundings...`);
        
        // Check nearby block to obtain the wood type
        const nearbyWood = bot.findBlock({
            matching: (block) => {
                const name = block.name;
                // Find any block that ends with a wood suffix
                return WOOD_SUFFIXES.some(suffix => name.endsWith(suffix));
            },
            maxDistance: 64
        });

        if (nearbyWood) {
            woodType = nearbyWood.name;
        } else {
            // Final fallback: oak 
            console.warn(`[Wood.js] No wood found nearby. Disconnecting.`);
            bot.end();
            throw new Error(`No wood found nearby`);
        }
    }
    return woodType;
}

/**
 * Maps a wood type to its corresponding plank type.
 * If no mapping exists, disconnects the bot.
 * @param {string} woodType - The wood type (e.g., "oak_log").
 * @returns {string} - The corresponding plank type (e.g., "oak_planks").
 */
async function obtainPlankType(bot, mcData, woodType, metricsCollector = null) {
    let plankType = logToPlank[woodType];
    if (!plankType) {
        console.warn(`[Wood.js] No plank mapping for "${woodType}"`);
        metricsCollector?.recordError(`No plank mapping for "${woodType}"`);
        await metricsCollector?.export(bot);
        bot.end();
        throw new Error(`No plank mapping for "${woodType}"`);
    }
    return plankType;
}

/**
 * Chops the specified amount of logs of the given wood type.
 * Detects biome, obtains correct wood type, and collects logs.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @param {number} logs - The number of logs to chop.
 * @param {Object} metricsCollector - The metrics collector instance (optional).
 * @return {Promise<number>} - The final count of logs obtained.
 */
async function chop(bot, mcData, logs = 4, metricsCollector = null) {
    const woodType = await obtainWoodType(bot, mcData);
    console.log(`[Wood.js] Detected biome. Searching for: ${woodType}`);
    const finalCount = await collectResource(bot, mcData, woodType, logs);
    return finalCount;
}

export {
    chop,
    obtainWoodType,
    obtainPlankType
}