const SMELT_INPUTS = {
    'coal': 'coal_ore',
    'iron_ingot': 'raw_iron'
}

const DROPS = {
    'stone': 'cobblestone',
    'coal_ore': 'coal',
    'iron_ore': 'raw_iron'
}

/**
 * Obtains the input item name for a given smelting output.
 * For example, "iron_ingot" would return "iron_ore".
 * @param {string} outputName - The name of the smelting output item.
 * @returns {string|null} - The corresponding input item name or null if not found.
 */
function getSmeltInputName(outputName) {
    return SMELT_INPUTS[outputName] || null
}

/**
 * Obtains the item name that would be dropped by mining a given block.
 * For example, "coal_ore" would return "coal".
 * @param {string} blockName - The name of the block.
 * @return {string} - The name of the item dropped by the block.
 */
function getItemNameFromBlock(blockName) {
    return DROPS[blockName] || blockName
}

/**
 * Obtains the name of a biome based on the bot's current position.
 * @param {Bot} bot - The mineflayer bot instance.
 * @param {Object} mcData - The minecraft data for the bot's version.
 * @return {string} - The name of the biome.
 */
function getBiomeName(bot, mcData) {
    const normalizeBiomeName = (name) => {
        if (typeof name !== 'string' || name.length === 0) return null
        return name.startsWith('minecraft:') ? name.slice('minecraft:'.length) : name
    }

    const biomeValue = bot.world.getBiome(bot.entity.position)

    // mineflayer/prismarine puede devolver:
    //  - número (id del bioma)
    //  - objeto { id, name, ... }
    if (typeof biomeValue === 'number') {
        const biomeName = mcData.biomes[biomeValue]?.name
        return normalizeBiomeName(biomeName) || 'plains'
    }

    if (biomeValue && typeof biomeValue === 'object') {
        if (typeof biomeValue.name === 'string' && biomeValue.name.length > 0) {
            return normalizeBiomeName(biomeValue.name) || 'plains'
        }
        if (typeof biomeValue.id === 'number') {
            const biomeName = mcData.biomes[biomeValue.id]?.name
            return normalizeBiomeName(biomeName) || 'plains'
        }
    }

    return 'plains'
}

export {
    getSmeltInputName,
    getItemNameFromBlock,
    getBiomeName
}