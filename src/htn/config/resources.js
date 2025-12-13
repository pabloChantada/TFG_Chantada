const ORE_CONFIGS = {
  coal_ore: { min: 0, max: 320, opt: 95 },
  iron_ore: { min: -64, max: 256, opt: 16 },
  gold_ore: { min: -64, max: 32, opt: -16 },
  diamond_ore: { min: -64, max: 16, opt: -59 },
}

const HARVEST_MAP = {
  oak_log: 'oak_log',
  dirt: 'dirt',
  cobblestone: 'stone',
  raw_iron: 'iron_ore',
  raw_gold: 'gold_ore',
  diamond: 'diamond_ore',
  coal: 'coal_ore',
  iron_ore: 'iron_ore',
}

const SMELT_MAP = {
  iron_ingot: 'raw_iron',
  gold_ingot: 'raw_gold',
}

const REQUIRED_TOOL = {
  stone: 'wooden_pickaxe',
  iron_ore: 'stone_pickaxe',
  gold_ore: 'iron_pickaxe',
  diamond_ore: 'iron_pickaxe',
}

module.exports = {
  ORE_CONFIGS,
  HARVEST_MAP,
  SMELT_MAP,
  REQUIRED_TOOL,
}