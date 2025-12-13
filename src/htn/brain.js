const { HARVEST_MAP, SMELT_MAP } = require('./config/resources')
const makeTasks = require('./tasks')
const makePrimitives = require('./primitives')

class Brain {
  constructor(bot, mcData) {
    this.bot = bot
    this.mcData = mcData
    this.primitives = makePrimitives(this)
    this.tasks = makeTasks(this)
  }

  async obtainItem(itemName, count) {
    console.log(`[Meta] Obtener ${count} de ${itemName}`)
    const item = this.mcData.itemsByName[itemName]
    if (!item) throw new Error(`Item desconocido: ${itemName}`)

    const itemId = item.id
    const currentCount = this.bot.inventory.count(itemId)
    if (currentCount >= count) {
      console.log(`[Check] Ya tengo suficientes ${itemName}`)
      return true
    }

    const needed = count - currentCount

    if (HARVEST_MAP[itemName]) {
      const blockToMine = HARVEST_MAP[itemName]
      await this.primitives.mineBlock(blockToMine, needed)
    } else if (SMELT_MAP[itemName]) {
      const rawItem = SMELT_MAP[itemName]
      await this.tasks.smeltItem(rawItem, itemName, needed)
    } else {
      await this.tasks.craftItem(itemName, needed)
    }

    return this.bot.inventory.count(itemId) >= count
  }

  async obtainBlock(blockName, count) {
    console.log(`[Meta] Obtener ${count} de bloque ${blockName}`)
    const block = this.mcData.blocksByName[blockName]
    if (!block) throw new Error(`Bloque desconocido: ${blockName}`)

    const blockId = block.id
    const currentCount = this.bot.inventory.count(blockId)
    if (currentCount >= count) {
      console.log(`[Check] Ya tengo suficientes bloques de ${blockName}`)
      return true
    }

    const needed = count - currentCount
    await this.primitives.mineBlock(blockName, needed)

    return this.bot.inventory.count(blockId) >= count
  }
}

module.exports = Brain