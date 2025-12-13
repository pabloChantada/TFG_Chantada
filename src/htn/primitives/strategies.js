const { GoalBlock } = require('mineflayer-pathfinder')
const { ORE_CONFIGS } = require('../config/resources')

module.exports = (brain) => {
  const bot = brain.bot
  const mcData = brain.mcData

  const findVisible = async (blockName) => {
    const ids = [mcData.blocksByName[blockName].id]
    if (mcData.blocksByName[`deepslate_${blockName}`]) {
      ids.push(mcData.blocksByName[`deepslate_${blockName}`].id)
    }
    return bot.findBlock({ maxDistance: 32, matching: ids })
  }

  const findNaturalOrStripMine = async (blockName) => {
    const visible = await findVisible(blockName)
    if (visible) return visible

    const oreConfig = ORE_CONFIGS[blockName]
    const optHeight = oreConfig.opt
    const currentY = Math.floor(bot.entity.position.y)

    if (Math.abs(currentY - optHeight) > 5) {
      console.log(`Bajando a capa ${optHeight}...`)
      const goal = new GoalBlock(bot.entity.position.x, optHeight, bot.entity.position.z)
      await bot.pathfinder.goto(goal)
    } else {
      const targetPos = bot.entity.position.offset(1, 0, 0)
      const blockInFront = bot.blockAt(targetPos)
      if (blockInFront && blockInFront.diggable && blockInFront.name !== 'bedrock') return blockInFront

      const goal = new GoalBlock(bot.entity.position.x + 2, bot.entity.position.y, bot.entity.position.z)
      await bot.pathfinder.goto(goal)
    }
    return null
  }

  return {
    findVisible,
    findNaturalOrStripMine,
    isOre: (blockName) => Boolean(ORE_CONFIGS[blockName]),
  }
}