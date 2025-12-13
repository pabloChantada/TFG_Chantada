const { goals: { GoalBlock } } = require('mineflayer-pathfinder')
const { REQUIRED_TOOL } = require('../config/resources')

module.exports = (brain, primitives) => async function mineBlock(blockName, count) {
  const bot = brain.bot

  if (REQUIRED_TOOL[blockName]) {
    const toolName = REQUIRED_TOOL[blockName]
    if (!bot.inventory.items().some((i) => i.name.includes(toolName))) {
      console.log(`!!! Necesito herramienta: ${toolName}`)
      await brain.obtainItem(toolName, 1)
    }
    const toolItem = bot.inventory.items().find((i) => i.name === toolName)
    if (toolItem) await bot.equip(toolItem, 'hand')
  }

  let collected = 0
  while (collected < count) {
    let targetBlock = null
    if (primitives.strategies.isOre(blockName)) {
      targetBlock = await primitives.strategies.findNaturalOrStripMine(blockName)
    } else {
      targetBlock = await primitives.strategies.findVisible(blockName)
    }

    if (!targetBlock) {
      await bot.waitForTicks(20)
      continue
    }

    try {
      const goal = new GoalBlock(targetBlock.position.x, targetBlock.position.y, targetBlock.position.z)
      await bot.pathfinder.goto(goal)

      if (REQUIRED_TOOL[blockName]) {
        const toolItem = bot.inventory.items().find((i) => i.name === REQUIRED_TOOL[blockName])
        if (toolItem) await bot.equip(toolItem, 'hand')
      }

      await bot.collectBlock.collect(targetBlock)
      collected++
    } catch (err) {
      console.log(`Fallo al minar: ${err.message}`)
      await bot.waitForTicks(10)
    }
  }
}