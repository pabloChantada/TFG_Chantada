module.exports = (brain) => async function smeltItem(rawItemName, resultItemName, count) {
  console.log(`[Smelt] Planificando fundición de ${count} ${rawItemName}`)
  const bot = brain.bot
  const mcData = brain.mcData

  await brain.obtainItem(rawItemName, count)

  const coalNeeded = Math.ceil(count / 8)
  if (!bot.inventory.items().some((i) => i.name === 'coal' && i.count >= coalNeeded)) {
    await brain.obtainItem('coal', coalNeeded)
  }

  if (!bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 32 })) {
    await brain.obtainItem('furnace', 1)
    await brain.primitives.placeBlock('furnace')
  }

  const furnaceBlock = bot.findBlock({ matching: mcData.blocksByName.furnace.id, maxDistance: 32 })
  if (!furnaceBlock) throw new Error('No encuentro el horno que acabo de poner')

  const furnace = await bot.openFurnace(furnaceBlock)
  const rawItem = mcData.itemsByName[rawItemName]
  const fuelItem = mcData.itemsByName.coal

  if (!furnace.fuelItem()) await furnace.putFuel(fuelItem.id, null, coalNeeded)
  if (!furnace.inputItem()) await furnace.putInput(rawItem.id, null, count)

  while (bot.inventory.count(mcData.itemsByName[resultItemName].id) < count) {
    await bot.waitForTicks(20)
    if (furnace.outputItem() && furnace.outputItem().count > 0) {
      await furnace.takeOutput()
    }
  }
  furnace.close()
}