const Vec3 = require('vec3')

module.exports = (brain) => async function placeBlock(itemName) {
  const bot = brain.bot
  const referenceBlock = bot.findBlock({ matching: (blk) => blk.boundingBox === 'block', maxDistance: 3 })
  if (!referenceBlock) throw new Error('No hay suelo para poner bloque')

  const item = bot.inventory.items().find((i) => i.name === itemName)
  if (item) {
    await bot.equip(item, 'hand')
    try {
      await bot.placeBlock(referenceBlock, new Vec3(0, 0, 1))
    } catch (err) {
      // Ignorar errores de colocación ocasionales
    }
  } else {
    throw new Error(`Intenté poner ${itemName} pero no lo tengo en inventario.`)
  }
}