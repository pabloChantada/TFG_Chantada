module.exports = (brain) => async function craft(recipe, count, table) {
  const bot = brain.bot
  try {
    const yieldPerCraft = recipe.result && recipe.result.count ? recipe.result.count : 1
    const loops = Math.ceil(count / yieldPerCraft)

    console.log(`[Action] Crafteando ${loops} veces (Mesa: ${table ? 'SI' : 'NO'})...`)
    await bot.craft(recipe, loops, table)
    console.log('Crafteo completado exitosamente.')
  } catch (err) {
    console.error('Error crítico crafteando:', err.message)
    throw err
  }
}