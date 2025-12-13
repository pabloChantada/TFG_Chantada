// Crafteo genérico usando recetas nativas de Mineflayer (sin recetas manuales).
module.exports = (brain) =>
  async function craftItem(itemName, count) {
    const { bot, mcData } = brain
    const itemDef = mcData.itemsByName[itemName]
    if (!itemDef) throw new Error(`Item desconocido: ${itemName}`)
    const itemId = itemDef.id
    const isCraftingTable = itemName === 'crafting_table'

    // 1) Buscar receta sin mesa
    let craftingTableBlock = null
    let recipe = findRecipe(bot, itemId, null)

    // 2) Si no hay receta y NO es crafting_table, preparar mesa y reintentar
    if (!recipe && !isCraftingTable) {
      craftingTableBlock = await ensureCraftingTable(bot, mcData, brain)
      recipe = findRecipe(bot, itemId, craftingTableBlock)
    }

    // 3) Si seguimos sin receta, abortar
    if (!recipe) throw new Error(`No hay receta nativa para ${itemName}`)

    const yieldPerCraft = recipe.result && recipe.result.count ? recipe.result.count : 1
    const crafts = Math.ceil(count / yieldPerCraft)

    // 4) Asegurar ingredientes recursivamente
    const ingredients = collectIngredients(recipe, mcData)
    for (const [ingName, perCraft] of Object.entries(ingredients)) {
      const needed = perCraft * crafts
      const has = bot.inventory.count(mcData.itemsByName[ingName].id)
      if (has < needed) {
        console.log(`[Craft] Falta ${ingName} (${has}/${needed}). Obteniendo...`)
        await brain.obtainItem(ingName, needed - has)
      }
    }

    // 5) Si la receta requiere mesa y aún no la tenemos detectada, buscar/colocar
    if (recipe.requiresTable && !craftingTableBlock) {
      craftingTableBlock = await ensureCraftingTable(bot, mcData, brain)
    }

    console.log(`[Craft] Ejecutando ${crafts} crafteos de ${itemName}...`)
    await brain.primitives.craft(recipe, crafts, craftingTableBlock)
  }

// Helpers

function findRecipe(bot, itemId, table) {
  const recipes = bot.recipesFor(itemId, null, 1, table)
  return recipes && recipes.length > 0 ? recipes[0] : null
}

async function ensureCraftingTable(bot, mcData, brain) {
  let table = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 4 })
  if (table) return table
  await brain.obtainBlock('oak_log', 4)
  await brain.obtainItem('planks', 4)
  await brain.obtainItem('crafting_table', 1)
  await brain.primitives.placeBlock('crafting_table')
  table = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 4 })
  if (!table) throw new Error('Puse la mesa pero no la veo.')
  return table
}

function collectIngredients(recipe, mcData) {
  const counts = {}

  const add = (id) => {
    if (!id) return
    const name = mcData.items[id].name
    counts[name] = (counts[name] || 0) + 1
  }

  if (recipe.inShape) {
    for (const row of recipe.inShape) {
      if (!row) continue
      for (const cell of row) add(cell && cell.type ? cell.type : cell && cell.id ? cell.id : cell)
    }
  } else if (recipe.ingredients) {
    for (const ing of recipe.ingredients) add(ing && ing.type ? ing.type : ing && ing.id ? ing.id : ing)
  } else {
    throw new Error('Receta sin datos de ingredientes.')
  }

  return counts
}