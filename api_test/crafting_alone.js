const mineflayer = require('mineflayer')
const { goals: { GoalBlock } } = require('mineflayer-pathfinder')
const Vec3 = require('vec3')

// Configuración de minería
const ORE_CONFIGS = {
    'coal_ore': { min: 0, max: 320, opt: 95 },
    'iron_ore': { min: -64, max: 256, opt: 16 },
    'gold_ore': { min: -64, max: 32, opt: -16 },
    'diamond_ore': { min: -64, max: 16, opt: -59 },
}

const HARVEST_MAP = {
    'oak_log': 'oak_log',
    'dirt': 'dirt',
    'cobblestone': 'stone',
    'raw_iron': 'iron_ore',
    'raw_gold': 'gold_ore',
    'diamond': 'diamond_ore',
    'coal': 'coal_ore',
    'iron_ore': 'iron_ore'
}

const SMELT_MAP = {
    'iron_ingot': 'raw_iron',
    'gold_ingot': 'raw_gold'
}

// Mapa de herramientas requeridas
const REQUIRED_TOOL = {
    'stone': 'wooden_pickaxe', // Necesita pico de madera para dar cobblestone
    'iron_ore': 'stone_pickaxe',
    'gold_ore': 'iron_pickaxe',
    'diamond_ore': 'iron_pickaxe'
}

class Brain {
    constructor(bot, mcData) {
        this.bot = bot
        this.mcData = mcData
        this.tasks.parent = this
        this.primitives.parent = this
        this.primitives.strategies.parent = this.primitives
    }

    async obtainItem(itemName, count) {
        console.log(`[Meta] Obtener ${count} de ${itemName}`)
        const item = this.mcData.itemsByName[itemName]
        if (!item) throw new Error(`Item desconocido: ${itemName}`)
        
        const itemId = item.id
        
        let currentCount = this.bot.inventory.count(itemId)
        if (currentCount >= count) {
            console.log(`[Check] Ya tengo suficientes ${itemName}`)
            return true 
        }

        const needed = count - currentCount
        
        if (HARVEST_MAP[itemName]) {
            const blockToMine = HARVEST_MAP[itemName]
            await this.primitives.mineBlock(blockToMine, needed)
        } 
        else if (SMELT_MAP[itemName]) {
            const rawItem = SMELT_MAP[itemName]
            await this.tasks.smeltItem(rawItem, itemName, needed)
        }
        else {
            await this.tasks.craftItem(itemName, needed)
        }
        
        // Verificación final recursiva
        return this.bot.inventory.count(itemId) >= count
    }
    
    async obtainBlock(blockName, count) {
        console.log(`[Meta] Obtener ${count} bloques de ${blockName}`)
        const block = this.mcData.blocksByName[blockName]
        if (!block) throw new Error(`Bloque desconocido: ${blockName}`)

        let currentCount = this.bot.inventory.count(block.id)
        if (currentCount >= count) {
            console.log(`[Check] Ya tengo suficientes ${blockName}`)
            return true
        }

        const needed = count - currentCount
        await this.primitives.mineBlock(blockName, needed)

        // Verificación final recursiva
        return this.bot.inventory.count(block.id) >= count
    }

    tasks = {
        async smeltItem(rawItemName, resultItemName, count) {
            // (Tu lógica de smelt se mantiene igual, omitida por brevedad si no se usa ahora)
            // ... [Mantener tu código de smelt aquí si lo deseas] ...
        },

        async craftItem(itemName, count) {
            const bot = this.parent.bot
            const mcData = this.parent.mcData
            const itemId = mcData.itemsByName[itemName].id

            // --- IMPORTANTE: Clases internas para arreglar el error 'reading 0' ---
            const Recipe = require('prismarine-recipe')(bot.version).Recipe
            const Item = require('prismarine-item')(bot.version)

            const findRecipe = (table) => {
                // 1. Intentar receta oficial
                const recipes = bot.recipesFor(itemId, null, 1, table)
                if (recipes.length > 0) return recipes[0]

                // 2. Recetas Manuales (Fallback)
                console.log(`[Recipe] Creando receta manual para ${itemName}...`)

                // Helper: Crea una instancia REAL de Item (esto arregla el crash)
                const makeIng = (name) => new Item(mcData.itemsByName[name].id, 1)
                
                // Helper: Define costes/beneficios
                const makeDelta = (ingName, ingCount, resCount) => [
                    { id: mcData.itemsByName[ingName].id, count: -ingCount },
                    { id: itemId, count: resCount }
                ]

                if (itemName === 'crafting_table') {
                    const plank = makeIng('oak_planks')
                    const recipe = new Recipe(
                        new Item(itemId, 1),              
                        [[plank, plank], [plank, plank]], // Shape
                        [[plank, plank], [plank, plank]]  // InShape
                    )
                    recipe.delta = makeDelta('oak_planks', 4, 1)
                    return recipe
                }

                if (itemName === 'oak_planks') {
                    const log = makeIng('oak_log')
                    const recipe = new Recipe(
                        new Item(itemId, 4), 
                        [[log]], // Shape
                        [[log]]  // InShape
                    )
                    recipe.delta = makeDelta('oak_log', 1, 4)
                    return recipe
                }

                if (itemName === 'stick') {
                    const plank = makeIng('oak_planks')
                    const recipe = new Recipe(
                        new Item(itemId, 4),
                        [[plank], [plank]],
                        [[plank], [plank]]
                    )
                    recipe.delta = makeDelta('oak_planks', 2, 4)
                    return recipe
                }
                
                // Receta manual para PICO DE MADERA (necesaria si la mesa no desbloquea la receta al instante)
                if (itemName === 'wooden_pickaxe') {
                    const stick = makeIng('stick')
                    const plank = makeIng('oak_planks')
                    // Forma del pico:
                    // [P, P, P]
                    // [ , S,  ]
                    // [ , S,  ]
                    const recipe = new Recipe(
                        new Item(itemId, 1),
                        [[plank, plank, plank], [null, stick, null], [null, stick, null]],
                        [[plank, plank, plank], [null, stick, null], [null, stick, null]]
                    )
                    recipe.delta = [
                        { id: mcData.itemsByName.oak_planks.id, count: -3 },
                        { id: mcData.itemsByName.stick.id, count: -2 },
                        { id: itemId, count: 1 }
                    ]
                    recipe.requiresTable = true // Importante
                    return recipe
                }

                return null
            }

            // --- Lógica de Obtención de Mesa ---
            let recipe = findRecipe(null)
            let craftingTableBlock = null

            // Si no hay receta sin mesa, o la receta requiere mesa explícitamente
            if (!recipe || recipe.requiresTable) {
                // Buscar mesa existente
                craftingTableBlock = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 4 })
                
                // Si encontramos mesa, re-buscamos receta (Mineflayer a veces necesita la mesa para dar la receta oficial)
                if (craftingTableBlock) recipe = findRecipe(craftingTableBlock)

                // Si seguimos sin receta O sin mesa y la necesitamos
                if ((!recipe && itemName !== 'crafting_table') || (recipe && recipe.requiresTable && !craftingTableBlock)) {
                    console.log("[Craft] Necesito una mesa de crafteo cercana.")
                    
                    if (itemName === 'crafting_table') throw new Error("Error lógico: Bucle al intentar hacer mesa.")

                    // 1. Obtener item mesa
                    await this.parent.obtainItem('crafting_table', 1)
                    
                    // 2. Colocar mesa
                    console.log("Colocando mesa...")
                    await this.parent.primitives.placeBlock('crafting_table')
                    
                    // 3. ESPERAR (Crucial para arreglar 'no veo la mesa')
                    await bot.waitForTicks(20) 
                    
                    // 4. Buscarla de nuevo
                    craftingTableBlock = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 5 })
                    if (!craftingTableBlock) throw new Error("Puse la mesa pero el servidor no me la devuelve aún (Lag o bug).")
                    
                    recipe = findRecipe(craftingTableBlock)
                }
            }

            if (!recipe) throw new Error(`No hay receta para ${itemName} (ni oficial ni manual)`)

            // --- Verificación de Ingredientes ---
            if (recipe.delta) {
                for (const d of recipe.delta) {
                    if (d.count < 0) {
                        const ingredientId = d.id
                        const ingredientName = mcData.items[ingredientId].name
                        const amountPerCraft = Math.abs(d.count)
                        const resultYield = (recipe.result && recipe.result.count) ? recipe.result.count : 1
                        const loops = Math.ceil(count / resultYield)
                        const totalRequired = amountPerCraft * loops

                        const currentHas = bot.inventory.count(ingredientId)
                        if (currentHas < totalRequired) {
                            console.log(`[Craft] Falta ingrediente: ${ingredientName}. Voy a buscarlo.`)
                            await this.parent.obtainItem(ingredientName, totalRequired - currentHas)
                        }
                    }
                }
            }

            // Re-verificar mesa si la receta la pide (por si se rompió o algo)
            if (recipe.requiresTable && !craftingTableBlock) {
                 craftingTableBlock = bot.findBlock({ matching: mcData.blocksByName.crafting_table.id, maxDistance: 4 })
            }

            console.log(`[Craft] Ejecutando crafteo de ${itemName}...`)
            const yieldPerCraft = (recipe.result && recipe.result.count) ? recipe.result.count : 1
            const loops = Math.ceil(count / yieldPerCraft)
            
            try {
                await bot.craft(recipe, loops, craftingTableBlock)
                console.log(`[Craft] ${itemName} creado exitosamente.`)
            } catch (err) {
                console.error("Error en bot.craft:", err.message)
                throw err
            }
        }
    }

    primitives = {
        async placeBlock(itemName) {
            const bot = this.parent.bot
            
            // Buscar dónde ponerla. Buscamos cualquier bloque sólido cerca.
            const referenceBlock = bot.findBlock({ 
                matching: (blk) => blk.boundingBox === 'block', 
                maxDistance: 4,
                useExtraInfo: true 
            })
            
            if (!referenceBlock) throw new Error("No hay suelo cercano para poner el bloque")

            const item = bot.inventory.items().find(i => i.name === itemName)
            if (!item) throw new Error(`No tengo ${itemName} en inventario para ponerlo.`)

            await bot.equip(item, 'hand')
            await bot.waitForTicks(5) // Esperar equipamiento

            // Intentar ponerlo en la cara de arriba (0, 1, 0)
            try {
                await bot.placeBlock(referenceBlock, new Vec3(0, 1, 0))
            } catch (err) {
                console.log(`Error al colocar (puede estar obstruido): ${err.message}`)
                // Intentar en otra cara si falla (opcional)
            }
        },

        async mineBlock(blockName, count) {
            const bot = this.parent.bot
            
            console.log(`[Mine] Iniciando búsqueda de ${count} ${blockName}`)

            // --- GESTIÓN DE HERRAMIENTAS INTELIGENTE ---
            if (REQUIRED_TOOL[blockName]) {
                const toolName = REQUIRED_TOOL[blockName]
                const hasTool = bot.inventory.items().some(i => i.name.includes(toolName))
                
                if (!hasTool) {
                    console.log(`[Tool] Necesito ${toolName} para minar ${blockName}.`)
                    // RECURSIÓN: Ir a obtener la herramienta
                    await this.parent.obtainItem(toolName, 1)
                }
                
                const toolItem = bot.inventory.items().find(i => i.name === toolName)
                if (toolItem) await bot.equip(toolItem, 'hand')
            }
            // ------------------------------------------

            let collected = 0
            while (collected < count) {
                let targetBlock = null
                
                // Si es un mineral raro, usamos lógica de búsqueda profunda
                if (ORE_CONFIGS[blockName]) {
                    targetBlock = await this.strategies.findNaturalOrStripMine(blockName)
                } else {
                    // Si es madera o tierra, búsqueda simple visible
                    targetBlock = await this.strategies.findVisible(blockName)
                }

                if (!targetBlock) {
                    console.log("No veo bloque objetivo, esperando...")
                    await bot.waitForTicks(40)
                    continue 
                }

                try {
                    // Ir al bloque
                    const goal = new GoalBlock(targetBlock.position.x, targetBlock.position.y, targetBlock.position.z)
                    await bot.pathfinder.goto(goal)
                    
                    // Minar
                    await bot.collectBlock.collect(targetBlock)
                    collected++
                    console.log(`[Mine] ${blockName}: ${collected}/${count}`)
                } catch (err) {
                    console.log(`Error minando bloque: ${err.message}`)
                    // Si falla pathfinding o mining, romper un poco de alrededor o reintentar
                    await bot.waitForTicks(20)
                }
            }
        },
        
        strategies: {
            async findVisible(blockName) {
                const bot = this.parent.parent.bot
                const mcData = this.parent.parent.mcData
                let matchingIds = [mcData.blocksByName[blockName].id]
                // Soporte para variantes deepslate
                if (mcData.blocksByName[`deepslate_${blockName}`]) matchingIds.push(mcData.blocksByName[`deepslate_${blockName}`].id)
                
                return bot.findBlock({ maxDistance: 32, matching: matchingIds })
            },

            async findNaturalOrStripMine(blockName) {
                // (Misma estrategia que tenías, funciona bien para empezar)
                const bot = this.parent.parent.bot
                const visible = await this.findVisible(blockName)
                if (visible) return visible

                // Si no es visible, moverse aleatoriamente o strip mining simple
                const targetPos = bot.entity.position.offset(1, 0, 0)
                const blockInFront = bot.blockAt(targetPos)
                if (blockInFront && blockInFront.diggable && blockInFront.name !== 'bedrock') return blockInFront
                
                return null 
            }
        }
    }
}

module.exports = Brain