import pkg from 'mineflayer-pathfinder'
const { Movements } = pkg
import minecraftData from 'minecraft-data'
import inventoryViewer from 'mineflayer-web-inventory'

// Tareas custom
import { clearInventory } from './primitive_task.js'
import { runFullProgression } from './tasks/progression.js'

// ========================================================
// --- CONSTANTES Y CONFIGURACIONES ---
// ========================================================

let mcData

// =========================================================
// --- LÓGICA DEL BOT ---
// =========================================================

export async function startHTN(bot, inventoryPort = 3001) {
    mcData = minecraftData(bot.version)
    
    inventoryViewer(bot, { port: inventoryPort }) // Optional: might conflict with existing viewer
    // mineflayerViewer(bot, { port: 3001, firstPerson: true }) // Already handled by Mindcraft
    
    const defaultMove = new Movements(bot, mcData)
    defaultMove.canDig = true
    defaultMove.dontMineUnderFallingBlock = false 
    bot.pathfinder.setMovements(defaultMove)

    bot.chat('Bot listo. Escribe "craft" para ir a por el hierro.')

    bot.on('chat', async (username, message) => {
        if (username === bot.username) return
        // if (message === 'craft') await startFullProgression(bot)
        if (message === 'clear') await clearInventory(bot)
    })
    await startFullProgression(bot)
}

// =========================================================
// --- TAREA PRINCIPAL: PROGRESIÓN COMPLETA HASTA HIERRO ---
// =========================================================

async function startFullProgression(bot) {
    try {
        await runFullProgression(bot, mcData)
        await bot.quit()
    } catch (err) {
        bot.chat(`Proceso detenido: ${err.message}`)
        console.error(err)
    }
}