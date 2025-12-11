const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals: { GoalBlock } } = require('mineflayer-pathfinder')
const Vec3 = require('vec3')
const inventoryViewer = require('mineflayer-web-inventory')
const collectBlock = require('mineflayer-collectblock').plugin
const mineflayerViewer = require('prismarine-viewer').mineflayer

const bot = mineflayer.createBot({
    host: 'localhost',
    port: parseInt(process.argv[2]) || 25565, // Pass port as command line argument, or use default 25565
    username: 'collecterBot',
    version: '1.20.1',
    auth: 'offline',
})

// Base de conocimientos simplificada (Mapeo Item -> Bloque para minar)
const HARVEST_MAP = {
    'oak_log': 'oak_log',
    'cobblestone': 'stone',
    'raw_iron': 'iron_ore' // Or deepslate_iron_ore
}

bot.once('spawn', () => {
    // Inicializar plugins
    bot.loadPlugin(pathfinder)
    
    bot.on('chat', async (username, message) => {
        if (message === 'iron_pickaxe') {
            // META SUPREMA: Quiero un pico de hierro
            await brain.obtainItem('iron_pickaxe', 1)
        }
    })
})