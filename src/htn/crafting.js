const mineflayer = require('mineflayer')
const { pathfinder } = require('mineflayer-pathfinder')
const collectBlock = require('mineflayer-collectblock').plugin
const inventoryViewer = require('mineflayer-web-inventory')
const Brain = require('./brain')

const bot = mineflayer.createBot({
  host: 'localhost',
  port: parseInt(process.argv[2]) || 25565,
  username: 'IronBot',
  version: '1.20.1',
  auth: 'offline',
})

bot.once('spawn', () => {
  bot.loadPlugin(pathfinder)
  bot.loadPlugin(collectBlock)

  inventoryViewer(bot)

  const mcData = require('minecraft-data')(bot.version)
  const brain = new Brain(bot, mcData)

  console.log("Bot listo. Escribe 'stone_pickaxe' en el chat (objetivo actual).")

  bot.on('chat', async (username, message) => {
    if (message === 'stone_pickaxe') {
      bot.chat('Entendido, voy por un Pico de Piedra.')
      try {
        await brain.obtainItem('stone_pickaxe', 1)
        bot.chat('¡Misión cumplida! Tengo el pico de piedra.')
      } catch (err) {
        bot.chat(`Fallé: ${err.message}`)
        console.error(err)
      }
    }
  })
})

bot.on('error', console.log)
bot.on('kicked', console.log)