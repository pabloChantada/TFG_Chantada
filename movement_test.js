const mineflayer = require('mineflayer')
const pathfinder = require('mineflayer-pathfinder').pathfinder
const Movements = require('mineflayer-pathfinder').Movements
const { GoalNear } = require('mineflayer-pathfinder').goals

const bot = mineflayer.createBot({
    host: 'localhost',
    port: 62792,
    username: 'Bot',
    version: '1.20.1',
    auth: 'offline',
});

bot.loadPlugin(pathfinder)

bot.once('spawn', () => {
    const defaultMove = new Movements(bot)
	// defaultMove.scafoldingBlocks.push(bot.registry.itemsByName['netherrack'].id) // Add nether rack to allowed scaffolding items
	bot.setControlState('sprint', true) // Enable sprinting
    let shouldFollow = false
    let followTarget = null
    
    bot.on('chat', function(username, message) {
        if (username === bot.username) return
        
        if (message === 'come') {
            const target = bot.players[username] ? bot.players[username].entity : null
            if (!target) {
                bot.chat('I don\'t see you !')
                return
            }
            
            shouldFollow = true
            followTarget = username
            bot.chat('Following you!')
        }
        
        if (message === 'stop') {
            shouldFollow = false
            followTarget = null
            bot.pathfinder.setGoal(null)
            bot.chat('Stopped following!')
        }
    })
    
    const followInterval = setInterval(() => {
        if (shouldFollow && followTarget) {
            const target = bot.players[followTarget] ? bot.players[followTarget].entity : null
            if (target) {
                const p = target.position
                bot.pathfinder.setMovements(defaultMove)
                bot.pathfinder.setGoal(new GoalNear(p.x, p.y, p.z, 1))
            } else {
                shouldFollow = false
                bot.chat('I lost sight of you!')
            }
        }
    }, 100)
})