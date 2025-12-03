const mineflayer = require('mineflayer')

const bot = mineflayer.createBot({
  host: 'localhost',  
  port: 62792,        // Minecraft LAN port
  username: 'Bot',      
  version: '1.20.1',  // Need the exact version to avoid compatibility issues
  auth: 'offline',    // Offline mode since we run in a local server
});


function lookAtNearestEntity() {
    const playerFilter = (entity) => entity.type === 'player';
    const playerEntity = bot.nearestEntity(playerFilter)

    if (!playerEntity) return; // No player found

    // Calculate the position to look at (eye level of the player)
    // Base look direction is on the player's feet
    const playerPosition = playerEntity.position.offset(0, playerEntity.height, 0);
    bot.lookAt(playerPosition, true);
}

bot.on('physicTick', lookAtNearestEntity)