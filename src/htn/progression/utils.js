/**
 * Get the label of a bot.
 * @param {*} bot - The bot instance
 * @returns {string} The label of the bot
 */
function getBotLabel(bot) {
    return bot?.name || bot?.username || 'bot'
}

export { getBotLabel }