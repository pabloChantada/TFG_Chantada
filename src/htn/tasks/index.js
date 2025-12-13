const makeCraftTask = require('./craftItem')
const makeSmeltTask = require('./smelt')

module.exports = (brain) => ({
  craftItem: makeCraftTask(brain),
  smeltItem: makeSmeltTask(brain),
})