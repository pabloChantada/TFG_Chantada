const makeCraftPrimitive = require('../tasks/craftItem')
const makePlacePrimitive = require('./place')
const makeStrategies = require('./strategies')
const makeMinePrimitive = require('./mine')

module.exports = (brain) => {
  const primitives = {}
  primitives.craft = makeCraftPrimitive(brain)
  primitives.placeBlock = makePlacePrimitive(brain)
  primitives.strategies = makeStrategies(brain)
  primitives.mineBlock = makeMinePrimitive(brain, primitives)
  return primitives
}