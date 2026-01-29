# TODO

## Completado
- [x] Probar langchain
- [x] Llegar hasta hierro con HTN (usando replanificación)
  - [x] Añadir precondiciones y consecuencias a la realización de acciones
  - [x] Probar API de Mindcraft
- [x] Coger viewer de visión multiagente de LLM's

## En Progreso
- [ ] Llegar hasta hierro end-to-end estable
  - [x] Reordenar progreso: minar carbón/hierro → colocar horno → fundir
  - [x] Descartar straight mining, usar búsqueda omnisciente con findBlock
  - [ ] Estabilizar recolección madera/piedra (exploración consistente)
  - [ ] Robustecer crafteo con mesa (evitar windowOpen timeout)
  - [x] Ajustar criterios "has()" para items vs bloques
  - [x] Añadir telemetría mínima para diagnosticar fallos

## Por Hacer
- [ ] Comunicación multiagente para replanificación
- [ ] Recipes no hardcoded (planificador recursivo/crafting graph)
- [ ] Automatizar obtención de objetos (incluye exploración)
- [ ] Estabilizar exploración (evitar loops infinitos)
- [ ] Manejo robusto de agua
- [ ] Mejora de sistema de memorias

## Investigación
- [ ] Algoritmo RL
- [ ] Algoritmo LLM
- [ ] Algoritmo Genético
- [ ] Algoritmo Simbólico (HTN)

## Notas Técnicas
- Prioridad: mejor exploración y condiciones de salida claras en recolección de materias primas
- Mundo de cuevas multiagente: maximizar minería de ores con exploración coordinada. Mundo de cuevas -> varios agentes -> maximizar la mineria de ores

---

- getBiomeName() puede ayudar en búsqueda de madera según bioma
- En la obtención de materias primas aún se “raya” (loops). Prioridad: mejor exploración y condiciones de salida claras.

```javascript
export function getBiomeName(bot) {
    /**
     * Get the name of the biome the bot is in.
     * @param {Bot} bot - The bot to get the biome for.
     * @returns {string} - The name of the biome.
     * @example
     * let biome = world.getBiomeName(bot);
     **/
    const biomeId = bot.world.getBiome(bot.entity.position);
    return mc.getAllBiomes()[biomeId].name;
}
```

Esta funcion puede ayudar al conseguir madera