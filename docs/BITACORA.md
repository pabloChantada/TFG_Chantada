# BITÁCORA

## Sesión 2026-01-26

### Cambios Implementados
- **Recolección robusta**: `src/htn/primitive_task.js` - búsqueda por rango creciente (32→64→128), exploración aleatoria, tolerancia a fallos
- **Movimiento/minado estable**: `moveToBlock` usa `GoalNear`, `mineBlock` se acerca primero, fallback a `bot.dig()`
- **Crafteo con mesa**: `smartCraft` se mueve a rango ~3, referencia correcta a mesa → mitigado timeout
- **Minería**: desactivado straight mining, enfoque omnisciente con `findBlock` + exploración
- **Progresión**: horno colocado solo cuando hay carbón/hierro para fundir
- **Mantenimiento**: evento `physicTick` (deprecated) → `physicsTick`
- **Documentación**: README actualizado con guía instalación/ejecución

## APIs & Tecnologías
- **Mineflayer**: Considerar modelos más recientes (Claude 4 vs 3.5, GPT-4o)
- **Mindcraft**: Fork actualizado con agentes personalizados
- **Mineways**: Exportación de mundos para análisis visual
- **Kaggle**: Dataset de texturas de bloques para visión por computadora
- **McIcons**: imagenes de minecraft (https://mcicons.ccleaf.com/)
## Algoritmos
- [ ] RL
- [ ] LLM
- [ ] Genético
- [ ] Simbólico (HTN)

## Decisiones de Diseño

### Estrategia de Minería
Dado el coste computacional de LLM's locales, se usa búsqueda omnisciente (`findBlock`) en lugar de exploración pixel-a-pixel.
- Filtro de ores vs bloques visibles (`blockName.includes('ore')`)
- Para objetivos como minar hierro, el standart es usar LLM's. Pero para un uso local es imposible por el costo computacional. 
Por ello se debe usar una estrategia menos eficiente (straight minning), antes que buscar una cueva por ejemplo que requiere una procesamiento
"pixel a pixel"

En la seleccion de estrategias podemos:
 - Seleccionar solo los bloques que queremos minar
 - O considerar que si tiene ore, debemos hacer straight mining

Yo me quedaria con una sola. Pero no se si es mejor hacer un filtrado (nos permite simpleficar
las tareas ?) o usar una aproximacion mas grnade con el __.includes__

```javascript
if (MINING_ORES[blockName] || blockName.includes('ore')) {
        targetBlock = await this.strategies.findNaturalOrStripMine(blockName)
} else {
        targetBlock = await this.strategies.findVisible(blockName)
}
```

### Visión
Análisis actual: `findBlock` omnisciente
- Alternativa futura: modelo convolucional/autoencoder en lugar de endpoint LLM
- Dataset: [Minecraft Block Textures](https://www.kaggle.com/datasets/urvishahir/minecraft-block-textures-dataset)
- Ejemplo:

```javascript
const result = await this.agent.prompter.promptVision(messages, imageBuffer);
```

### Crafteo
- [x] Mesa de crafteo: guardar posición al colocar
- [x] Viewer multiagente integrado

## Pendiente
- [x] Exploración de madera con `getBiomeName()`
- [x] Rotura accidental de mesa al explorar → mejorar placement logic
- [ ] Manejo de agua en movimiento
- [x] Acceso concurrente a memoria → revisar sistema

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

---

## 2 feb
- Adicion de iconos de minecraft a matplotlib
- Usar JSON para manejar el inventario ? o el progreso de las tareas en vez de un js ?

---

# DUDAS 

# DUDAS RESUELTAS
- [x] **Recolección de bloques (API vs Custom)**: Se opta por utilizar la API/Plugins existentes. Reescribir la lógica básica es redundante y propenso a errores, aunque interesante educativamente.
- [x] **Algoritmos no simbólicos (AG/RL/DL)**: Su aplicación en un entorno puramente simbólico (HTN) es compleja.
        - *Conclusión*: El Algoritmo Genético requeriría un entorno semi-determinista para ser viable, no aleatorio puro.
- [x] **Variantes de bloques (e.g., Deepslate)**: ¿Son relevantes?
        - *Conclusión*: Se descartan. Añaden complejidad excesiva al reconocimiento de bloques para un retorno de utilidad bajo en el alcance actual.
- [x] **Persistencia de Mesa de Crafteo**:
        - *Estrategia*: **Guardar la posición** en memoria al momento de colocarla.
        - *Rango*: Se opta por recordar la **coordenada exacta** (memoria global) en lugar de depender de búsquedas por rango/radio cada vez que se necesite.
