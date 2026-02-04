# APIs & Tecnologías
- **Mineflayer**: Considerar modelos más recientes (Claude 4 vs 3.5, GPT-4o)
- **Mindcraft**: Fork actualizado con agentes personalizados
- **Mineways**: Exportación de mundos para análisis visual
- **Kaggle**: Dataset de texturas de bloques para visión por computadora
- **McIcons**: imagenes de minecraft (https://mcicons.ccleaf.com/)
- **Puppeteer**: Puppeteer captura esa página como imagen PNG de forma automatizada.
## Decisiones de Diseño
### Datasets
https://github.com/cosmoharrigan/minecraft-dataset-generation

este se podria usar, pero es mas rentable usar el custom

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

### MAS
- Para el multiagente, podemos hacer supertarea de hierro:
- uno solo hace picos de piedra
- otro va a por hierro
- etc.

### World Model
png -> facil
minar:
    - Si el inventario cambia -> succes
    - si el bloque que miramos cambia, se podia obtener el state del bloque
    - Con bot.dig() se hace una promesa
movement: 
    - con posiciones igual que antes

---

- En la obtención de materias primas aún se “raya” (loops). Prioridad: mejor exploración y condiciones de salida claras.