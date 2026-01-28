# BITÁCORA

## Sesión 2026-01-26

- Recolección más robusta: en `src/htn/primitive_task.js` se mejoró `collectResource` con búsqueda por rango creciente (32→64→128), exploración aleatoria cuando no hay bloques visibles, y tolerancia a fallos.
- Movimiento/minado más estable: `moveToBlock` usa `GoalNear` y `mineBlock` primero se acerca al bloque; si `collectBlock` falla, hace fallback a `bot.dig()`.
- Fix del crafteo con mesa: `smartCraft` ahora se mueve a rango de interacción (~3) y usa la referencia correcta a la mesa al llamar a `craftItem`, mitigando `windowOpen timeout`.
- Minería: se desactivó la heurística de “straight mining” (se dejó comentada) y se pasó a un enfoque “omnisciente” basado en `findBlock` + exploración (`src/htn/tasks/mining.js`).
- Progresión: se reordenó el flujo para colocar el horno sólo cuando ya hay carbón/hierro y se va a fundir (`src/htn/tasks/progression.js`).
- Mantenimiento: evento deprecated `physicTick`→`physicsTick` en `api_test/hello_world.js`.
- Documentación: README raíz actualizado con guía completa de instalación/ejecución y referencias.

# API's  
- Mindflayer
    - The models used in the paper are kinda old (claude 3.5 and chatgpt 4o); using newer models in comparation may be interesting
- Mindcraft

# ALOGIRTHMS
- [ ] RL
- [ ] LLM
- [ ] GENETIC 
- [ ] SIMBOLIC (HTN)

# DUDAS
- ~~Con la recoleccion de bloques por ejemplo. Es mas interesante usar lo que da la api, o crear yo por mi cuenta la funcion.~~
~~En realidad es redudante, ya que el plugin esta diseñado para evitar este codigo. Pero a lo mejor es interesante el crear las funciones ?~~

- ~~La aplicacion de LLM's y HTN la entiendo para aplicarla, pero cosas como algoritmos genetico/RL/DL no entiendo como se aplicaria a un entorno puramente simbolico~~

    - ~~En el caso del AG no se podria aplicar a un entorno aleatorio, tendria que ser semi-determinista~~

-~~Las variantes (deepslate) son relevantes ? Para una tarea muy compleja o un endgame puede. Pero me parece un execeso de complejidad para poco retorno. ~~

-~~La mesa de crafteo seria mejor guardar la posición cuando la colocamos, o  buscarla en un rango amplio.> guardar la posicion~~
        - **La posicion deberia ser por rango ? i.e: posicion en 32 bloques  o infinita ??**

# OBSERVACIONES
- Para objetivos como minar hierro, el standart es usar LLM's. Pero para un uso local es imposible por el costo computacional. 
Por ello se debe usar una estrategia menos eficiente (straight minning), antes que buscar una cueva por ejemplo que requiere una procesamiento
"pixel a pixel"

En la seleccion de estrategias podemos:
 - Seleccionar solo los bloques que queremos minar
 - O considerar que si tiene ore, debemos hacer straight mining

Yo me quedaria con una sola. Pero no se si es mejor hacer un filtrado (nos permite simpleficar
las tareas ?) o usar una aproximacion mas grnade con el __.includes__

if (MINING_ORES[blockName] || blockName.includes('ore')) {
        targetBlock = await this.strategies.findNaturalOrStripMine(blockName)
} else {
        targetBlock = await this.strategies.findVisible(blockName)
}


---

En el LLM, el analisis de la vision es usando una prompt al LLM.
```javascript
const result = await this.agent.prompter.promptVision(messages, imageBuffer);
```

En nuestro caso sería entrenar un modelo simple de vision ? Seria eliminar el endpoint de un LLM
por una red convolucional/autoencoder -> [Dataset de bloques de minecraft](https://www.kaggle.com/datasets/urvishahir/minecraft-block-textures-dataset)

---

- [x] Coger el viewer de vision multiagente de los LLM's 
    - Hay que ir testeando