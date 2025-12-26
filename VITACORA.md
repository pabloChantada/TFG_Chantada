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

- Las variantes (deepslate) son relevantes ? Para una tarea muy compleja o un endgame puede. Pero me parece un execeso de complejidad para poco retorno.
- La mesa de crafteo seria mejor guardar la posición cuando la colocamos,
- O buscarla en un rango amplio.

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