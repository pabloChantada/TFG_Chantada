# BITÁCORA

## DUDAS

## DUDAS RESUELTAS
- [x] **Recolección de bloques (API vs Custom)**: Se opta por utilizar la API/Plugins existentes. Reescribir la lógica básica es redundante y propenso a errores, aunque interesante educativamente.
- [x] **Algoritmos no simbólicos (AG/RL/DL)**: Su aplicación en un entorno puramente simbólico (HTN) es compleja.
        - *Conclusión*: El Algoritmo Genético requeriría un entorno semi-determinista para ser viable, no aleatorio puro.
- [x] **Variantes de bloques (e.g., Deepslate)**: ¿Son relevantes?
        - *Conclusión*: Se descartan. Añaden complejidad excesiva al reconocimiento de bloques para un retorno de utilidad bajo en el alcance actual.
- [x] **Persistencia de Mesa de Crafteo**:
        - *Estrategia*: **Guardar la posición** en memoria al momento de colocarla.
        - *Rango*: Se opta por recordar la **coordenada exacta** (memoria global) en lugar de depender de búsquedas por rango/radio cada vez que se necesite.
