# TODO

## Completado
- [x] Probar langchain
- [x] Llegar hasta hierro con HTN (usando replanificación)
  - [x] Añadir precondiciones y consecuencias a la realización de acciones
  - [x] Probar API de Mindcraft
- [x] Coger viewer de visión multiagente de LLM's
- [x] Mejora de sistema de memorias
- [x] Mejorar madera
- [x] Cambiar iconos matplotlib
- [x] Automatizar obtención de objetos (incluye exploración)
- [x] Juntar: accion (andar, picar, supersimple) + imagen + reward + completado o no
- [x] Mesa de crafteo: guardar posición al colocar
- [x] Viewer multiagente integrado
- [x] Exploración de madera con `getBiomeName()`
- [x] Rotura accidental de mesa al explorar → mejorar placement logic
- [x] Acceso concurrente a memoria → revisar sistema
- [x] World Models -> Investigar
  - [x] Crear red de refuerzo para: predecir acciones | generar imagenes
  - [x] Las imagenes seran como maxisimo 256x256


## En Progreso
- [ ] Llegar hasta hierro end-to-end estable
  - [x] Reordenar progreso: minar carbón/hierro → colocar horno → fundir
  - [x] Descartar straight mining, usar búsqueda omnisciente con findBlock
  - [ ] Robustecer crafteo con mesa (evitar windowO fpen timeout)
  - [x] Ajustar criterios "has()" para items vs bloques
  - [x] Añadir telemetría mínima para diagnosticar fallos
  - [ ] Implementar replanificacino completa

 - [ ] Limpiar y refactorizar
  - [x] Agents
  - [ ] Evaluations
  - [ ] HTN
  - [ ] LLM ?
  - [x] Metrics
  - [ ] Server

## Por Hacer
- [ ] Comunicación multiagente para replanificación
- [ ] Recipes no hardcoded (planificador recursivo/crafting graph)
- [ ] Estabilizar exploración (evitar loops infinitos)
- [ ] Manejo robusto de agua
- [ ] Automatizar creacion de mundos
- [ ] Probar autoencoder sin decoder VS solo autoencoder -> Para las acciones


## Notas Técnicas
- Prioridad: mejor exploración y condiciones de salida claras en recolección de materias primas
- Mundo de cuevas multiagente: maximizar minería de ores con exploración coordinada. Mundo de cuevas -> varios agentes -> maximizar la mineria de ores
- Manejo de agua en movimiento