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
- [x] Llegar hasta hierro end-to-end estable
  - [x] Reordenar progreso: minar carbón/hierro → colocar horno → fundir
  - [x] Descartar straight mining, usar búsqueda omnisciente con findBlock
  - [x] Robustecer crafteo con mesa (evitar windowO fpen timeout)
  - [x] Ajustar criterios "has()" para items vs bloques
  - [x] Añadir telemetría mínima para diagnosticar fallos
- [x] Implementar replanificacino completa
- [x] Estabilizar exploración (evitar loops infinitos)
- [x] Aumentar el numero de accinoes:
  - [x] Movimiento de camara
  - [x] Todos los botones que podria usar el jugador, mas low-level que las accinoes basicas (aunque estas no se quitan)
- [x] Hacer que el agente de refuerzo herede de base_agent
- [x] Añadir limpieza de metricas y memorias al run.js

## En Progreso
- [ ] Analisis de entrenamiento

## Por Hacer
- [ ] Comunicación multiagente para replanificación
- [ ] Recipes no hardcoded (planificador recursivo/crafting graph)
- [ ] Manejo robusto de agua
- [ ] Automatizar creacion de mundos
- [ ] Probar autoencoder sin decoder VS solo autoencoder -> Para las acciones

---
- [ ] ROBAR -> https://github.com/medipixel/rl_algorithms
- eliminar sprint y solo hacer andar
- mirar dataset
- imitation learning con red convolucional
- arreglar omnisciencia y agua

- [ ] Se puede hacer que el prismarine viewer haga un draw del movimiento del agente
- [ ] openWindow solo se ejecuta una vez cuandod deberia ser como minumo 3(?) (inventario, crafting table, furnace)
- [ ] Usar rl para talar madera (maybe crear un pico de madera, aunque seria redundante ya que son tareas "automaticas")
  - [ ] Quitar la omnisciencia, puede saber las coordenadas de los bloques que ve como muchisimo. Pero no debe saber donde esta
  el hierro por ejemplo. 

## Notas Técnicas
- Prioridad: mejor exploración y condiciones de salida claras en recolección de materias primas
- Mundo de cuevas multiagente: maximizar minería de ores con exploración coordinada. Mundo de cuevas -> varios agentes -> maximizar la mineria de ores
- Manejo de agua en movimiento