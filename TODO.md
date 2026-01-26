- [x] Probar langchain 
- [x] Llegar hasta hierro con HTN (usando replanificacion)
    - [x] Añadir precondiciones y consecuencias a la realizacion de las acciones
    - Capax de hacer acciones y replanificar -> Paper (primero, luego mirar los otros 2)
- [x] Probar la api de mindcraft

--- 

- [ ] Llegar hasta hierro de forma correcta (end-to-end estable)
    - [x] Reordenar progreso: minar carbón/hierro y DESPUÉS colocar horno y fundir
    - [x] Descartar straight mining (dejarlo comentado) y usar búsqueda “omnisciente” con findBlock
    - [ ] Estabilizar recolección de madera/piedra: exploración consistente (evitar bucles “dando vueltas”)
    - [ ] Robustecer crafteo con mesa: evitar windowOpen timeout (reintentos + re-detectar mesa + rango interacción)
    - [ ] Ajustar criterios de “has()”: contar items resultantes (coal/raw_iron/cobblestone) vs blocks (coal_ore/iron_ore/stone)
    - [ ] Añadir telemetría mínima (logs) para diagnosticar por qué falla cada fase
- [ ] Comunicacion multiagente para hacer replanificacion
    - NOTA: esto es mas simple con los LLM's y podemos empezar por ahi
- [x] Coger el viewer de vision multiagente de los LLM's 
    - Hay que ir testeando

---
- [ ] Hacer las recipes no hardcoded (planificador recursivo / crafting graph)
- [x] No X-Ray, hacer "vision" (enfoque actual: findBlock/omnisciencia; revisar limitaciones)
- [x] Problema con las update del inventario e items
- [ ] Hacer que la obtencion de objetos sea automatica (incluye exploración)
- [x] Implementar replanificacion
    - [x] Comprobar -> la replanificacion es rara en el caso de minecraft, ya que seria repetir las acciones hasta que consigas lo que quieras. 

---

Notas:
- En la obtención de materias primas aún se “raya” (loops). Prioridad: mejor exploración y condiciones de salida claras.