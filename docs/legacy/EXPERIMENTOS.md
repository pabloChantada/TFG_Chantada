# Experimentos descartados — TFG Minecraft IL

Este documento registra los experimentos realizados durante el TFG que fueron
descartados en favor del pipeline actual (`src/il/` con ResNet18 preentrenado).
Los archivos han sido movidos a `legacy/` para conservar la historia de
investigación y facilitar la redacción de la memoria.

---

## 1. VAE para compresión de screenshots

**Archivos:** `legacy/evaluation/vae_train.py`, `legacy/evaluation/vae_visuals.py`

**Qué era:**
Autoencoder Variacional (VAE) para comprimir las capturas de pantalla de Minecraft
en un espacio latente compacto, con la idea de usarlo como backbone visual
en lugar de ResNet preentrenado.

**Arquitecturas probadas:**
- `small` — imágenes 64×64, espacio latente de 32 dimensiones
- `medium` — imágenes 128×128, espacio latente de 64 dimensiones
- `large` — imágenes 128×128, espacio latente de 128 dimensiones

**Función de pérdida:** MSE (reconstrucción) + KLD (regularización), con
beta-annealing para equilibrar los dos términos durante el entrenamiento.

**Motivo del descarte:**
La reconstrucción visual requería muchos más datos de los disponibles para que
las features latentes fueran suficientemente discriminativas para clasificar
acciones. Un ResNet preentrenado en ImageNet proporciona features de alta
calidad desde el primer epoch sin necesidad de preentrenamiento adicional.

---

## 2. Pipeline alternativo con estado global (`llm.py`)

**Archivo:** `legacy/evaluation/llm.py`

**Qué era:**
Pipeline completo alternativo que incluía carga de datos, entrenamiento,
evaluación y predicción en un único script. Usaba ResNet-18 con fusión de
estado (posición x,y,z + yaw/pitch + historial de acciones) mediante un MLP
multi-capa.

**Diferencias con `src/il/main.py` (pipeline activo):**

| Aspecto | llm.py (descartado) | il/main.py (activo) |
|---------|--------------------|--------------------|
| Backbone | ResNet-18 fijo | Configurable: 18/34/50/101 |
| Fusión de estado | Sí (MLP con estado) | No (solo imagen) |
| Gestión de acciones | Variables globales mutables | Dinámico vía dataset |
| Timestamps de run | No | Sí (directorio por run) |
| Bugs | 7 instancias de `/n` en lugar de `\n` | Corregido |

**Estado:** Descartado en favor de `src/il/main.py`. Conservado como
referencia de la arquitectura inicial con fusión multimodal.

---

## 3. Behavioral Cloning básico con CNN propia

**Archivo:** `legacy/rl/train.py`

**Qué era:**
Script de Behavioral Cloning con una CNN entrenada desde cero (sin transfer
learning). Usaba el espacio de acciones MultiDiscrete (11 dimensiones) en lugar
del espacio Discrete que se adoptó finalmente.

**Diferencias con el pipeline activo:**

| Aspecto | rl/train.py (descartado) | il/main.py (activo) |
|---------|-------------------------|-------------------|
| Backbone | CNN propia (desde cero) | ResNet preentrenado |
| Transfer learning | No | Sí (ImageNet) |
| Espacio de acciones | MultiDiscrete (11-dim) | Discrete (18 clases string) |
| Calidad de features | Baja con pocos datos | Alta desde el primer epoch |

**Motivo del descarte:**
Entrenar una CNN desde cero con los ~4k ejemplos disponibles no convergía de
forma consistente. La transición a ResNet preentrenado con espacio Discrete
mejoró significativamente la convergencia y la interpretabilidad.

---

## 4. Script de comparación de VAEs

**Archivo:** `legacy/evaluation/compartion.py` *(typo intencional preservado)*

**Qué era:**
Script de ~14 líneas para comparar checkpoints de distintas configuraciones
de VAE. Dependía de `vae_visuals.py` para las métricas.

**Estado:** 14 líneas sin funcionalidad standalone. Sin uso activo desde que
se descartó el enfoque VAE.

---

## 5. Prototipos iniciales de HTN

**Archivos:** `legacy/ideas_test/`

**Qué era:**
Experimentos de prueba de las primitivas HTN antes de integrarlas en
`src/htn/`. Incluyen:

- `crafting.js`, `crafting_alone.js` — pruebas de crafting de madera/herramientas
- `smelting.js` — prueba de fundición con horno
- `movement_test.js` — pruebas de movimiento básico
- `hello_world.js` — primer bot funcional (conexión y chat)
- `alg_test.js`, `alg_test.py` — pruebas del algoritmo de planificación HTN
- `dataset/analyze.ipynb` — análisis exploratorio del dataset original

**Estado:** Funcionalidad completamente integrada en `src/htn/primitives/` y
`src/htn/progression/`. Conservados como referencia de la evolución del proyecto.
