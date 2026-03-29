# Pipeline de Imitation Learning

Flujo completo desde la grabación de episodios hasta el modelo entrenado.

```
scripts/mass_record.js  →  scripts/prepare_dataset.py  →  src/il/main.py
      (grabar)                    (limpiar)                  (entrenar)
```

---

## Paso 1 — Grabar episodios

Requiere un servidor Minecraft con LAN abierta.

```bash
node scripts/mass_record.js --episodes 20 --minecraft-port 25565
```

**Opciones:**
| Flag | Default | Descripción |
|------|---------|-------------|
| `--episodes N` | 10 | Número de episodios a grabar |
| `--minecraft-port` | 25565 | Puerto del servidor Minecraft |
| `--clean` | false | Limpiar directorio de métricas antes de empezar |
| `--output` | data/train.jsonl | Ruta de salida del JSONL |

Los episodios se graban en: `src/metrics/agent_metrics/`
El dataset bruto se genera en: `data/train.jsonl`

---

## Paso 2 — Limpiar dataset (recomendado)

Prepara el dataset para IL eliminando ruido y frames no informativos.

```bash
python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl
```

Con parámetros explícitos:

```bash
python scripts/prepare_dataset.py \
    --input data/train.jsonl \
    --output data/train_clean.jsonl \
    --min_samples 50 \
    --max_repeat 3 \
    --frame_skip 1
```

**Parámetros:**
| Flag | Default | Efecto |
|------|---------|--------|
| `--min_samples` | 50 | Elimina clases con menos de N ejemplos |
| `--max_repeat` | 3 | Máx. frames consecutivos con la misma acción |
| `--frame_skip` | 1 | Subsampling temporal (1=off, 3=1 de cada 3) |

---

## Paso 3 — Entrenar

```bash
python src/il/main.py --mode train --dataset data/train_clean.jsonl --backbone resnet18
```

Con más opciones:

```bash
python src/il/main.py \
    --mode train \
    --dataset data/train_clean.jsonl \
    --backbone resnet18 \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4
```

**Backbones disponibles:** `resnet18` (default), `resnet34`, `resnet50`, `resnet101`

El modelo se guarda en: `src/il/runs/{timestamp}/model_r18_{timestamp}.pth`
Los plots se guardan en: `src/il/runs/{timestamp}/plots/`

---

## Paso 4 — Evaluar

```bash
python src/il/main.py \
    --mode eval \
    --dataset data/train_balanced.jsonl \
    --model src/il/runs/{timestamp}/model_r18_{timestamp}.pth
```

---

## Estructura del JSONL

Cada línea del dataset tiene el formato:

```json
{
  "image": "src/metrics/agent_metrics/AgentName_screenshots/control_2026-03-01T12-00-00-000Z.png",
  "state": {"x": 10.5, "y": 64.0, "z": -5.3},
  "action": "move_forward_walk"
}
```

---

## Decisiones de diseño (para memoria TFG)

### Discrete vs MultiDiscrete
Se pasó de un espacio de acciones MultiDiscrete (11 dimensiones: movimiento,
cámara, ataque, crafteo…) a un espacio Discrete con 18 clases string. Motivos:
- CrossEntropyLoss es más estable que múltiples BCEs independientes.
- Reducción de la complejidad del modelo: una cabeza clasificadora vs. 11.
- Mejor interpretabilidad: cada clase tiene un nombre semántico claro.

### Optimizadores probados
- **SGD** — descartado (convergencia lenta, inestable con pocos datos).
- **Adam** — convergencia rápida pero tendencia a plateau prematuro.
- **AdamW** (elegido) — mejor generalización gracias al weight decay desacoplado.

### Class weights
Se añadieron pesos de clase inversos a la frecuencia para compensar el
desequilibrio `idle >> otras acciones`. Sin este ajuste el modelo colapsaba
prediciendo `idle` para la mayoría de frames.
Implementado con `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`.
Los pesos se recortan a máximo 10 (`np.clip`) para evitar pesos extremos
en clases con muy pocos ejemplos.

### Dropout 0.5 en la cabeza clasificadora
La cabeza final (`nn.Sequential(Dropout(0.5), Linear(...))`) se añadió
para regularizar y reducir overfitting dado el tamaño reducido del dataset.

### Split cronológico (no aleatorio)
El split train/val respeta el orden temporal de los episodios (80% primeros,
20% últimos). Un split aleatorio contaminaría el conjunto de validación con
frames del mismo episodio que el entrenamiento, inflando artificialmente la
accuracy de validación.

### Transfer learning con ResNet preentrenado
Usar pesos de ImageNet permite extraer features visuales de alta calidad
desde el primer epoch, sin necesidad de un corpus de preentrenamiento propio.
El backbone completo se fine-tunea (sin congelar capas) dada la diferencia
entre imágenes naturales e imágenes de Minecraft.
