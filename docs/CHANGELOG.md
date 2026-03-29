# Changelog técnico — TFG Minecraft IL

Registro de decisiones técnicas y cambios relevantes. Los cambios menores
(imports, rutas, comentarios) se agrupan al final de cada sesión.

---

## 2026-03-29 — Auditoría, reorganización y mejoras a la red

### Bug fixes críticos

- **`/n` en lugar de `\n`** en `legacy/evaluation/llm.py` (7 ocurrencias).
  El escape incorrecto imprimía literalmente "/n" en consola.
- **`balance_dataset.py` inexistente**: `mass_record.js` lo llamaba pero el archivo no existía.
  Resuelto creando el script (y posteriormente fusionado en `scripts/prepare_dataset.py`).
- **Paths Windows** `data\\train.jsonl` en `src/il/main.py` → convertidos a `data/train.jsonl`.
- **Errores silenciosos** en `load_dataset.py`: `except Exception: continue` reemplazado
  por logging del error con número de línea y tipo de excepción.

### Reorganización de archivos

**Carpeta `scripts/`** — scripts de orquestación del pipeline:

| Nuevo | Origen |
|-------|--------|
| `scripts/run.js` | `run.js` (raíz) |
| `scripts/mass_record.js` | `src/evaluation/mass_record.js` |
| `scripts/dataset.py` | `src/evaluation/dataset.py` |
| `scripts/prepare_dataset.py` | nuevo — unificación de scripts de limpieza |

**Carpeta `legacy/`** — experimentos descartados (movidos con `git mv`, historial preservado):
- `legacy/evaluation/llm.py` — pipeline alternativo ResNet-18
- `legacy/evaluation/vae_train.py` + `vae_visuals.py` — experimento VAE
- `legacy/evaluation/compartion.py` — comparador sin uso
- `legacy/rl/train.py` — BC básico sin transfer learning
- `legacy/ideas_test/` — prototipos iniciales HTN

Ver `docs/legacy/EXPERIMENTOS.md` para el razonamiento de cada descarte.

### Script único de preparación IL: `scripts/prepare_dataset.py`

Sustituye a `clean_dataset.py` y `balance_dataset.py` (ambos eliminados). Aplica:
1. Filtro de clases minoritarias (`--min_samples`, default 50)
2. Eliminación de frames consecutivos idénticos (`--max_repeat`, default 3)
3. Subsampling temporal (`--frame_skip`, default 1 = desactivado)

### Simplificación de scripts

- `mass_record.js`: eliminados `--balance`, `--max-repeat`, `--upsample-rare`, `--agent-type`.
  La limpieza del dataset pasa a ser un paso manual explícito.
- `scripts/run.js`: eliminado `--metrics-path` (hardcodeado el patrón estándar).
- `scripts/dataset.py`: eliminado `--root_dir` (siempre `"."` en la práctica).

### Numeración continua de episodios en `mass_record.js`

Al ejecutar `--episodes N` varias veces, el script detecta el número más alto
de episodio existente en `metricsDir` (patrón `Rec_N_*`) y continúa desde `N+1`.
Si se usa `--clean`, se limpia el directorio y vuelve a empezar desde 1.

### Centralización de constantes: `src/il/constants.py`

`ACTIONS`, `IDLE_ACTION`, `BATCH_SIZE`, `IMAGENET_MEAN`, `IMAGENET_STD` centralizados.
`load_dataset.py` los importa desde aquí para que un único cambio se propague.

---

## 2026-03-29 — Mejoras a la red IL (anti-overfitting)

### Diagnóstico (últimas curvas de entrenamiento)

| Métrica | Epoch 0 | Epoch 9 |
|---------|---------|---------|
| Train Loss | 1.80 | 0.13 |
| Val Loss | 1.80 | **2.80** ↑ |
| Train Acc | 35% | **95%** |
| Val Acc | 38% | **50%** (plana) |

Causa: ResNet18 tiene ~11M parámetros. Con ~4k ejemplos el modelo memoriza
el training set desde el epoch 2.

### 1. Congelación parcial del backbone (`model.py`) — cambio principal

Se congelan `layer1`, `layer2`, `layer3` del ResNet. Solo se entrenan
`layer4` (~2.1M params) y la cabeza FC (~0.1M params).

**Razonamiento:** Las capas iniciales (entrenadas en ImageNet) detectan bordes,
texturas y formas básicas, útiles para cualquier dominio visual. `layer4` captura
features semánticas de alto nivel que deben adaptarse al dominio Minecraft.
Reducir los parámetros entrenables de 11M a ~2.2M es la principal barrera
contra el overfitting con pocos datos.

### 2. Label smoothing 0.1 (`main.py`)

`CrossEntropyLoss(label_smoothing=0.1)` distribuye el 10% de la probabilidad
target entre todas las clases. Penaliza predicciones demasiado confiadas y
mejora la generalización.

### 3. Scheduler `ReduceLROnPlateau` (`main.py`)

Reduce el LR por 0.5 si `val_loss` no mejora en 2 epochs consecutivos (mínimo 1e-6).
Permite que el modelo siga aprendiendo después de plateaus en lugar de oscilar
con LR fijo.

El optimizador pasa a recibir solo los parámetros con `requires_grad=True`,
reduciendo también el overhead de memoria.

### 4. Augmentación ampliada (`load_dataset.py`)

Añadidos al pipeline de train:
- `RandomHorizontalFlip()` — Minecraft es simétrico izquierda/derecha para la
  mayoría de acciones visuales. Dobla la diversidad sin coste real.
- `RandomGrayscale(p=0.1)` — evita que la red clasifique por el tono de verde del
  bioma en lugar de las formas y el movimiento.

### Limpieza de código legacy en `src/il/`

**`load_dataset.py`**: eliminada `vector_to_action_label()` (80 líneas, formato
MultiDiscrete que ya no se usa) y simplificada `normalize_action_label()` de
30 líneas a 3 (solo string, fallback a `"idle"`).

**`plots.py`**: eliminado `ACTIONS` dict multidiscrete (`{0: "move_forward", ...}`)
y la rama de formato tuple legacy en `plot_gradcam`.

---

## Estado del pipeline activo

```
node scripts/mass_record.js --episodes N --minecraft-port PORT
    ↓  src/metrics/agent_metrics/*_metrics.json
python scripts/dataset.py
    ↓  data/train.jsonl
python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl
    ↓  data/train_clean.jsonl
python src/il/main.py --mode train --dataset data/train_clean.jsonl --backbone resnet18
    ↓  src/il/runs/{timestamp}/model_r18_{timestamp}.pth
```
