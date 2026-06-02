# SAC + HTN como tutor (DAgger) — Trabajo futuro

> **Estado: APARCADO** (2026-06-02). Prioridad actual: redactar la memoria del
> TFG. Este documento congela el diseño acordado para retomarlo sin re-derivar
> el contexto. No hay código nuevo escrito todavía para este plan.

## Objetivo

SAC discreto visual donde, **en los steps de exploración**, el control lo toma
el **sistema HTN JS real** (`src/htn/`, pathfinder + `bot.dig` multi-tick), no
una heurística reactiva. Esto convierte el HTN (el "agente clásico" que sí
completa la tarea) en el **experto/tutor** de un esquema tipo DAgger.

## Punto de partida (lo que ya existe)

- `src/rl/visual/train_sac_dagger.py` — ya implementa el **patrón DAgger**:
  - Cada step pide al actor SAC `sampled` (categórica) y `greedy` (argmax).
  - `sampled == greedy` → SAC actuó confiado, se usa su acción.
  - `sampled != greedy` → era exploración → **el tutor decide en su lugar**.
  - El decay de exploración **emerge solo**: actor sin entrenar ≈ uniforme →
    tutor casi siempre; según converge el softmax, el tutor se llama menos.
  - Transiciones del tutor entran al replay buffer; se anota `frame + regla`
    en `<run_dir>/dagger/` (dataset IL reutilizable).
- `src/rl/visual/symbolic.py` — el tutor **actual**, pero es un **PLACEHOLDER**:
  política reactiva en Python sobre el vector de estado normalizado. Vive en el
  mismo espacio discreto por-step que SAC. **No** invoca el HTN real.
- El run `sac_dagger` **aún no se ha ejecutado**.

## El problema de diseño: choque de granularidad

| | SAC (Python) | HTN (`src/htn/`, JS) |
|---|---|---|
| Granularidad | 1 acción discreta por `/step` | macro-tarea multi-tick |
| Interfaz | `/step {action}` → `{state, events}` | `runChopProgression` async run-to-completion |
| Mecanismo | `executeRLAction` (1 acción, ~50-250ms) | pathfinder + `bot.dig` + watchdog |

El HTN **no cabe** directamente en un slot de "una acción discreta por step".

## Decisión tomada: **camino B2 — proyectar la traza del macro**

Opciones consideradas:
- **A** — oráculo reactivo por-step en JS (`/htn_action`): mínimo cambio pero
  conceptualmente ≈ `symbolic.py`. Descartado.
- **B1** — HTN macro real + pérdida IL separada (DAgger clásico): mezcla RL+IL,
  rompe contigüidad del MDP. Descartado.
- **B2 (ELEGIDO)** — el HTN macro corre de verdad y su traza tick-a-tick se
  **proyecta a acciones discretas** que entran al buffer de SAC. Un solo
  objetivo RL, MDP contiguo, usa el experto real.
- **C** — refactor del HTN a generador step-able: lo más elegante pero el
  refactor más grande. Sobra para un TFG en cierre. Descartado.

## Plan de implementación (3 componentes)

### 1. Bridge — `src/agents/types/rl_agent.js`: nuevo macro acotado
Nuevo endpoint `POST /htn_macro {budget_steps: K}` junto a `/step`, `/reset`,
`/world_reset`:
- Corre un macro **bounded**: `mineBlock(bot, mcData, woodType, 1, …,
  {useFovCone:true})` — "consigue **un** tronco" (respeta la regla del cono FOV).
  Termina al romper 1 log o al agotar `budget_steps` ticks.
- Mientras corre, samplea `{state, events}` cada `STEP_HOLD_MS` (misma cadencia
  que `executeRLAction`). Cada muestra = un pseudo-step.
- Etiqueta la **acción discreta dominante** por pseudo-step:
  - `bot.dig` / bloque roto → `attack`
  - Δyaw > +ε → `camera_left`; Δyaw < −ε → `camera_right`
    (ojo signo: en el código `camera_left` es `yaw + RAD`)
  - Δpitch < −ε → `camera_up`; > +ε → `camera_down`
  - Δpos hacia delante → `move_forward_sprint` (Δy>0 → `move_forward_jump`)
- Devuelve `{substeps: [{action, state, events}, …]}`.
- ⚠️ Reconciliar `canDig`: el RL bot usa `bot.dig` directo; el chop progression
  monta su pathfinder con `canDig:false`. El macro necesita su config propia.

### 2. Env — `src/rl/shared/env.py`: separar transporte de procesamiento
Extraer de `step()` (líneas ~175-239) un método puro:
```python
def _process_response(self, resp) -> (obs, reward, terminated, truncated, info)
```
`step()` queda: `resp = self._post("/step", payload)` → `_process_response(resp)`.
Así cada substep del macro pasa por el **mismo reward/terminación/gate** que un
step normal (gate de "tronco roto", shaping de approach, deltas de inventario).
Refactor de bajo riesgo: no cambia el comportamiento de DQN/PPO/SAC actuales.

### 3. Loop — nuevo `src/rl/visual/train_sac_htn.py` (clon de `train_sac_dagger.py`)
En el `else` de exploración (líneas ~322-328), en vez de `symbolic.act`:
```python
resp = post("/htn_macro", {"budget_steps": K})
for sub in resp["substeps"]:
    next_obs, r, term, trunc, info = env._process_response(sub)
    agent.step(obs, sub["action_idx"], r, next_obs, term or trunc)
    annotator.annotate(...)            # frame + acción + rule="htn_macro"
    obs = next_obs; global_step += 1
    if term or trunc:
        break
```
`DaggerAnnotator` y `metrics.log_step` se reutilizan; solo cambia
`rule`/`action_src` a `htn`.

## Riesgos (los mismos que mataron Hybrid SAC)

1. **Wall-clock**: en exploración temprana el actor es casi uniforme → el macro
   se dispara casi cada step → pathfinder(≤2s)+dig(≤8s) por macro = episodios de
   decenas de minutos (mismo problema de 20min/ep del Hybrid SAC).
   **Mitigación obligatoria**: `budget_steps` pequeño (5-10 ticks) y/o cap de
   macros por episodio.
2. **Fidelidad del mapeo**: pathfinder hace strafe/diagonal/jump no siempre
   expresable en las 7 acciones → traza proyectada aproximada. Documentar en la
   memoria como "behavioral projection del experto HTN".

## Próxima acción concreta al retomar

**De-riskear el punto 1 antes de invertir en el refactor**: prototipar solo el
endpoint `/htn_macro` en `rl_agent.js` y medir el **wall-clock real** de un macro
con `budget_steps` pequeño. Si el coste por macro es viable, seguir con el
refactor de `env.py` y el nuevo loop. Si no, replantear el budget o volver a
evaluar B1.

## Referencias cruzadas

- Cierre del bloque RL: `docs/bitacora/cierre_TFG_2026-04-20_2026-05-05.tex`
- Fases DQN previas: `docs/bitacora/rl_next.md`
- Regla del cono FOV: siempre `useFovCone:true` en búsqueda de bloques.
