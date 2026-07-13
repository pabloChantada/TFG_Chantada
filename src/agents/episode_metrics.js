/**
 * episode_metrics.js — Registro unificado de métricas por episodio.
 *
 * Escribe un registro por episodio en `data/eval/<technique>/eval.jsonl`, con el
 * MISMO esquema para HTN / IL / RL, de modo que la comparativa del Cap. 8 sea justa:
 *
 *   {
 *     technique, ts, success_raw,        // éxito reportado por la propia técnica
 *     logs_collected, logs_broken, target,
 *     steps, time_s
 *   }
 *
 * La métrica de éxito "unificada" (recoger >= K Y romper >= 1) NO se decide aquí:
 * se aplica en src/evaluation/summarize_eval.py sobre logs_collected/logs_broken,
 * para poder usar el mismo umbral K en las tres técnicas.
 *
 * Uso (desde el flujo de un agente):
 *   import { startEpisodeTracker } from '../agents/episode_metrics.js'
 *   const tracker = startEpisodeTracker(bot, { technique: 'htn', target: logCount })
 *   ... // ejecutar el episodio
 *   tracker.finish(result?.success)
 */

import fs   from 'fs'
import path from 'path'

const LOG_TYPES = new Set([
    'oak_log', 'birch_log', 'spruce_log',
    'dark_oak_log', 'jungle_log', 'acacia_log',
    'mangrove_log', 'cherry_log',
])

const DEFAULT_OUT_DIR = 'data/eval'

/**
 * Arranca el seguimiento de un episodio. Devuelve un objeto con finish(success).
 *
 * @param {import('mineflayer').Bot} bot
 * @param {Object} opts
 * @param {string} [opts.technique='htn']  Etiqueta de la técnica (htn|il|rl).
 * @param {number} [opts.target=5]         Troncos objetivo del episodio.
 * @param {string} [opts.outDir]           Directorio base de salida.
 */
export function startEpisodeTracker(bot, { technique = 'htn', target = 5, outDir = DEFAULT_OUT_DIR, label = null } = {}) {
    // Nombre del fichero: eval.jsonl o eval_<label>.jsonl (p.ej. eval_gru.jsonl)
    // para no sobreescribir al evaluar varios modelos de la misma técnica.
    const safeLabel = label ? String(label).replace(/[^a-zA-Z0-9_-]/g, '') : ''
    const evalFile  = safeLabel ? `eval_${safeLabel}.jsonl` : 'eval.jsonl'
    const t0 = Date.now()
    let logsBroken = 0
    // Conteo de acciones ejecutadas en el episodio (best-effort; solo lo alimentan
    // las técnicas que llaman a recordAction, p.ej. el ILAgent en modo eval). Sirve
    // para ver la distribución de acciones en inferencia y detectar el colapso a la
    // clase mayoritaria (p.ej. ConvLSTM → casi todo 'attack').
    const actionCounts = {}

    // Cuenta de troncos rotos por el bot (best-effort vía evento de minado).
    const onDig = (block) => {
        if (block && LOG_TYPES.has(block.name)) logsBroken++
    }
    bot.on('diggingCompleted', onDig)

    const countLogsInInventory = () => {
        try {
            return bot.inventory.items()
                .filter(it => LOG_TYPES.has(it.name))
                .reduce((sum, it) => sum + it.count, 0)
        } catch (_) {
            return 0
        }
    }

    return {
        /**
         * Estado actual del episodio (para decidir terminación en bucles sin
         * condición de éxito explícita, como el ILAgent).
         * @returns {{logs_collected: number, logs_broken: number, elapsed_s: number}}
         */
        state() {
            return {
                logs_collected: countLogsInInventory(),
                logs_broken:    logsBroken,
                elapsed_s:      Math.round((Date.now() - t0) / 100) / 10,
            }
        },

        /**
         * Acumula una acción ejecutada para el histograma del episodio. Llamar una
         * vez por paso con la acción finalmente ejecutada (tras el umbral de cambio).
         * @param {string} action
         */
        recordAction(action) {
            if (!action) return
            actionCounts[action] = (actionCounts[action] || 0) + 1
        },

        /**
         * Cierra el episodio y persiste el registro.
         * @param {boolean} [successRaw] Éxito reportado por la técnica.
         * @param {number}  [stepsOverride] Nº de pasos del agente. Si se omite, se
         *   toma del dataset recorder (solo disponible al grabar). El ILAgent en
         *   modo eval no graba dataset, así que pasa aquí su propio contador.
         * @returns {Object} El registro escrito.
         */
        finish(successRaw, stepsOverride) {
            try { bot.removeListener('diggingCompleted', onDig) } catch (_) {}

            const logsCollected = countLogsInInventory()
            const elapsedS = Math.round((Date.now() - t0) / 100) / 10
            const steps    = stepsOverride ?? bot._datasetRecorder?._frameCount ?? null

            const record = {
                technique,
                ts:             new Date().toISOString(),
                success_raw:    successRaw === undefined ? null : !!successRaw,
                logs_collected: logsCollected,
                logs_broken:    logsBroken,
                target,
                steps,
                time_s:         elapsedS,
                action_counts:  Object.keys(actionCounts).length ? actionCounts : null,
            }

            try {
                const dir = path.join(outDir, technique)
                fs.mkdirSync(dir, { recursive: true })
                fs.appendFileSync(path.join(dir, evalFile), JSON.stringify(record) + '\n')
            } catch (e) {
                console.warn(`[episode_metrics] No se pudo escribir el registro: ${e.message}`)
            }

            console.log(
                `[eval] ${technique} | success_raw=${record.success_raw} ` +
                `logs=${logsCollected}/${target} broken=${logsBroken} ` +
                `steps=${steps} t=${elapsedS}s`
            )
            return record
        },
    }
}
