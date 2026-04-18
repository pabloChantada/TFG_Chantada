/**
 * ILAgent — Agente de Imitation Learning.
 *
 * Loop principal:
 *   1. Captura screenshot del prismarine-viewer via Puppeteer
 *   2. Envía la imagen al inference server (Python, port 8765)
 *   3. Recibe la acción predicha por el modelo
 *   4. Mapea la acción discreta predicha y la ejecuta
 *   5. Repite
 *
 * Requisito: inference server corriendo antes de iniciar el agente.
 */

import fs   from 'fs'
import path from 'path'
import http from 'http'

import { BaseAgent }               from './base_agent.js'
import { logInfo, logError }       from '../logging.js'
import puppeteer                   from 'puppeteer'
import minecraftData               from 'minecraft-data'
import { findNearestVisibleBlock } from '../../htn/primitives/blocks.js'

// ── Configuración ─────────────────────────────────────────────────────────────
const INFERENCE_HOST  = 'localhost'
const INFERENCE_PORT  = 8765
const INFERENCE_PATH  = '/predict'

const STEP_INTERVAL_MS  = 1000   // Mínimo ms entre pasos del modelo
const MOVE_HOLD_MS      = 100   // Duración de acciones de movimiento
const BROWSER_WARMUP_MS = 2000  // Espera inicial del viewer
const VIEWER_WIDTH      = 854
const VIEWER_HEIGHT     = 480

const ACTION_SWITCH_THRESHOLD = 0.7  // Confianza mínima para cambiar de acción

const CAMERA_DEAD_ZONE = 0.01   // rad (~0.6°) — deltas menores se ignoran (ruido)
const PITCH_DECAY      = 0.9   // Factor de decaimiento del pitch hacia horizonte cada step

const LOG_TYPES = ['oak_log', 'birch_log', 'spruce_log', 'dark_oak_log', 'jungle_log', 'acacia_log']

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms))
}

function clampPitch(p) {
    return Math.max(-Math.PI / 2, Math.min(Math.PI / 2, p))
}

async function releaseAll(bot) {
    for (const c of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
        bot.setControlState(c, false)
    }
}

// ── Ejecución de acciones ─────────────────────────────────────────────────────

/**
 * Ejecuta una acción discreta + delta de cámara continuo.
 *
 * Acciones discretas: move_forward_walk, move_forward_sprint, move_backward_walk,
 *   move_left, move_right, jump, sneak, attack, equip_wooden_axe
 *
 * Camera delta: {dyaw, dpitch} en radianes, aplicado siempre tras la acción.
 */
async function executeILAction(bot, action, cameraDelta = null) {
    await releaseAll(bot)
    let skipCameraDelta = false

    switch (action) {

        // ── Acciones de movimiento ────────────────────────────────────────────

        case 'move_forward_jump':
            bot.setControlState('forward', true)
            bot.setControlState('sprint', true)
            bot.setControlState('jump', true)
            await sleep(MOVE_HOLD_MS)
            await releaseAll(bot)
            break

        case 'move_forward_sprint':
            bot.setControlState('forward', true)
            bot.setControlState('sprint', true)
            await sleep(MOVE_HOLD_MS)
            await releaseAll(bot)
            break

        case 'move_backward_walk':
            bot.setControlState('back', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('back', false)
            break

        case 'move_left':
            bot.setControlState('left', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('left', false)
            break

        case 'move_right':
            bot.setControlState('right', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('right', false)
            break

        case 'jump':
            bot.setControlState('jump', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('jump', false)
            break

        case 'sneak':
            bot.setControlState('sneak', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('sneak', false)
            break

        // ── Ataque → primitiva HTN: minar bloque en cursor ───────────────────

        case 'attack': {
            const block = bot.blockAtCursor(4.5)
            if (block && block.type !== 0 && bot.canDigBlock(block)) {
                const bp = block.position.offset(0.5, 0.5, 0.5)
                await bot.lookAt(bp, false)
                try { await bot.dig(block, 'ignore') } catch (_) {}
                skipCameraDelta = true
            }
            break
        }

        // ── Equipar hacha ─────────────────────────────────────────────────────

        case 'equip_wooden_axe': {
            const item = bot.inventory.items().find(i => i.name === 'wooden_axe')
            if (item) {
                try { await bot.equip(item, 'hand') } catch (_) {}
            }
            break
        }

        default:
            logError('ILAgent', new Error(`Acción desconocida: "${action}"`))
    }

    // ── Aplicar delta de cámara continuo (salvo durante attack, donde se fija al bloque) ──
    if (cameraDelta && !skipCameraDelta) {
        let { dyaw, dpitch } = cameraDelta

        // Dead zone: ignorar deltas menores al umbral (ruido del modelo)
        const dyawFiltered   = Math.abs(dyaw)   < CAMERA_DEAD_ZONE ? 0 : dyaw
        const dpitchFiltered = Math.abs(dpitch)  < CAMERA_DEAD_ZONE ? 0 : dpitch

        if (dyaw !== dyawFiltered || dpitch !== dpitchFiltered) {
            logInfo('Camera', `dead-zone: dyaw ${dyaw.toFixed(4)}→${dyawFiltered.toFixed(4)}  dpitch ${dpitch.toFixed(4)}→${dpitchFiltered.toFixed(4)}`)
        }

        // Pitch decay: tirar hacia horizonte (0) para contrarrestar deriva
        const rawPitch    = bot.entity.pitch + dpitchFiltered
        const decayedPitch = rawPitch * PITCH_DECAY

        if (Math.abs(rawPitch - decayedPitch) > 0.001) {
            logInfo('Camera', `pitch-decay: ${rawPitch.toFixed(4)} → ${decayedPitch.toFixed(4)}`)
        }

        await bot.look(
            bot.entity.yaw + dyawFiltered,
            clampPitch(decayedPitch),
            false,
        )
    }
}

// ── ILAgent ───────────────────────────────────────────────────────────────────

export class ILAgent extends BaseAgent {
    constructor(agentName, inferencePort = INFERENCE_PORT, recordInference = false) {
        super(agentName, 'il')
        this.memoryPath     = `src/agents/memories/${agentName}_memory.json`
        this.inferencePort  = inferencePort
        this._browser       = null
        this._page          = null
        this._running       = false
        this._stepCount     = 0
        this._mcData        = null
        // Inference recording
        this._record        = recordInference
        this._recordDir     = null
        this._recordStream  = null
        this._lastAction    = null  // Última acción ejecutada (para umbral de cambio)
    }

    // ── Inicio ────────────────────────────────────────────────────────────────

    async start(settings, viewerPort) {
        // Leer URL del inference server desde settings (pasada por add_agent.js)
        if (settings.inference_url) {
            try {
                const url = new URL(settings.inference_url)
                this.inferencePort = parseInt(url.port) || INFERENCE_PORT
            } catch (_) {}
        }

        try {
            await this.connectBot(settings)
            this._mcData = minecraftData(this.bot.version)
            await this.clearMemory()
            await this.setupViewer(viewerPort)
            this._setupErrorHandlers()

            logInfo(this.name, 'Iniciando Puppeteer...')
            await this._startPuppeteer(viewerPort)

            // Inference recording setup
            if (this._record) {
                const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
                this._recordDir = path.join('data', 'inference_logs', `${this.name}_${ts}`)
                fs.mkdirSync(path.join(this._recordDir, 'screenshots'), { recursive: true })
                this._recordStream = fs.createWriteStream(
                    path.join(this._recordDir, 'log.jsonl'), { flags: 'a' }
                )
                logInfo(this.name, `Recording inferencia en: ${this._recordDir}`)
            }

            logInfo(this.name, `Inference server: http://${INFERENCE_HOST}:${this.inferencePort}${INFERENCE_PATH}`)
            logInfo(this.name, 'Arrancando loop IL...')
            await this.runLogic()

        } catch (error) {
            logError(this.name, new Error(`Error en start: ${error.message}`))
            await this._shutdown()
            process.exit(1)
        }
    }

    async _startPuppeteer(viewerPort) {
        this._browser = await puppeteer.launch({ headless: 'new' })
        this._page    = await this._browser.newPage()
        await this._page.setViewport({ width: VIEWER_WIDTH, height: VIEWER_HEIGHT })
        await this._page.goto(`http://localhost:${viewerPort}`, {
            waitUntil: 'domcontentloaded',
            timeout:   15000,
        })
        await this._page.waitForSelector('canvas', { timeout: 10000 })
        await sleep(BROWSER_WARMUP_MS)
        logInfo(this.name, 'Puppeteer listo')
    }

    // ── Loop principal ────────────────────────────────────────────────────────

    async runLogic() {
        this._running = true
        try {
            while (this._running) {
                const t0 = Date.now()

                const imgBuffer = await this._captureScreenshot()
                if (!imgBuffer) { await sleep(STEP_INTERVAL_MS); continue }

                const prediction = await this._requestPrediction(imgBuffer)
                if (!prediction) { await sleep(STEP_INTERVAL_MS); continue }

                const { action: rawAction, confidence, camera_delta, top5 } = prediction
                this._stepCount++

                // Umbral de confianza: solo cambiar de acción si supera el threshold
                let action = rawAction
                if (this._lastAction && action !== this._lastAction && confidence < ACTION_SWITCH_THRESHOLD) {
                    logInfo(this.name, `step=${this._stepCount}  pred=${action} (${(confidence * 100).toFixed(1)}%) < ${(ACTION_SWITCH_THRESHOLD * 100)}% → mantiene ${this._lastAction}`)
                    action = this._lastAction
                } else {
                    const camStr = camera_delta
                        ? `  cam=(${camera_delta.dyaw.toFixed(3)}, ${camera_delta.dpitch.toFixed(3)})`
                        : ''
                    logInfo(this.name, `step=${this._stepCount}  action=${action}  conf=${(confidence * 100).toFixed(1)}%${camStr}`)
                }
                this._lastAction = action

                // Record inference step
                if (this._record && this._recordStream) {
                    const pos  = this.bot.entity.position
                    const tree = this._getTreeInfo()
                    const stepNum = String(this._stepCount).padStart(5, '0')
                    const imgFile = `step_${stepNum}.png`
                    fs.writeFileSync(
                        path.join(this._recordDir, 'screenshots', imgFile),
                        imgBuffer,
                    )
                    this._recordStream.write(JSON.stringify({
                        step:     this._stepCount,
                        image:    imgFile,
                        action,
                        confidence,
                        camera_delta,
                        top5,
                        state: {
                            x: Math.round(pos.x * 100) / 100,
                            y: Math.round(pos.y * 100) / 100,
                            z: Math.round(pos.z * 100) / 100,
                            yaw:   Math.round(this.bot.entity.yaw * 1000) / 1000,
                            pitch: Math.round(this.bot.entity.pitch * 1000) / 1000,
                        },
                        tree_visible:  tree.tree_visible,
                        tree_distance: tree.tree_distance,
                        block_at_cursor: !!this.bot.blockAtCursor(4.5),
                    }) + '\n')
                }

                await executeILAction(this.bot, action, camera_delta)

                const remaining = STEP_INTERVAL_MS - (Date.now() - t0)
                if (remaining > 0) await sleep(remaining)
            }
        } catch (error) {
            logError(this.name, new Error(`Error en loop: ${error.message}`))
        } finally {
            await this._shutdown()
            process.exit(0)
        }
    }

    // ── Captura de pantalla ───────────────────────────────────────────────────

    async _captureScreenshot() {
        try {
            const canvas = await this._page.$('canvas')
            if (!canvas) return null
            return await canvas.screenshot({ type: 'png' })
        } catch (e) {
            logError(this.name, new Error(`Screenshot fallido: ${e.message}`))
            return null
        }
    }

    // ── Llamada al inference server ───────────────────────────────────────────

    _getTreeInfo() {
        try {
            for (const logType of LOG_TYPES) {
                const block = findNearestVisibleBlock(this.bot, this._mcData, logType, 32)
                if (block) {
                    const dist = this.bot.entity.position.distanceTo(block.position)
                    return { tree_visible: 1, tree_distance: Math.round(dist * 10) / 10, block }
                }
            }
            return { tree_visible: 0, tree_distance: null, block: null }
        } catch (_) {
            return { tree_visible: 0, tree_distance: null, block: null }
        }
    }

    _requestPrediction(pngBuffer) {
        const pos   = this.bot.entity.position
        const tree  = this._getTreeInfo()
        const state = [pos.x, pos.y, pos.z, this.bot.entity.yaw, this.bot.entity.pitch, tree.tree_visible, tree.tree_distance]
            .map(v => Number(v).toFixed(4)).join(',')

        return new Promise((resolve) => {
            const options = {
                hostname: INFERENCE_HOST,
                port:     this.inferencePort,
                path:     INFERENCE_PATH,
                method:   'POST',
                headers:  {
                    'Content-Type':   'image/png',
                    'Content-Length': pngBuffer.length,
                    'X-Bot-State':    state,
                },
            }

            const req = http.request(options, (res) => {
                let data = ''
                res.on('data', chunk => data += chunk)
                res.on('end', () => {
                    try {
                        resolve(JSON.parse(data))
                    } catch (_) {
                        logError(this.name, new Error('Respuesta inválida del inference server'))
                        resolve(null)
                    }
                })
            })

            req.on('error', (e) => {
                logError(this.name, new Error(`Inference server no disponible: ${e.message}`))
                resolve(null)
            })

            req.write(pngBuffer)
            req.end()
        })
    }

    // ── Manejo de errores y shutdown ──────────────────────────────────────────

    _setupErrorHandlers() {
        this._onError  = (err)    => { logError(this.name, new Error(`Bot error: ${err.message}`)); this._shutdown().then(() => process.exit(1)) }
        this._onKicked = (reason) => { logError(this.name, new Error(`Bot kicked: ${reason}`));     this._shutdown().then(() => process.exit(1)) }
        this.bot.on('error',  this._onError)
        this.bot.on('kicked', this._onKicked)
    }

    async _shutdown() {
        this._running = false

        if (this._recordStream) {
            await new Promise(r => this._recordStream.end(r))
            this._recordStream = null
            logInfo(this.name, `Inference log guardado en: ${this._recordDir}`)
        }

        if (this._browser) {
            try { await this._browser.close() } catch (_) {}
            this._browser = null
            this._page    = null
        }

        if (this.bot) {
            if (this._onError)  this.bot.removeListener('error',  this._onError)
            if (this._onKicked) this.bot.removeListener('kicked', this._onKicked)
        }

        await super.shutdown()
    }

    async shutdown() {
        logInfo(this.name, 'Cerrando ILAgent...')
        await this._shutdown()
    }
}
