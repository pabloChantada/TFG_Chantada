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

import http from 'http'

import { BaseAgent }               from './base_agent.js'
import { logInfo, logError }       from '../logging.js'
import puppeteer                   from 'puppeteer'

// ── Configuración ─────────────────────────────────────────────────────────────
const INFERENCE_HOST  = 'localhost'
const INFERENCE_PORT  = 8765
const INFERENCE_PATH  = '/predict'

const STEP_INTERVAL_MS  = 800   // Mínimo ms entre pasos del modelo
const MOVE_HOLD_MS      = 250   // Duración de acciones de movimiento
const BROWSER_WARMUP_MS = 2000  // Espera inicial del viewer
const VIEWER_WIDTH      = 854
const VIEWER_HEIGHT     = 480

const DEG_TO_RAD = Math.PI / 180

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

    switch (action) {

        // ── Acciones de movimiento ────────────────────────────────────────────

        case 'move_forward_walk':
            bot.setControlState('forward', true)
            await sleep(MOVE_HOLD_MS)
            bot.setControlState('forward', false)
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
            await sleep(200)
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
                try { await bot.dig(block, 'ignore') } catch (_) {}
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

    // ── Aplicar delta de cámara continuo (siempre, tras la acción) ────────
    if (cameraDelta) {
        const { dyaw, dpitch } = cameraDelta
        await bot.look(
            bot.entity.yaw + dyaw,
            clampPitch(bot.entity.pitch + dpitch),
            false,
        )
    }
}

// ── ILAgent ───────────────────────────────────────────────────────────────────

export class ILAgent extends BaseAgent {
    constructor(agentName, inferencePort = INFERENCE_PORT) {
        super(agentName, 'il')
        this.memoryPath     = `src/agents/memories/${agentName}_memory.json`
        this.inferencePort  = inferencePort
        this._browser       = null
        this._page          = null
        this._running       = false
        this._stepCount     = 0
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
            await this.clearMemory()
            await this.setupViewer(viewerPort)
            this._setupErrorHandlers()

            logInfo(this.name, 'Iniciando Puppeteer...')
            await this._startPuppeteer(viewerPort)

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

                const { action, confidence, camera_delta } = prediction
                this._stepCount++
                const camStr = camera_delta
                    ? `  cam=(${camera_delta.dyaw.toFixed(3)}, ${camera_delta.dpitch.toFixed(3)})`
                    : ''
                logInfo(this.name, `step=${this._stepCount}  action=${action}  conf=${(confidence * 100).toFixed(1)}%${camStr}`)

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

    _requestPrediction(pngBuffer) {
        const pos   = this.bot.entity.position
        const state = [pos.x, pos.y, pos.z, this.bot.entity.yaw, this.bot.entity.pitch]
            .map(v => v.toFixed(4)).join(',')

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
