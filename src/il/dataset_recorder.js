/**
 * DatasetRecorder - Recolección de dataset con acciones discretas de bajo nivel.
 *
 * En cada frame capturado se detecta la acción discreta real que ejecuta el bot
 * leyendo bot.controlState (movimiento) y el delta de yaw/pitch entre capturas (cámara).
 * Los frames sin acción detectable se descartan.
 *
 * Acciones discretas detectables:
 *   move_forward_sprint  - forward + sprint activos
 *   move_forward_walk    - forward activo
 *   move_backward_walk   - back activo
 *   move_left            - left activo
 *   move_right           - right activo
 *   jump                 - jump activo
 *   sneak                - sneak activo
 *   attack               - bot.targetDigBlock no nulo
 *
 * Además, cada frame incluye camera_delta: {dyaw, dpitch} (radianes),
 * el cambio de cámara respecto al frame anterior, grabado de forma continua
 * e independiente de la acción discreta.
 *
 * Nota: equip_wooden_axe no se detecta automáticamente (acción instantánea sin
 * estado persistente). Se puede añadir en sesiones de teleoperación manual.
 *
 * Metadatos adicionales grabados por frame:
 *   state        - {x, y, z, yaw, pitch} del bot en el momento de captura
 *   tree_visible - si hay un tronco visible dentro de 32 bloques
 *   tree_distance - distancia al tronco visible más cercano
 *
 * Integración:
 *   import { setupRecorder } from '../il/dataset_recorder.js'
 *   await setupRecorder(bot, mcData)
 *   bot._datasetRecorder?.setWoodType('oak_log')  // desde HTN primitives
 */

import fs      from 'fs'
import path    from 'path'
import puppeteer from 'puppeteer'
import { findNearestVisibleBlock } from '../htn/primitives/blocks.js'

// - Timing ----------------------------------
const TICK_MS              = 200   // Resolución del timer
const DEFAULT_INTERVAL_MS  = 800   // Intervalo de captura por defecto

// - Screenshot / viewer ----------------------------
const VIEWER_WIDTH      = 854
const VIEWER_HEIGHT     = 480
const BROWSER_WARMUP_MS = 2000

// - Output directory -----------------------------
const RECORDINGS_DIR = 'data/recordings'

// ---------------------------------------

function nextSessionNumber() {
    let maxN = 0
    try {
        if (fs.existsSync(RECORDINGS_DIR)) {
            for (const entry of fs.readdirSync(RECORDINGS_DIR)) {
                const m = entry.match(/^Rec_(\d+)_/)
                if (m) maxN = Math.max(maxN, parseInt(m[1], 10))
            }
        }
    } catch (_) {}
    return maxN + 1
}

function generateSessionId() {
    const n    = nextSessionNumber()
    const hash = Math.random().toString(36).slice(2, 8).padEnd(6, '0')
    return `Rec_${n}_${hash}`
}

// - DatasetRecorder ------------------------------

class DatasetRecorder {
    /**
     * @param {import('mineflayer').Bot} bot
     * @param {Object} mcData  - resultado de minecraftData(bot.version)
     */
    /**
     * @param {import('mineflayer').Bot} bot
     * @param {Object} mcData  - resultado de minecraftData(bot.version)
     * @param {Object} [opts]
     * @param {number} [opts.intervalMs] - Intervalo de captura en ms (default: 800)
     */
    constructor(bot, mcData, opts = {}) {
        this.bot    = bot
        this.mcData = mcData

        // Session
        this._sessionId  = generateSessionId()
        this._frameCount = 0

        // Runtime
        this._running    = false
        this._timer      = null
        this._capturing  = false
        this._lastCaptureTime = 0
        this._intervalMs = opts.intervalMs || DEFAULT_INTERVAL_MS

        // Seguimiento de cámara (delta entre capturas)
        this._prevYaw   = 0
        this._prevPitch = 0

        // Puppeteer
        this._browser = null
        this._page    = null

        // Output
        this._screenshotsDir = null
        this._jsonlPath      = null
        this._jsonlStream    = null

        // Tipo de madera (para metadatos tree_visible / tree_distance)
        this._woodType = null
    }

    // - Public API ------------------------------

    /**
     * Tipo de madera que el bot está buscando/talando.
     * Usado para calcular tree_visible y tree_distance.
     * @param {string} woodType  e.g. 'oak_log'
     */
    setWoodType(woodType) {
        this._woodType = woodType
    }

    // - Lifecycle -------------------------------

    async start(viewerPort = 3000) {
        this._screenshotsDir = path.join(RECORDINGS_DIR, `${this._sessionId}_screenshots`)
        this._jsonlPath      = path.join(RECORDINGS_DIR, `${this._sessionId}.jsonl`)

        fs.mkdirSync(this._screenshotsDir, { recursive: true })
        this._jsonlStream = fs.createWriteStream(this._jsonlPath, { flags: 'a' })

        console.log(`[DatasetRecorder] Session  : ${this._sessionId}`)
        console.log(`[DatasetRecorder] JSONL    : ${this._jsonlPath}`)
        console.log(`[DatasetRecorder] Viewer   : http://localhost:${viewerPort}`)

        this._browser = await puppeteer.launch({ headless: 'new' })
        this._page    = await this._browser.newPage()
        await this._page.setViewport({ width: VIEWER_WIDTH, height: VIEWER_HEIGHT })
        await this._page.goto(`http://localhost:${viewerPort}`, {
            waitUntil: 'domcontentloaded',
            timeout:   15000
        })
        await this._page.waitForSelector('canvas', { timeout: 10000 })
        await new Promise(r => setTimeout(r, BROWSER_WARMUP_MS))

        // Inicializar seguimiento de cámara
        this._prevYaw   = this.bot.entity.yaw
        this._prevPitch = this.bot.entity.pitch

        this._running         = true
        this._lastCaptureTime = Date.now()
        this._scheduleTick()

        console.log(`[DatasetRecorder] Iniciado — intervalo ${this._intervalMs}ms`)
    }

    async stop() {
        this._running = false
        if (this._timer) { clearTimeout(this._timer); this._timer = null }

        await new Promise(r => setTimeout(r, TICK_MS * 3))

        if (this._jsonlStream) {
            await new Promise(r => this._jsonlStream.end(r))
            this._jsonlStream = null
        }

        if (this._browser) {
            try { await this._browser.close() } catch (_) {}
            this._browser = null
            this._page    = null
        }

        console.log(
            `[DatasetRecorder] Parado — ${this._frameCount} frames guardados` +
            ` — sesión ${this._sessionId}`
        )
    }

    // - Internal timer loop --------------------------

    _scheduleTick() {
        if (!this._running) return
        this._timer = setTimeout(() => this._tick(), TICK_MS)
    }

    async _tick() {
        if (!this._running || this._capturing) {
            this._scheduleTick()
            return
        }

        const now = Date.now()
        if (now < this._lastCaptureTime + this._intervalMs) {
            this._scheduleTick()
            return
        }

        this._capturing = true
        try {
            await this._capture()
        } catch (e) {
            console.warn(`[DatasetRecorder] Error de captura: ${e.message}`)
        }
        this._capturing       = false
        this._lastCaptureTime = Date.now()

        this._scheduleTick()
    }

    // - Detección de acción discreta + delta de cámara ------

    /**
     * Calcula el delta de cámara respecto al frame anterior (en radianes).
     * Se graba siempre, independientemente de la acción discreta.
     */
    _getCameraDelta() {
        let dyaw   = this.bot.entity.yaw   - this._prevYaw
        let dpitch = this.bot.entity.pitch - this._prevPitch
        // Normalizar yaw delta a [-π, π] para evitar saltos por wrapping
        while (dyaw >  Math.PI) dyaw -= 2 * Math.PI
        while (dyaw < -Math.PI) dyaw += 2 * Math.PI
        return {
            dyaw:   Math.round(dyaw   * 10000) / 10000,
            dpitch: Math.round(dpitch * 10000) / 10000,
        }
    }

    /**
     * Detecta la acción discreta que el bot está ejecutando en este momento.
     * Prioridad: movimiento > ataque.
     * La cámara ya NO es una acción discreta (se graba como camera_delta continuo).
     * Devuelve null si no hay acción detectable (frame a descartar).
     */
    _detectAction() {
        const cs = this.bot.controlState

        // Jump + forward primero (pathfinding sube bloques)
        if (cs.forward && cs.jump)   return 'move_forward_jump'
        if (cs.forward)              return 'move_forward_sprint'
        if (cs.back)                 return 'move_backward_walk'
        if (cs.left)                 return 'move_left'
        if (cs.right)                return 'move_right'
        if (cs.jump)                 return 'jump'
        if (cs.sneak)                return 'sneak'

        // Ataque: bot.targetDigBlock es no nulo mientras bot.dig() está activo
        if (this.bot.targetDigBlock) return 'attack'

        return null  // sin acción detectable → descartar frame
    }

    /**
     * Metadatos del árbol más cercano (para auxiliar al modelo).
     */
    _getTreeInfo() {
        if (!this._woodType) return { tree_visible: false, tree_distance: null }
        try {
            const block = findNearestVisibleBlock(this.bot, this.mcData, this._woodType, 32)
            if (!block) return { tree_visible: false, tree_distance: null }
            const dist = Math.round(
                this.bot.entity.position.distanceTo(block.position) * 10
            ) / 10
            return { tree_visible: true, tree_distance: dist }
        } catch (_) {
            return { tree_visible: false, tree_distance: null }
        }
    }

    // - Frame capture -----------------------------

    async _capture() {
        if (!this._page) return

        const curYaw   = this.bot.entity.yaw
        const curPitch = this.bot.entity.pitch

        const camera_delta = this._getCameraDelta()
        const action       = this._detectAction()

        // Actualizar seguimiento de cámara siempre (tanto si se guarda como si no)
        this._prevYaw   = curYaw
        this._prevPitch = curPitch

        if (action === null) return  // frame sin acción → descartar

        const canvas = await this._page.$('canvas')
        if (!canvas) return

        const { tree_visible, tree_distance } = this._getTreeInfo()
        const pos      = this.bot.entity.position
        const frameNum = String(this._frameCount).padStart(5, '0')
        const ts       = new Date().toISOString().replace(/:/g, '-').replace('.', '-')
        const filename = `frame_${frameNum}_${ts}.png`
        const imgPath  = path.join(this._screenshotsDir, filename)

        await canvas.screenshot({ type: 'png', path: imgPath })

        const entry = {
            image: imgPath.replace(/\\/g, '/'),
            state: {
                x:     Math.round(pos.x     * 1000) / 1000,
                y:     Math.round(pos.y     * 1000) / 1000,
                z:     Math.round(pos.z     * 1000) / 1000,
                yaw:   Math.round(curYaw    * 10000) / 10000,
                pitch: Math.round(curPitch  * 10000) / 10000
            },
            action,
            camera_delta,
            session_id:   this._sessionId,
            tree_visible,
            tree_distance
        }

        this._jsonlStream.write(JSON.stringify(entry) + '\n')
        this._frameCount++
    }
}

// - Public factory ------------------------------

/**
 * Crea e inicia un DatasetRecorder, adjuntándolo a bot._datasetRecorder.
 * @param {Object} [opts]
 * @param {number} [opts.intervalMs] - Intervalo de captura en ms (default: 800)
 */
export async function setupRecorder(bot, mcData, viewerPort = 3000, opts = {}) {
    const recorder = new DatasetRecorder(bot, mcData, opts)
    bot._datasetRecorder = recorder

    try {
        await recorder.start(viewerPort)
    } catch (e) {
        console.warn(`[DatasetRecorder] No se pudo iniciar (¿viewer en puerto ${viewerPort}?): ${e.message}`)
        console.warn(`[DatasetRecorder] Grabación desactivada para esta sesión`)
        bot._datasetRecorder = null
        return recorder
    }

    bot.once('end', () => {
        if (bot._datasetRecorder) {
            bot._datasetRecorder.stop().catch(() => {})
            bot._datasetRecorder = null
        }
    })

    return recorder
}
