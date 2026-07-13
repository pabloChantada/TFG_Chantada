/**
 * RLAgent — Bridge HTTP entre el entorno Gymnasium (Python) y Minecraft.
 *
 * Endpoints:
 *   POST /step        {action: str}  → {state, events}
 *   POST /reset       {}             → {state}
 *   POST /world_reset {}             → {ok: true}   (reinicia servidor Minecraft)
 *
 * Modo servidor gestionado (--server-dir):
 *   El agente arranca y gestiona el ciclo de vida del servidor Paper.
 *   Cada /world_reset para el servidor, borra el mundo y arranca uno nuevo
 *   con la misma seed (si --use-seed) o una aleatoria.
 *
 * Comunicación:
 *   Python Gymnasium ←HTTP:8766→ RLAgent (Node.js) ←Mineflayer→ Minecraft
 */

import http          from 'http'
import path          from 'path'
import fs            from 'fs'
import puppeteer     from 'puppeteer'
import { BaseAgent } from './base_agent.js'
import { logInfo, logError } from '../logging.js'
import minecraftData from 'minecraft-data'
import { findNearestVisibleBlock } from '../../htn/primitives/blocks.js'
import { validateSetup, startServer, stopServer } from '../../../scripts/paper_server.js'

// ── Configuración ─────────────────────────────────────────────────────────────
const BRIDGE_PORT      = 8766
const VIEWER_WIDTH      = 854   // dimensiones del render del viewer (env lo reescala a IMG_SIZE)
const VIEWER_HEIGHT     = 480
const BROWSER_WARMUP_MS = 2000  // espera inicial a que el viewer renderice
const SHOT_FAIL_LIMIT   = 3     // capturas fallidas seguidas antes de recrear Puppeteer
const STEP_HOLD_MS     = 50     // ms que se mantiene una acción de movimiento (1 tick MC)
const CAMERA_TURN_RAD  = 0.15   // ~8.6° por step de camera_right / camera_left
const CAMERA_PITCH_RAD = 0.10   // ~5.7° por step de camera_up / camera_down

// Espera post-acción por tipo: las acciones de cámara no necesitan que la física se estabilice
const STEP_WAIT_MS = {
    attack:               100,   // bot.dig() ya es await; solo margen para inventario
    move_forward_sprint:  200,   // física + colisión
    move_forward_jump:    250,   // física + arco de salto
    camera_right:          50,   // solo cambio de cursor, sin física
    camera_left:           50,
    camera_up:             50,
    camera_down:           50,
}

// Hybrid SAC (espacio MineRL): pausa única tras aplicar flags concurrentes + cámara continua.
// 200 ms cubre el caso peor (forward+jump) sin penalizar steps de solo cámara.
const HYBRID_STEP_WAIT_MS = 200
const HYBRID_FLAGS = ['forward', 'jump', 'sprint', 'attack']

const LOG_TYPES = ['oak_log', 'birch_log', 'spruce_log', 'dark_oak_log', 'jungle_log', 'acacia_log']

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

async function releaseAll(bot) {
    for (const c of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
        bot.setControlState(c, false)
    }
}

// ── Ejecución de acciones ─────────────────────────────────────────────────────

async function executeRLAction(bot, actionName) {
    await releaseAll(bot)
    let attackedTree  = false
    let attackedBlock = null
    const brokenBlocks = []

    switch (actionName) {

        case 'move_forward_sprint':
            bot.setControlState('forward', true)
            bot.setControlState('sprint',  true)
            await sleep(STEP_HOLD_MS)
            await releaseAll(bot)
            break

        case 'move_forward_jump':
            bot.setControlState('forward', true)
            bot.setControlState('sprint',  true)
            bot.setControlState('jump',    true)
            await sleep(STEP_HOLD_MS)
            await releaseAll(bot)
            break

        case 'camera_right':
            await bot.look(bot.entity.yaw - CAMERA_TURN_RAD, bot.entity.pitch, false)
            break

        case 'camera_left':
            await bot.look(bot.entity.yaw + CAMERA_TURN_RAD, bot.entity.pitch, false)
            break

        case 'camera_up':
            await bot.look(
                bot.entity.yaw,
                Math.max(-Math.PI/2, bot.entity.pitch - CAMERA_PITCH_RAD),
                false
            )
            break

        case 'camera_down':
            await bot.look(
                bot.entity.yaw,
                Math.min(Math.PI/2, bot.entity.pitch + CAMERA_PITCH_RAD),
                false
            )
            break

        case 'attack': {
            const block = bot.blockAtCursor(4.5)
            if (block && block.type !== 0 && bot.canDigBlock(block)) {
                attackedTree  = LOG_TYPES.includes(block.name)
                attackedBlock = block.name
                const blockPos = block.position
                const bp = blockPos.offset(0.5, 0.5, 0.5)
                await bot.lookAt(bp, false)
                try {
                    const DIG_TIMEOUT_MS = 8000
                    await Promise.race([
                        bot.dig(block, true),
                        new Promise((_, reject) =>
                            setTimeout(() => reject(new Error('dig timeout')), DIG_TIMEOUT_MS)
                        ),
                    ])
                    // Verificar si el bloque desapareció — más fiable que diggingCompleted
                    const blockAfter = bot.blockAt(blockPos)
                    if (!blockAfter || blockAfter.type === 0) {
                        brokenBlocks.push(block.name)
                    }
                } catch (err) {
                    logInfo('RLAgent', `dig abortado: ${err.message}`)
                    try { bot.stopDigging() } catch (_) {}
                }
            }
            break
        }

        default:
            logError('RLAgent', new Error(`Acción desconocida: "${actionName}"`))
    }

    return { attackedTree, attackedBlock, brokenBlocks }
}

// ── Hybrid SAC: flags binarios concurrentes + cámara continua ─────────────────

async function executeHybridAction(bot, payload) {
    const flags  = payload.flags  || {}
    const camera = payload.camera || {}
    const dyaw   = Number(camera.dyaw   || 0)
    const dpitch = Number(camera.dpitch || 0)

    let attackedTree  = false
    let attackedBlock = null
    const brokenBlocks = []

    // 1. Cámara continua (un solo bot.look acumulando ambos deltas)
    if (Math.abs(dyaw) > 1e-4 || Math.abs(dpitch) > 1e-4) {
        const newYaw   = bot.entity.yaw   + dyaw            // mineflayer: izq=+yaw, igual que camera_left discreta
        const newPitch = Math.max(-Math.PI/2, Math.min(Math.PI/2, bot.entity.pitch + dpitch))
        await bot.look(newYaw, newPitch, false)
    }

    // 2. Flags concurrentes de movimiento
    bot.setControlState('forward', !!flags.forward)
    bot.setControlState('jump',    !!flags.jump)
    bot.setControlState('sprint',  !!flags.sprint)
    // back/left/right/sneak no soportados en HYBRID_FLAGS (descarta el subset
    // <2.5% del dataset según EDA). Si llegan en payload, se ignoran.

    // 3. Attack (si flag activo) — solo cuenta si el cursor apunta a un log.
    //    El BC del dataset MineRL tiene attack=1 en ~67% de steps; sin este
    //    filtro, cada attack sobre tierra/grass/hojas dispara bot.dig() con
    //    timeout de 8s, multiplicando el tiempo de episodio. Mantiene la
    //    semántica "attack útil" del dataset (en MineRL solo el reward por log
    //    es lo que importa) sin penalizar la política a nivel de gradient
    //    (el agente aprende online que attack vale solo apuntando a un log).
    if (flags.attack) {
        const block = bot.blockAtCursor(4.5)
        if (block && LOG_TYPES.includes(block.name) && bot.canDigBlock(block)) {
            attackedTree  = true
            attackedBlock = block.name
            const blockPos = block.position
            try {
                const DIG_TIMEOUT_MS = 8000
                await Promise.race([
                    bot.dig(block, true),
                    new Promise((_, reject) =>
                        setTimeout(() => reject(new Error('dig timeout')), DIG_TIMEOUT_MS)
                    ),
                ])
                const blockAfter = bot.blockAt(blockPos)
                if (!blockAfter || blockAfter.type === 0) {
                    brokenBlocks.push(block.name)
                }
            } catch (err) {
                logInfo('RLAgent', `dig abortado: ${err.message}`)
                try { bot.stopDigging() } catch (_) {}
            }
        }
    } else {
        // Mantener el hold de movimiento durante el step, sin atacar.
        await sleep(STEP_HOLD_MS)
    }

    // 4. Soltar todos los flags al final del step (modelo "step = 1 tick controlado")
    await releaseAll(bot)

    return { attackedTree, attackedBlock, brokenBlocks }
}

// ── RLAgent ───────────────────────────────────────────────────────────────────

export class RLAgent extends BaseAgent {
    /**
     * @param {string} agentName
     * @param {number} bridgePort
     * @param {object} opts
     * @param {string|null} opts.serverDir  - Ruta al directorio del servidor Paper.
     *                                        Si se indica, el agente gestiona el servidor.
     * @param {boolean}     opts.useSeed    - Usar seed fija de server/seed.txt
     */
    constructor(agentName, bridgePort = BRIDGE_PORT, { serverDir = null, useSeed = false, singleBiome = true } = {}) {
        super(agentName, 'rl')
        this._bridgePort      = bridgePort
        this._httpServer      = null   // servidor HTTP del bridge
        this._mcData          = null
        this._blocksBroken    = []
        this._isAttackingTree = false
        this._attackedBlock   = null
        this._isDead          = false
        this._spawnPos        = null

        // Captura visual (Puppeteer sobre prismarine-viewer), para que el env
        // reciba la imagen en primera persona en cada /step y /reset.
        this._browser = null
        this._page    = null
        this._shotFailStreak = 0      // capturas fallidas seguidas → dispara recuperación
        this._recovering     = false  // evita recreaciones concurrentes de Puppeteer

        // Gestión del servidor Minecraft (opcional)
        this._serverDir  = serverDir ? path.resolve(serverDir) : null
        this._useSeed    = useSeed
        this._singleBiome = singleBiome   // false = mundo normal (--no-datapack), igual que HTN/IL
        this._mcServer   = null   // proceso del servidor Paper
        this._settings   = null   // guardado para reconexión
        this._viewerPort = null   // guardado para re-inicializar tras world reset
    }

    // ── Inicio ────────────────────────────────────────────────────────────────

    async start(settings, viewerPort) {
        this._settings   = settings
        this._viewerPort = viewerPort || null
        try {
            if (this._serverDir) {
                await validateSetup(this._serverDir)
                logInfo(this.name, `Arrancando servidor Paper en: ${this._serverDir}`)
                const { process: mcProc, seed } = await startServer(
                    this._serverDir,
                    settings.port,
                    { useSeed: this._useSeed, singleBiome: this._singleBiome }
                )
                this._mcServer = mcProc
                logInfo(this.name, `Servidor listo (seed=${seed})`)
            }

            await this._connectAndSetup()

            if (viewerPort) {
                await this.setupViewer(viewerPort)
                await this._startPuppeteer(viewerPort)
            } else {
                logInfo(this.name, 'Sin --viewer-port: el env recibirá imágenes en negro (modo state-only)')
            }

            this._startBridgeServer()
            await this.runLogic()

        } catch (error) {
            logError(this.name, new Error(`Error en start: ${error.message}`))
            await this._shutdown()
            process.exit(1)
        }
    }

    // ── Conexión y setup del bot ──────────────────────────────────────────────

    async _connectAndSetup() {
        await this.connectBot(this._settings)
        this._mcData = minecraftData(this.bot.version)

        const p = this.bot.entity.position
        this._spawnPos = { x: Math.round(p.x), y: Math.round(p.y), z: Math.round(p.z) }
        logInfo(this.name, `Spawnpoint registrado: ${JSON.stringify(this._spawnPos)}`)

        this.bot.chat('/time set day')

        this.bot.on('death', () => {
            this._isDead = true
            logInfo(this.name, 'Bot murió')
        })
    }

    // ── Captura visual (Puppeteer) ────────────────────────────────────────────

    /**
     * Abre un navegador headless sobre el prismarine-viewer para capturar la
     * vista en primera persona del bot. El env Python la recibe en base64 en
     * cada respuesta del bridge (campo `screenshot`).
     */
    async _startPuppeteer(viewerPort) {
        try {
            if (this._browser) { try { await this._browser.close() } catch (_) {} }
            this._browser = await puppeteer.launch({ headless: 'new' })
            this._page    = await this._browser.newPage()
            await this._page.setViewport({ width: VIEWER_WIDTH, height: VIEWER_HEIGHT })
            await this._page.goto(`http://localhost:${viewerPort}`, {
                waitUntil: 'domcontentloaded',
                timeout:   15000,
            })
            await this._page.waitForSelector('canvas', { timeout: 10000 })
            await sleep(BROWSER_WARMUP_MS)
            logInfo(this.name, 'Puppeteer listo (captura visual activa)')
        } catch (e) {
            logError(this.name, new Error(`No se pudo iniciar Puppeteer: ${e.message}. El env recibirá imágenes en negro.`))
            this._page = null
        }
    }

    /**
     * Captura el canvas del viewer y lo devuelve como string base64 (PNG).
     * Devuelve "" si no hay página (modo state-only) o si la captura falla.
     * @returns {Promise<string>}
     */
    async _captureScreenshot() {
        if (!this._page) return ''
        try {
            const canvas = await this._page.$('canvas')
            if (!canvas) return ''
            // Timeout en la captura: si Puppeteer se cuelga (degradación tras horas
            // o tras un world_reset), devolvemos "" para ese step en vez de bloquear
            // el bridge y tirar el entrenamiento por timeout en el lado Python.
            const SHOT_TIMEOUT_MS = 8000
            const b64 = await Promise.race([
                canvas.screenshot({ type: 'png', encoding: 'base64' }),
                new Promise((_, rej) => setTimeout(() => rej(new Error('screenshot timeout')), SHOT_TIMEOUT_MS)),
            ])
            this._shotFailStreak = 0   // captura OK → resetear racha de fallos
            // Verificación del visor: con RL_DEBUG_SHOT=1 vuelca el último frame a
            // disco para confirmar visualmente que NO es negro (que el visor capta).
            if (process.env.RL_DEBUG_SHOT && b64) {
                try {
                    fs.writeFileSync('data/_rl_last_frame.png', Buffer.from(b64, 'base64'))
                } catch (_) { /* no romper el step por el debug */ }
            }
            return b64
        } catch (e) {
            logError(this.name, new Error(`Screenshot fallido: ${e.message}`))
            // Recuperación: si Puppeteer falla repetidamente (página colgada tras
            // horas o tras un world_reset), recrear la página para no quedarse ciego
            // en negro de forma permanente. No-await: no bloquea el step.
            this._shotFailStreak += 1
            if (this._shotFailStreak >= SHOT_FAIL_LIMIT && !this._recovering && this._viewerPort) {
                this._recoverPuppeteer()
            }
            return ''
        }
    }

    /**
     * Recrea la página de Puppeteer cuando la captura lleva varios fallos seguidos.
     * Se ejecuta en segundo plano (sin await desde el step) para no bloquear.
     */
    async _recoverPuppeteer() {
        if (this._recovering) return
        this._recovering = true
        logError(this.name, new Error(`Puppeteer: ${this._shotFailStreak} capturas fallidas seguidas, recreando página...`))
        try {
            await this._startPuppeteer(this._viewerPort)
            this._shotFailStreak = 0
            logInfo(this.name, 'Puppeteer recuperado tras fallos de captura')
        } catch (e) {
            logError(this.name, new Error(`Recuperación de Puppeteer falló: ${e.message}`))
        } finally {
            this._recovering = false
        }
    }

    // ── Servidor HTTP ─────────────────────────────────────────────────────────

    _startBridgeServer() {
        this._httpServer = http.createServer(async (req, res) => {
            if (req.method !== 'POST') {
                res.writeHead(405); res.end('Method Not Allowed'); return
            }

            let body = ''
            req.on('data', chunk => body += chunk)
            req.on('end', async () => {
                try {
                    const payload = body ? JSON.parse(body) : {}
                    let result

                    if (req.url === '/step') {
                        result = await this._handleStep(payload)
                    } else if (req.url === '/reset') {
                        result = await this._handleReset()
                    } else if (req.url === '/world_reset') {
                        result = await this._handleWorldReset()
                    } else {
                        res.writeHead(404); res.end('Not Found'); return
                    }

                    res.writeHead(200, { 'Content-Type': 'application/json' })
                    res.end(JSON.stringify(result))

                } catch (err) {
                    logError(this.name, new Error(`Bridge error [${req.url}]: ${err.message}`))
                    res.writeHead(500)
                    res.end(JSON.stringify({ error: err.message }))
                }
            })
        })

        this._httpServer.on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                logError(this.name, new Error(`Puerto ${this._bridgePort} ocupado. Ejecuta: npx kill-port ${this._bridgePort}`))
            } else {
                logError(this.name, err)
            }
            process.exit(1)
        })

        this._httpServer.listen(this._bridgePort, () => {
            logInfo(this.name, `Bridge HTTP listo en :${this._bridgePort}`)
        })
    }

    // ── Handlers ──────────────────────────────────────────────────────────────

    async _handleStep(payload) {
        this._blocksBroken    = []
        this._isAttackingTree = false
        this._attackedBlock   = null
        this._isDead          = false

        let result, waitMs
        if (payload && payload.flags !== undefined) {
            // Modo Hybrid SAC: payload = {flags: {...}, camera: {dyaw, dpitch}}
            result = await executeHybridAction(this.bot, payload)
            waitMs = HYBRID_STEP_WAIT_MS
        } else {
            // Modo discreto legacy (DQN / PPO / SAC discreto): payload = {action: name}
            const { action } = payload
            result = await executeRLAction(this.bot, action)
            waitMs = STEP_WAIT_MS[action] ?? 150
        }

        const { attackedTree, attackedBlock, brokenBlocks } = result
        this._isAttackingTree = attackedTree
        this._attackedBlock   = attackedBlock
        this._blocksBroken.push(...brokenBlocks)

        await sleep(waitMs)

        return {
            state:      this._getState(),
            screenshot: await this._captureScreenshot(),
            events: {
                blocks_broken:     [...this._blocksBroken],
                is_attacking_tree: this._isAttackingTree,
                attacked_block:    this._attackedBlock,
                is_dead:           this._isDead,
            },
        }
    }

    async _handleReset() {
        this._blocksBroken    = []
        this._isAttackingTree = false

        this.bot.chat('/kill RLBot')

        await new Promise((resolve) => {
            const onSpawn = () => { clearTimeout(timeout); resolve() }
            const timeout = setTimeout(resolve, 3000)
            this.bot.once('spawn', onSpawn)
        })

        // En evaluación (FIXED_SPAWN definido) reproducimos EXACTAMENTE el spawn
        // inicial — posición Y orientación (yaw/pitch)— para que todos los episodios
        // (y todas las técnicas) arranquen idénticos. En entrenamiento, sin
        // FIXED_SPAWN, volvemos a la posición registrada del spawn scouteado.
        if (process.env.FIXED_SPAWN) {
            await this._applyFixedSpawn()
        } else {
            const { x, y, z } = this._spawnPos
            this.bot.chat(`/tp RLBot ${x} ${y} ${z}`)
        }
        await sleep(500)

        return { state: this._getState(), screenshot: await this._captureScreenshot() }
    }

    async _handleWorldReset() {
        if (!this._serverDir) {
            logInfo(this.name, 'world_reset ignorado: no hay servidor gestionado')
            return { ok: true, managed: false }
        }

        logInfo(this.name, 'World reset: parando servidor...')
        if (this._viewerPort) {
            try { this.bot.viewer?.close() } catch (_) {}
        }
        this.bot.end()

        await stopServer(this._mcServer)
        this._mcServer = null

        logInfo(this.name, 'World reset: arrancando servidor nuevo...')
        const { process: mcProc, seed } = await startServer(
            this._serverDir,
            this._settings.port,
            { useSeed: this._useSeed, singleBiome: this._singleBiome }
        )
        this._mcServer = mcProc
        logInfo(this.name, `Nuevo mundo listo (seed=${seed}). Reconectando bot...`)

        await this._connectAndSetup()
        if (this._viewerPort) {
            await this.setupViewer(this._viewerPort)
            // El viewer anterior quedó muerto al reconectar el bot: recrear la
            // página de Puppeteer contra el nuevo render.
            await this._startPuppeteer(this._viewerPort)
        }
        logInfo(this.name, 'World reset completado')
        return { ok: true, managed: true, seed: String(seed) }
    }

    // ── Observación ───────────────────────────────────────────────────────────

    _getState() {
        const pos    = this.bot.entity.position
        const tree   = this._getTreeInfo()
        const cursor = this.bot.blockAtCursor(4.5)
        return {
            x:                 Math.round(pos.x * 1000) / 1000,
            y:                 Math.round(pos.y * 1000) / 1000,
            z:                 Math.round(pos.z * 1000) / 1000,
            yaw:               Math.round(this.bot.entity.yaw   * 1000) / 1000,
            pitch:             Math.round(this.bot.entity.pitch * 1000) / 1000,
            tree_visible:      tree.tree_visible,
            tree_distance:     tree.tree_distance,
            log_count:         this._getLogCount(),
            is_looking_at_log: (cursor && LOG_TYPES.includes(cursor.name)) ? 1 : 0,
        }
    }

    _getLogCount() {
        let count = 0
        for (const item of this.bot.inventory.items()) {
            if (LOG_TYPES.includes(item.name)) count += item.count
        }
        return count
    }

    _getTreeInfo() {
        try {
            for (const logType of LOG_TYPES) {
                const block = findNearestVisibleBlock(this.bot, this._mcData, logType, 32, true)
                if (block) {
                    const dist = this.bot.entity.position.distanceTo(block.position)
                    return { tree_visible: 1, tree_distance: Math.round(dist * 10) / 10 }
                }
            }
            return { tree_visible: 0, tree_distance: null }
        } catch (_) {
            return { tree_visible: 0, tree_distance: null }
        }
    }

    // ── Loop principal ────────────────────────────────────────────────────────

    async runLogic() {
        logInfo(this.name, 'RLAgent listo. Esperando peticiones del entorno Python...')
        await new Promise(() => {})
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    async _shutdown() {
        if (this._browser) {
            try { await this._browser.close() } catch (_) {}
            this._browser = null
            this._page    = null
        }
        if (this._httpServer) {
            await new Promise(r => this._httpServer.close(r))
            this._httpServer = null
        }
        if (this._mcServer) {
            await stopServer(this._mcServer)
            this._mcServer = null
        }
        await super.shutdown()
    }

    async shutdown() {
        logInfo(this.name, 'Cerrando RLAgent...')
        await this._shutdown()
    }
}
