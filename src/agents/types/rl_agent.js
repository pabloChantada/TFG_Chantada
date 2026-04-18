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
import { BaseAgent } from './base_agent.js'
import { logInfo, logError } from '../logging.js'
import minecraftData from 'minecraft-data'
import { findNearestVisibleBlock } from '../../htn/primitives/blocks.js'
import { validateSetup, startServer, stopServer } from '../../../scripts/paper_server.js'

// ── Configuración ─────────────────────────────────────────────────────────────
const BRIDGE_PORT      = 8766
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
    constructor(agentName, bridgePort = BRIDGE_PORT, { serverDir = null, useSeed = false } = {}) {
        super(agentName, 'rl')
        this._bridgePort      = bridgePort
        this._httpServer      = null   // servidor HTTP del bridge
        this._mcData          = null
        this._blocksBroken    = []
        this._isAttackingTree = false
        this._attackedBlock   = null
        this._isDead          = false
        this._spawnPos        = null

        // Gestión del servidor Minecraft (opcional)
        this._serverDir  = serverDir ? path.resolve(serverDir) : null
        this._useSeed    = useSeed
        this._mcServer   = null   // proceso del servidor Paper
        this._settings   = null   // guardado para reconexión
    }

    // ── Inicio ────────────────────────────────────────────────────────────────

    async start(settings, viewerPort) {
        this._settings = settings
        try {
            if (this._serverDir) {
                await validateSetup(this._serverDir)
                logInfo(this.name, `Arrancando servidor Paper en: ${this._serverDir}`)
                const { process: mcProc, seed } = await startServer(
                    this._serverDir,
                    settings.port,
                    { useSeed: this._useSeed }
                )
                this._mcServer = mcProc
                logInfo(this.name, `Servidor listo (seed=${seed})`)
            }

            await this._connectAndSetup()

            if (viewerPort) {
                await this.setupViewer(viewerPort)
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
        const { action } = payload

        this._blocksBroken    = []
        this._isAttackingTree = false
        this._attackedBlock   = null
        this._isDead          = false

        const { attackedTree, attackedBlock, brokenBlocks } = await executeRLAction(this.bot, action)
        this._isAttackingTree = attackedTree
        this._attackedBlock   = attackedBlock
        this._blocksBroken.push(...brokenBlocks)  // añadir los rotos por posición (más fiable que diggingCompleted)

        await sleep(STEP_WAIT_MS[action] ?? 150)

        return {
            state:  this._getState(),
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

        const { x, y, z } = this._spawnPos
        this.bot.chat(`/tp RLBot ${x} ${y} ${z}`)
        await sleep(500)

        return { state: this._getState() }
    }

    async _handleWorldReset() {
        if (!this._serverDir) {
            logInfo(this.name, 'world_reset ignorado: no hay servidor gestionado')
            return { ok: true, managed: false }
        }

        logInfo(this.name, 'World reset: parando servidor...')
        this.bot.end()

        await stopServer(this._mcServer)
        this._mcServer = null

        logInfo(this.name, 'World reset: arrancando servidor nuevo...')
        const { process: mcProc, seed } = await startServer(
            this._serverDir,
            this._settings.port,
            { useSeed: this._useSeed }
        )
        this._mcServer = mcProc
        logInfo(this.name, `Nuevo mundo listo (seed=${seed}). Reconectando bot...`)

        await this._connectAndSetup()
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
