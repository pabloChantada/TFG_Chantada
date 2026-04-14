/**
 * RLAgent — Bridge HTTP entre el entorno Gymnasium (Python) y Minecraft.
 *
 * El agente se conecta a Minecraft vía Mineflayer y expone un servidor HTTP
 * que el entorno Python consume. Python manda acciones; el agente las ejecuta
 * y devuelve observaciones + eventos para el cálculo de reward.
 *
 * Endpoints:
 *   POST /step  {action: str}
 *               → {state: {...}, events: {blocks_broken, is_attacking_tree, is_dead}}
 *   POST /reset {}
 *               → {state: {...}}
 *
 * Acciones soportadas:
 *   attack | move_forward_sprint | move_forward_jump | camera_right | camera_left | camera_up | camera_down
 *
 * Comunicación:
 *   Python Gymnasium ←HTTP:8766→ RLAgent (Node.js) ←Mineflayer→ Minecraft
 */

import http          from 'http'
import { BaseAgent } from './base_agent.js'
import { logInfo, logError } from '../logging.js'
import minecraftData from 'minecraft-data'
import { findNearestVisibleBlock } from '../../htn/primitives/blocks.js'

// ── Configuración ─────────────────────────────────────────────────────────────
const BRIDGE_PORT     = 8766
const STEP_HOLD_MS    = 50     // ms que se mantiene una acción de movimiento (mínimo seguro: 1 tick Minecraft)
const STEP_WAIT_MS    = 300    // ms de espera post-acción antes de devolver obs (~6 ticks para física + inventario)
const CAMERA_TURN_RAD  = 0.15   // ~8.6° por step de camera_right / camera_left
const CAMERA_PITCH_RAD = 0.10   // ~5.7° por step de camera_up / camera_down

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
    let attackedTree = false

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
            const block = bot.blockAtCursor(4.5)  // Max distancia de alcance para atacar
            if (block && block.type !== 0 && bot.canDigBlock(block)) {
                attackedTree = LOG_TYPES.includes(block.name)
                const bp     = block.position.offset(0.5, 0.5, 0.5)
                await bot.lookAt(bp, false)
                try { await bot.dig(block, 'ignore') } catch (_) {}
            }
            break
        }

        default:
            logError('RLAgent', new Error(`Acción desconocida: "${actionName}"`))
    }

    return { attackedTree }
}

// ── RLAgent ───────────────────────────────────────────────────────────────────

export class RLAgent extends BaseAgent {
    constructor(agentName, bridgePort = BRIDGE_PORT) {
        super(agentName, 'rl')
        this._bridgePort      = bridgePort
        this._server          = null
        this._mcData          = null
        this._blocksBroken    = []
        this._isAttackingTree = false
        this._isDead          = false
        this._spawnPos        = null   // posición capturada al conectar
    }

    // ── Inicio ────────────────────────────────────────────────────────────────

    async start(settings, _viewerPort) {
        try {
            await this.connectBot(settings)
            this._mcData = minecraftData(this.bot.version)

            // Guardar posición inicial como spawnpoint de entrenamiento
            const p = this.bot.entity.position
            this._spawnPos = { x: Math.round(p.x), y: Math.round(p.y), z: Math.round(p.z) }
            logInfo(this.name, `Spawnpoint registrado: ${JSON.stringify(this._spawnPos)}`)

            // Fijar hora y aplicar haste para romper bloques en un tick
            this.bot.chat('/time set day')
            this.bot.chat(`/effect give ${this.name} minecraft:haste 9999 255 true`)

            // Escuchar bloques rotos para incluirlos en la respuesta /step
            this.bot.on('diggingCompleted', (block) => {
                this._blocksBroken.push(block.name)
                logInfo(this.name, `Bloque roto: ${block.name}`)
            })

            // Detectar muerte vía evento (bot.health puede ser > 0 tras respawn)
            this.bot.on('death', () => {
                this._isDead = true
                logInfo(this.name, 'Bot murió')
            })

            this._startBridgeServer()
            await this.runLogic()

        } catch (error) {
            logError(this.name, new Error(`Error en start: ${error.message}`))
            await this._shutdown()
            process.exit(1)
        }
    }

    // ── Servidor HTTP ─────────────────────────────────────────────────────────

    _startBridgeServer() {
        this._server = http.createServer(async (req, res) => {
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

        this._server.on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                logError(this.name, new Error(`Puerto ${this._bridgePort} ocupado. Ejecuta: npx kill-port ${this._bridgePort}`))
            } else {
                logError(this.name, err)
            }
            process.exit(1)
        })

        this._server.listen(this._bridgePort, () => {
            logInfo(this.name, `Bridge HTTP listo en :${this._bridgePort}`)
        })
    }

    // ── Handlers ──────────────────────────────────────────────────────────────

    async _handleStep(payload) {
        const { action, camera = [0, 0] } = payload

        // Limpiar eventos del step anterior
        this._blocksBroken    = []
        this._isAttackingTree = false
        this._isDead          = false

        const { attackedTree } = await executeRLAction(this.bot, action, camera)
        this._isAttackingTree  = attackedTree

        // Esperar a que el servidor procese físicas y eventos del step
        await sleep(STEP_WAIT_MS)

        return {
            state:  this._getState(),
            events: {
                blocks_broken:    [...this._blocksBroken],
                is_attacking_tree: this._isAttackingTree,
                is_dead:           this._isDead,
            },
        }
    }

    async _handleReset() {
        this._blocksBroken    = []
        this._isAttackingTree = false

        // Matar al bot → respawnea automáticamente
        this.bot.chat('/kill ' + this.name)

        // Esperar a que el bot reaparezca
        await new Promise((resolve) => {
            const onSpawn = () => { clearTimeout(timeout); resolve() }
            const timeout = setTimeout(resolve, 3000)  // fallback si no llega el evento
            this.bot.once('spawn', onSpawn)
        })

        // Teleportar al spawnpoint inicial y restaurar haste
        const { x, y, z } = this._spawnPos
        this.bot.chat(`/tp ${this.name} ${x} ${y} ${z}`)
        this.bot.chat(`/effect give ${this.name} minecraft:haste 9999 255 true`)
        await sleep(500)

        return { state: this._getState() }
    }

    // ── Observación ───────────────────────────────────────────────────────────

    _getState() {
        const pos    = this.bot.entity.position
        const tree   = this._getTreeInfo()
        const cursor = this.bot.blockAtCursor(4.5)
        return {
            x:                Math.round(pos.x * 1000) / 1000,
            y:                Math.round(pos.y * 1000) / 1000,
            z:                Math.round(pos.z * 1000) / 1000,
            yaw:              Math.round(this.bot.entity.yaw   * 1000) / 1000,
            pitch:            Math.round(this.bot.entity.pitch * 1000) / 1000,
            tree_visible:     tree.tree_visible,
            tree_distance:    tree.tree_distance,
            log_count:        this._getLogCount(),
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
                const block = findNearestVisibleBlock(this.bot, this._mcData, logType, 32, true) // Queremos que use el fov
                if (block) {
                    const dist = this.bot.entity.position.distanceTo(block.position)
                    return { tree_visible: 1, tree_distance: Math.round(dist * 10) / 10 }
                }
            }
            return { tree_visible: 0, tree_distance: 0 }
        } catch (_) {
            return { tree_visible: 0, tree_distance: 0 }
        }
    }

    // ── Loop principal (obligatorio por BaseAgent; el RLAgent es reactivo) ────

    async runLogic() {
        logInfo(this.name, 'RLAgent listo. Esperando peticiones del entorno Python...')
        // El agente no tiene loop propio: reacciona a peticiones HTTP del gym.
        await new Promise(() => {})  // mantener proceso vivo
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    async _shutdown() {
        if (this._server) {
            await new Promise(r => this._server.close(r))
            this._server = null
        }
        await super.shutdown()
    }

    async shutdown() {
        logInfo(this.name, 'Cerrando RLAgent...')
        await this._shutdown()
    }
}
