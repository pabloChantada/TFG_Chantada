/**
 * Thin HTTP server wrapping a Mineflayer bot for Gym-compatible RL.
 * 
 * Exposes:
 *   POST /reset   → resets episode, returns {observation, info}
 *   POST /step    → applies action, returns {observation, reward, terminated, truncated, info}
 *   GET  /state   → returns current observation without stepping
 *   GET  /info    → returns env metadata (action_space, observation_space)
 *   POST /close   → disconnects bot and shuts down
 * 
 * Usage:
 *   node src/rl/server.js [--port 3001] [--mc_port 55916]
 */

import http from 'http'
import { createBot } from 'mineflayer'
import { pathfinder as pathfinderPlugin } from 'mineflayer-pathfinder'
import { applyMultiDiscreteAction, releaseAllControls, ACTION_SPACE } from './actions.js'
import { extractState, STATE_DIM } from './state.js'
import { computeReward, DEFAULT_REWARD_CONFIG } from './reward.js'
import { isLog, getLogCount } from './state.js'
import { argv } from 'process'

// ─── CLI args ────────────────────────────────────────────────────────────────
function getArg(name, fallback) {
    const idx = argv.indexOf(`--${name}`)
    return idx > -1 ? argv[idx + 1] : fallback
}

const SERVER_PORT = parseInt(getArg('port', '3001'))
const MC_HOST = getArg('mc_host', 'localhost')
const MC_PORT = parseInt(getArg('mc_port', '55916'))
const TARGET_LOGS = parseInt(getArg('target_logs', '5'))
const MAX_STEPS = parseInt(getArg('max_steps', '1000'))
const TICKS_PER_STEP = parseInt(getArg('ticks_per_step', '3'))
const BOT_NAME = getArg('bot_name', 'RLBot')

// ─── Episode state ───────────────────────────────────────────────────────────
let bot = null
let stepCount = 0
let totalReward = 0
let initialLogCount = 0
let prevMeta = null
let ready = false

const rewardConfig = { ...DEFAULT_REWARD_CONFIG, targetLogs: TARGET_LOGS }

// ─── Bot creation ────────────────────────────────────────────────────────────
function createMineflayerBot() {
    return new Promise((resolve, reject) => {
        const b = createBot({
            host: MC_HOST,
            port: MC_PORT,
            username: BOT_NAME,
            version: '1.20.4',
        })

        b.loadPlugin(pathfinderPlugin)

        b.once('spawn', async () => {
            console.log(`[Server] Bot "${BOT_NAME}" spawned at ${MC_HOST}:${MC_PORT}`)
            await b.waitForTicks(40) // wait for world to load
            resolve(b)
        })

        b.on('error', (err) => {
            console.error('[Server] Bot error:', err.message)
            reject(err)
        })

        b.on('kicked', (reason) => {
            console.error('[Server] Bot kicked:', reason)
        })
    })
}

// ─── Observation helpers ─────────────────────────────────────────────────────
function getObservation() {
    const { vector, meta } = extractState(bot)
    return { observation: vector, meta }
}

async function dropAllLogs() {
    const logItems = bot.inventory.items().filter(item => isLog(item.name))
    for (const item of logItems) {
        try {
            await bot.tossStack(item)
            await bot.waitForTicks(2)
        } catch (_) {}
    }
}

// ─── HTTP handler ────────────────────────────────────────────────────────────
async function handleRequest(req, res) {
    const url = new URL(req.url, `http://localhost:${SERVER_PORT}`)
    const method = req.method

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Content-Type', 'application/json')

    try {
        // ── GET /info ──
        if (method === 'GET' && url.pathname === '/info') {
            return send(res, 200, {
                action_space: ACTION_SPACE,
                observation_dim: STATE_DIM,
                target_logs: TARGET_LOGS,
                max_steps: MAX_STEPS,
                ticks_per_step: TICKS_PER_STEP,
            })
        }

        // ── GET /state ──
        if (method === 'GET' && url.pathname === '/state') {
            if (!ready) return send(res, 503, { error: 'Bot not ready' })
            const { observation, meta } = getObservation()
            return send(res, 200, { observation, info: meta })
        }

        // ── POST /reset ──
        if (method === 'POST' && url.pathname === '/reset') {
            if (!bot) {
                bot = await createMineflayerBot()
                ready = true
            }

            await releaseAllControls(bot)
            if (bot.targetDigBlock) {
                try { await bot.stopDigging() } catch (_) {}
            }

            await dropAllLogs()

            stepCount = 0
            totalReward = 0
            const { observation, meta } = getObservation()
            initialLogCount = meta.logCount
            prevMeta = meta

            console.log(`[Server] Episode reset. Initial logs: ${initialLogCount}`)
            return send(res, 200, { observation, info: meta })
        }

        // ── POST /step ──
        if (method === 'POST' && url.pathname === '/step') {
            if (!ready) return send(res, 503, { error: 'Bot not ready. Call /reset first.' })

            const body = await readBody(req)
            const action = body.action // array of ints: [fwd, back, lat, vert, yaw, pitch, attack, craft, smelt, place, equip]

            if (!Array.isArray(action) || action.length !== ACTION_SPACE.length) {
                return send(res, 400, {
                    error: `action must be array of ${ACTION_SPACE.length} ints`,
                    expected: ACTION_SPACE.map((s, i) => `dim${i}: 0..${s - 1}`),
                })
            }

            // 1. Apply action
            await applyMultiDiscreteAction(bot, action)

            // 2. Wait for action effect
            await bot.waitForTicks(TICKS_PER_STEP)

            // 3. Observe
            const { observation, meta } = getObservation()

            // 4. Reward
            const { reward, done, info } = computeReward(prevMeta, meta, bot, rewardConfig)

            stepCount++
            totalReward += reward
            prevMeta = meta

            const truncated = stepCount >= MAX_STEPS

            // Release controls on episode end
            if (done || truncated) {
                await releaseAllControls(bot)
                if (bot.targetDigBlock) {
                    try { await bot.stopDigging() } catch (_) {}
                }
            }

            info.step = stepCount
            info.total_reward = round(totalReward, 4)

            return send(res, 200, {
                observation,
                reward: round(reward, 4),
                terminated: done,
                truncated,
                info,
            })
        }

        // ── POST /close ──
        if (method === 'POST' && url.pathname === '/close') {
            console.log('[Server] Shutting down...')
            if (bot) {
                bot.quit()
                bot = null
                ready = false
            }
            send(res, 200, { status: 'closed' })
            setTimeout(() => process.exit(0), 500)
            return
        }

        // 404
        return send(res, 404, { error: `Unknown endpoint: ${method} ${url.pathname}` })

    } catch (err) {
        console.error('[Server] Error:', err)
        return send(res, 500, { error: err.message })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function send(res, status, data) {
    res.writeHead(status)
    res.end(JSON.stringify(data))
}

function readBody(req) {
    return new Promise((resolve, reject) => {
        let data = ''
        req.on('data', chunk => data += chunk)
        req.on('end', () => {
            try { resolve(JSON.parse(data)) }
            catch (e) { reject(new Error('Invalid JSON body')) }
        })
        req.on('error', reject)
    })
}

function round(n, d) {
    return Math.round(n * 10 ** d) / 10 ** d
}

// ─── Start server ────────────────────────────────────────────────────────────
const server = http.createServer(handleRequest)
server.listen(SERVER_PORT, () => {
    console.log('='.repeat(50))
    console.log(`RL Server listening on http://localhost:${SERVER_PORT}`)
    console.log(`Minecraft: ${MC_HOST}:${MC_PORT}`)
    console.log(`Target logs: ${TARGET_LOGS} | Max steps: ${MAX_STEPS}`)
    console.log('='.repeat(50))
    console.log('Endpoints:')
    console.log('  GET  /info   - action/observation space metadata')
    console.log('  POST /reset  - reset episode (creates bot on first call)')
    console.log('  POST /step   - step with action, body: {"action": [0,0,0,...]}')
    console.log('  GET  /state  - get current observation')
    console.log('  POST /close  - shutdown')
    console.log('='.repeat(50))
})
