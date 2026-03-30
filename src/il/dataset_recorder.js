/**
 * DatasetRecorder — Intent-aware dataset collection for the HTN chopping pipeline.
 *
 * Each captured frame is tagged with the HTN sub-task that caused it, eliminating
 * the label noise produced by continuous capture (e.g. "turning to search" vs
 * "turning to align for chopping").
 *
 * Intent labels:
 *   search_tree   — bot is scanning/rotating because no tree is visible yet
 *   approach_tree — bot is walking toward a visible tree (moveToBlock active)
 *   chop_tree     — bot is actively breaking a log (collectBlock.collect active)
 *   collect_wood  — bot is collecting broken log items after the block dropped
 *   explore       — bot is doing exploreRandom() because no tree was found
 *
 * Capture rates:
 *   Normal (search/approach/collect): 1 frame / 800 ms
 *   chop_tree:                        1 frame / 1200 ms  (animation is repetitive)
 *   explore:                          1 frame / 2000 ms  (least informative)
 *
 * Phase-transition handling:
 *   - First 300 ms after an intent change are skipped (visual state unsettled)
 *   - After that settle window, an additional 500 ms post-transition pause is
 *     enforced before the first new capture, so the bot posture has stabilised.
 *
 * Integration:
 *   // In main_htn.js startChopTrees():
 *   import { setupRecorder } from '../il/dataset_recorder.js'
 *   await setupRecorder(bot, mcData)
 *
 *   // In HTN primitives (wood.js, mining.js):
 *   bot._datasetRecorder?.setIntent('approach_tree')
 *   bot._datasetRecorder?.setWoodType(woodType)
 */

import fs   from 'fs'
import path from 'path'
import puppeteer from 'puppeteer'
import { findNearestVisibleBlock } from '../htn/primitives/blocks.js'

// ── Timing ────────────────────────────────────────────────────────────────────
const TICK_MS             = 200   // Timer resolution (how often we check)
const NORMAL_INTERVAL_MS  = 800   // Default capture interval
const CHOP_INTERVAL_MS    = 1200  // chop_tree: animation is repetitive
const EXPLORE_INTERVAL_MS = 2000  // explore: least informative phase
const PHASE_SETTLE_MS     = 300   // Skip frames this long after intent change
const PHASE_PAUSE_MS      = 500   // Additional wait before first capture post-transition

// ── Screenshot / viewer ───────────────────────────────────────────────────────
const VIEWER_WIDTH      = 854
const VIEWER_HEIGHT     = 480
const BROWSER_WARMUP_MS = 2000

// ── Chop validation ───────────────────────────────────────────────────────────
// A frame is only labeled chop_tree if the log is visible within this distance
const CHOP_MAX_DIST = 4

// ── Output directory ──────────────────────────────────────────────────────────
const RECORDINGS_DIR = 'data/recordings'

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Scan RECORDINGS_DIR for existing session files and return the next sequential N.
 * Falls back to 1 if the directory doesn't exist or is empty.
 */
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

function getCaptureInterval(intent) {
    if (intent === 'chop_tree') return CHOP_INTERVAL_MS
    if (intent === 'explore')   return EXPLORE_INTERVAL_MS
    return NORMAL_INTERVAL_MS
}

// ── DatasetRecorder ───────────────────────────────────────────────────────────

class DatasetRecorder {
    /**
     * @param {import('mineflayer').Bot} bot
     * @param {Object} mcData  — result of minecraftData(bot.version)
     */
    constructor(bot, mcData) {
        this.bot  = bot
        this.mcData = mcData

        // Intent state
        this.currentIntent     = 'search_tree'
        this._intentChangeTime = Date.now()

        // Capture timing
        this._lastCaptureTime = 0

        // Session
        this._sessionId    = generateSessionId()
        this._frameCount   = 0

        // Runtime state
        this._running   = false
        this._timer     = null
        this._capturing = false

        // Puppeteer
        this._browser = null
        this._page    = null

        // Output
        this._screenshotsDir = null
        this._jsonlPath      = null
        this._jsonlStream    = null

        // Wood type cache (set by chop() after obtainWoodType resolves)
        this._woodType = null
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Set the current HTN sub-task intent.
     * Must be called explicitly by HTN code — never inferred from heuristics.
     * @param {'search_tree'|'approach_tree'|'chop_tree'|'collect_wood'|'explore'} intent
     */
    setIntent(intent) {
        if (intent === this.currentIntent) return
        console.log(`[DatasetRecorder] ${this.currentIntent} → ${intent}`)
        this.currentIntent     = intent
        this._intentChangeTime = Date.now()
    }

    /**
     * Cache the detected wood type so that tree_visible / tree_distance
     * fields use findNearestVisibleBlock with the correct block name.
     * @param {string} woodType  e.g. 'oak_log'
     */
    setWoodType(woodType) {
        this._woodType = woodType
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    async start(viewerPort = 3000) {
        this._screenshotsDir = path.join(RECORDINGS_DIR, `${this._sessionId}_screenshots`)
        this._jsonlPath      = path.join(RECORDINGS_DIR, `${this._sessionId}.jsonl`)

        fs.mkdirSync(this._screenshotsDir, { recursive: true })
        this._jsonlStream = fs.createWriteStream(this._jsonlPath, { flags: 'a' })

        console.log(`[DatasetRecorder] Session  : ${this._sessionId}`)
        console.log(`[DatasetRecorder] JSONL    : ${this._jsonlPath}`)
        console.log(`[DatasetRecorder] Viewer   : http://localhost:${viewerPort}`)

        // Launch Puppeteer and connect to the prismarine-viewer
        this._browser = await puppeteer.launch({ headless: 'new' })
        this._page    = await this._browser.newPage()
        await this._page.setViewport({ width: VIEWER_WIDTH, height: VIEWER_HEIGHT })
        await this._page.goto(`http://localhost:${viewerPort}`, {
            waitUntil: 'domcontentloaded',
            timeout:   15000
        })
        await this._page.waitForSelector('canvas', { timeout: 10000 })
        // Let the renderer draw the first chunks before we start capturing
        await new Promise(r => setTimeout(r, BROWSER_WARMUP_MS))

        this._running          = true
        this._intentChangeTime = Date.now()
        this._scheduleTick()

        console.log(`[DatasetRecorder] Started — capturing at ${NORMAL_INTERVAL_MS}ms intervals`)
    }

    async stop() {
        this._running = false
        if (this._timer) { clearTimeout(this._timer); this._timer = null }

        // Let any in-progress capture finish
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
            `[DatasetRecorder] Stopped — ${this._frameCount} frames saved` +
            ` — session ${this._sessionId}`
        )
    }

    // ── Internal timer loop ───────────────────────────────────────────────────

    _scheduleTick() {
        if (!this._running) return
        this._timer = setTimeout(() => this._tick(), TICK_MS)
    }

    async _tick() {
        if (!this._running || this._capturing) {
            this._scheduleTick()
            return
        }

        const now             = Date.now()
        const timeSinceChange = now - this._intentChangeTime

        // Skip frames during the phase-transition ambiguity window
        if (timeSinceChange < PHASE_SETTLE_MS) {
            this._scheduleTick()
            return
        }

        // The next eligible capture time is the later of:
        //   (a) last capture + capture interval for this intent
        //   (b) intent change time + post-transition pause
        const nextCapture = Math.max(
            this._lastCaptureTime + getCaptureInterval(this.currentIntent),
            this._intentChangeTime + PHASE_PAUSE_MS
        )
        if (now < nextCapture) {
            this._scheduleTick()
            return
        }

        this._capturing = true
        try {
            await this._capture()
        } catch (e) {
            console.warn(`[DatasetRecorder] Capture error: ${e.message}`)
        }
        this._capturing       = false
        this._lastCaptureTime = Date.now()

        this._scheduleTick()
    }

    // ── Frame capture ─────────────────────────────────────────────────────────

    /**
     * Resolve the intent to use for labeling — chop_tree is only valid when the
     * log block is visible within CHOP_MAX_DIST, otherwise fall back to approach_tree.
     */
    _resolveIntent() {
        if (this.currentIntent !== 'chop_tree' || !this._woodType) {
            return this.currentIntent
        }
        try {
            const block = findNearestVisibleBlock(
                this.bot, this.mcData, this._woodType, CHOP_MAX_DIST
            )
            if (!block) return 'approach_tree'
        } catch (_) {}
        return 'chop_tree'
    }

    /**
     * Query findNearestVisibleBlock to get tree_visible / tree_distance metadata.
     * Uses the cached wood type; returns false/null if wood type is unknown yet.
     */
    _getTreeInfo() {
        if (!this._woodType) return { tree_visible: false, tree_distance: null }
        try {
            const block = findNearestVisibleBlock(
                this.bot, this.mcData, this._woodType, 32
            )
            if (!block) return { tree_visible: false, tree_distance: null }
            const dist = Math.round(
                this.bot.entity.position.distanceTo(block.position) * 10
            ) / 10
            return { tree_visible: true, tree_distance: dist }
        } catch (_) {
            return { tree_visible: false, tree_distance: null }
        }
    }

    async _capture() {
        if (!this._page) return

        const canvas = await this._page.$('canvas')
        if (!canvas) return

        const intent                     = this._resolveIntent()
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
                x:     Math.round(pos.x * 1000) / 1000,
                y:     Math.round(pos.y * 1000) / 1000,
                z:     Math.round(pos.z * 1000) / 1000,
                yaw:   Math.round(this.bot.entity.yaw   * 10000) / 10000,
                pitch: Math.round(this.bot.entity.pitch * 10000) / 10000
            },
            action:       intent,
            session_id:   this._sessionId,
            tree_visible,
            tree_distance
        }

        this._jsonlStream.write(JSON.stringify(entry) + '\n')
        this._frameCount++
    }
}

// ── Public factory ────────────────────────────────────────────────────────────

/**
 * Create and start a DatasetRecorder, attaching it to `bot._datasetRecorder`.
 * HTN primitives can then call `bot._datasetRecorder?.setIntent(...)` with no
 * risk of crashing if recording is disabled.
 *
 * @param {import('mineflayer').Bot} bot
 * @param {Object} mcData        — result of minecraftData(bot.version)
 * @param {number} viewerPort    — port of the running prismarine-viewer (default 3000)
 * @returns {Promise<DatasetRecorder>}
 */
export async function setupRecorder(bot, mcData, viewerPort = 3000) {
    const recorder = new DatasetRecorder(bot, mcData)
    bot._datasetRecorder = recorder

    try {
        await recorder.start(viewerPort)
    } catch (e) {
        console.warn(`[DatasetRecorder] Could not start (viewer on port ${viewerPort}?): ${e.message}`)
        console.warn(`[DatasetRecorder] Recording disabled for this session`)
        bot._datasetRecorder = null
        return recorder
    }

    // Automatically stop when the bot disconnects
    bot.once('end', () => {
        if (bot._datasetRecorder) {
            bot._datasetRecorder.stop().catch(() => {})
            bot._datasetRecorder = null
        }
    })

    return recorder
}
