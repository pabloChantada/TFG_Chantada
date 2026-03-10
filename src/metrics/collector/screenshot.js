/**
 * Screenshot capture and management
 * Handles Puppeteer browser lifecycle and queued captures
 */

import fs from 'fs'
import path from 'path'
import puppeteer from 'puppeteer'

export class ScreenshotManager {
    constructor(agentName) {
        this.agentName = agentName
        this.enabled = false
        this.viewerPort = null
        this.screenshotsDir = null
        this.screenshotFormat = 'png'
        // We could change these in the future if needed
        this.screenshotWidth = 1024
        this.screenshotHeight = 768
        this.browser = null
        this.page = null
        this._closed = false
        this._capturing = false
        this.screenshotQueue = Promise.resolve(null)
        this._everEnabled = false
    }

    /**
     * Enable screenshot capture
     */
    enable(viewerPort) {
        this.enabled = true
        this._everEnabled = true
        this.viewerPort = viewerPort
        this.screenshotsDir = `src/metrics/agent_metrics/${this.agentName}_screenshots/`
        console.log(`[${this.agentName}] Screenshots enabled on port ${viewerPort}`)
    }

    /**
     * Disable screenshot capture
     */
    disable() {
        this.enabled = false
    }

    /**
     * Check if screenshots were ever enabled during this run
     */
    wasEnabled() {
        return this._everEnabled
    }

    /**
     * Stop screenshot capture (disable + keep wasEnabled history)
     */
    stop() {
        this.enabled = false
    }

    /**
     * Flush the screenshot queue, waiting for all pending captures
     */
    async flushQueue(timeoutMs = 5000) {
        return Promise.race([
            this.screenshotQueue,
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Screenshot flush timed out')), timeoutMs)
            )
        ])
    }

    /**
     * Check if screenshots are enabled
     */
    isEnabled() {
        return this.enabled && this.viewerPort
    }

    /**
     * Pre-initialize browser so first screenshot doesn't have cold-start delay.
     * Should be called during setup, before tracking starts.
     * @returns {boolean} true if browser is ready
     */
    async warmup() {
        if (!this.isEnabled()) return false
        try {
            await this.ensureDirectory()
            await this.ensureBrowser()
            const ready = !!(this.browser && this.page)
            if (ready) {
                console.log(`[${this.agentName}] Screenshot browser ready (warmed up)`)
            }
            return ready
        } catch (err) {
            console.warn(`[${this.agentName}] Screenshot warmup failed: ${err.message}`)
            return false
        }
    }

    /**
     * Capture a screenshot immediately (blocking, no queue).
     * Used by the synchronous recording loop.
     * @returns {string|null} file path or null on failure
     */
    async captureNow() {
        if (!this.isEnabled() || this._closed) return null
        if (!this.browser || !this.page) return null

        try {
            const canvas = await this.page.$('canvas')
            if (!canvas) return null

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
            const filename = `screenshot_${timestamp}.${this.screenshotFormat}`
            const filePath = path.join(this.screenshotsDir, filename)

            await canvas.screenshot({
                path: filePath,
                type: this.screenshotFormat
            })

            return filePath
        } catch (error) {
            // If the browser died, clean up properly and stop trying
            if (error.message.includes('Target closed') ||
                error.message.includes('Execution context was destroyed') ||
                error.message.includes('Protocol error') ||
                error.message.includes('Session closed')) {
                try {
                    if (this.browser) await this.browser.close().catch(() => {})
                } catch (_) { /* ignore */ }
                this.browser = null
                this.page = null
                this.enabled = false
                console.warn(`[${this.agentName}] Screenshot browser died, disabling captures`)
            }
            return null
        }
    }

    /**
     * Ensure screenshots directory exists
     */
    async ensureDirectory() {
        if (!this.screenshotsDir) return
        if (!fs.existsSync(this.screenshotsDir)) {
            fs.mkdirSync(this.screenshotsDir, { recursive: true })
        }
    }

    /**
     * Initialize Puppeteer browser
     */
    async ensureBrowser() {
        if (!this.isEnabled()) return
        if (this.browser && this.page) return

        // We don't really need to see the browser, so we start it in headless mode.
        this.browser = await puppeteer.launch({ headless: 'new' })
        this.page = await this.browser.newPage()
        await this.page.setViewport({ 
            width: this.screenshotWidth, 
            height: this.screenshotHeight 
        })
        // domcontentloaded is enough — the 3D viewer streams chunks forever
        // so networkidle0 will always time out.
        await this.page.goto(`http://localhost:${this.viewerPort}`, { 
            waitUntil: 'domcontentloaded',
            timeout: 10000
        })
        await this.page.waitForSelector('canvas', { timeout: 10000 })
        // Give the renderer a moment to draw the first real frame
        await new Promise(r => setTimeout(r, 2000))
    }

    /**
     * Capture a single screenshot
     */
    async captureScreenshot() {
        if (!this.isEnabled() || this._closed) return null

        try {
            // Check we have everything ready
            await this.ensureDirectory()
            await this.ensureBrowser()

            const canvas = await this.page.$('canvas')
            if (!canvas) return null

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
            const filename = `screenshot_${timestamp}.${this.screenshotFormat}`
            const filePath = path.join(this.screenshotsDir, filename)

            await canvas.screenshot({ 
                path: filePath, 
                type: this.screenshotFormat 
            })

            return filePath
        } catch (error) {
            console.warn(`[${this.agentName}] Screenshot failed: ${error.message}`)
            return null
        }
    }

    /**
     * Queue a screenshot capture to avoid overlapping captures
     */
    async queueCapture() {
        if (!this.isEnabled() || this._capturing) return null
        this._capturing = true
        this.screenshotQueue = this.screenshotQueue
            .then(() => this.captureScreenshot())
            .finally(() => { this._capturing = false })
        return this.screenshotQueue
    }

    /**
     * Close browser and cleanup
     */
    async cleanup() {
        this._closed = true
        this.enabled = false
        if (this.browser) {
            try {
                // Kill the browser process to ensure Chromium doesn't linger
                const proc = this.browser.process()
                await this.browser.close().catch(() => {})
                // Force-kill if still alive after close
                if (proc && !proc.killed) {
                    proc.kill('SIGKILL')
                }
            } catch (_) { /* ignore cleanup errors */ }
            this.browser = null
            this.page = null
        }
    }
}