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
        this.screenshotQueue = Promise.resolve(null)
    }

    /**
     * Enable screenshot capture
     */
    enable(viewerPort) {
        this.enabled = true
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
     * Check if screenshots are enabled
     */
    isEnabled() {
        return this.enabled && this.viewerPort
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
        // Wait for the viewer to be ready before capturing
        await this.page.goto(`http://localhost:${this.viewerPort}`, { 
            waitUntil: 'networkidle0' 
        })
        await this.page.waitForSelector('canvas')
    }

    /**
     * Capture a single screenshot
     */
    async captureScreenshot() {
        if (!this.isEnabled()) return null

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
        if (!this.isEnabled()) return null
        this.screenshotQueue = this.screenshotQueue.then(() => this.captureScreenshot())
        return this.screenshotQueue
    }

    /**
     * Close browser and cleanup
     */
    async cleanup() {
        if (this.browser) {
            await this.browser.close()
            this.browser = null
            this.page = null
        }
    }
}