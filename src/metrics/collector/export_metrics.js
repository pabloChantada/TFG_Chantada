import fs from 'fs'
import path from 'path'

export class MetricsExporter {
    constructor(agentName) {
        this.agentName = agentName
        this.exportPath = null
    }

    /**
     * Set export path (file or directory)
     */
    setExportPath(exportPath) {
        if (!exportPath) {
            this.exportPath = `src/metrics/agent_metrics/metrics_${this.agentName}_${Date.now()}.json`
            return
        }

        // If path ends with / or has no extension, treat as directory
        if (exportPath.endsWith('/') || !exportPath.includes('.')) {
            this.exportPath = `${exportPath.replace(/\/$/, '')}/metrics_${this.agentName}_${Date.now()}.json`
        } else {
            this.exportPath = exportPath
        }
    }

    /**
     * Get export path
     */
    getExportPath() {
        return this.exportPath
    }

    /**
     * Ensure export directory exists
     */
    ensureDirectory() {
        if (!this.exportPath) return

        const dir = path.dirname(this.exportPath)
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true })
            console.log(`[${this.agentName}] Created metrics directory: ${dir}`)
        }
    }

    /**
     * Export metrics to JSON file
     */
    async exportJSON(metrics) {
        if (!this.exportPath) {
            console.log(`[${this.agentName}] Metrics export disabled (no path specified)`)
            return
        }

        try {
            this.ensureDirectory()
            fs.writeFileSync(
                this.exportPath,
                JSON.stringify(metrics, null, 2)
            )
            console.log(`[${this.agentName}] Metrics exported to ${this.exportPath}`)
        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics: ${error.message}`)
            throw error
        }
    }
}