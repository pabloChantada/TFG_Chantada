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
            console.log(`[${this.agentName}] Export path auto-generated: ${this.exportPath}`)
            return
        }

        // If path ends with / or has no extension, treat as directory
        if (exportPath.endsWith('/') || exportPath.endsWith('\\') || !exportPath.includes('.')) {
            const cleanPath = exportPath.replace(/[\/\\]+$/, '')
            this.exportPath = `${cleanPath}/metrics_${this.agentName}_${Date.now()}.json`
            console.log(`[${this.agentName}] Export path set to directory: ${this.exportPath}`)
        } else {
            this.exportPath = exportPath
            console.log(`[${this.agentName}] Export path set to file: ${this.exportPath}`)
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
        if (!this.exportPath) {
            console.log(`[${this.agentName}] No export path set`)
            return false
        }

        const dir = path.dirname(this.exportPath)
        console.log(`[${this.agentName}] Ensuring directory exists: ${dir}`)
        
        if (!fs.existsSync(dir)) {
            console.log(`[${this.agentName}] Creating directory: ${dir}`)
            fs.mkdirSync(dir, { recursive: true })
            console.log(`[${this.agentName}] Created metrics directory: ${dir}`)
        } else {
            console.log(`[${this.agentName}] Directory already exists: ${dir}`)
        }
        return true
    }

    /**
     * Export metrics to JSON file
     */
    async exportJSON(metrics) {
        if (!this.exportPath) {
            console.log(`[${this.agentName}] Metrics export disabled (no path specified)`)
            return false
        }

        try {
            console.log(`[${this.agentName}] Starting JSON export to: ${this.exportPath}`)
            
            if (!this.ensureDirectory()) {
                console.error(`[${this.agentName}] Failed to ensure directory`)
                return false
            }

            const jsonContent = JSON.stringify(metrics, null, 2)
            console.log(`[${this.agentName}] Writing ${jsonContent.length} bytes`)
            
            fs.writeFileSync(this.exportPath, jsonContent, 'utf8')
            
            // Verify file was created
            if (fs.existsSync(this.exportPath)) {
                const stats = fs.statSync(this.exportPath)
                console.log(`[${this.agentName}] ✓ Metrics exported to ${this.exportPath} (${stats.size} bytes)`)
                return true
            } else {
                console.error(`[${this.agentName}] File not found after write: ${this.exportPath}`)
                return false
            }
        } catch (error) {
            console.error(`[${this.agentName}] Failed to export metrics:`, error)
            console.error(`[${this.agentName}] Export path was: ${this.exportPath}`)
            throw error
        }
    }
}