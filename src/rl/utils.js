/**
 * Shared utilities for the RL wood-chopping system.
 */

import fs from 'fs'
import path from 'path'

/**
 * Save episode transitions to a JSONL file (append mode).
 * Compatible with data/train.jsonl format for retraining.
 * 
 * @param {object[]} transitions - Array of transition objects from environment
 * @param {string} outputPath - Output JSONL file path
 */
export function saveTransitionsJSONL(transitions, outputPath = 'data/rl_episodes.jsonl') {
    const dir = path.dirname(outputPath)
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
    }

    const lines = transitions.map(t => JSON.stringify(t)).join('\n') + '\n'
    fs.appendFileSync(outputPath, lines, 'utf-8')
    console.log(`[RL-Utils] Saved ${transitions.length} transitions to ${outputPath}`)
}

/**
 * Save episode summary to JSON file.
 * @param {object} summary - Episode summary from environment
 * @param {string} outputDir - Output directory
 */
export function saveEpisodeSummary(summary, outputDir = 'data/rl_summaries/') {
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true })
    }

    const filename = `episode_${Date.now()}.json`
    const filepath = path.join(outputDir, filename)
    fs.writeFileSync(filepath, JSON.stringify(summary, null, 2), 'utf-8')
    console.log(`[RL-Utils] Episode summary saved to ${filepath}`)
}

/**
 * Load training config with defaults.
 * @param {object} overrides 
 * @returns {object}
 */
export function loadConfig(overrides = {}) {
    return {
        targetLogs: 5,
        maxSteps: 2000,
        ticksPerStep: 4,
        numEpisodes: 100,
        policy: 'random',           // 'random', 'model', 'epsilon_greedy'
        modelUrl: 'http://localhost:5000/predict',
        epsilon: 0.1,
        metricsEnabled: true,
        screenshotsEnabled: false,
        viewerPort: 3000,
        outputPath: 'data/rl_episodes.jsonl',
        resetMode: 'clear_inventory', // 'clear_inventory' | 'relative' | 'none'
        stuckThreshold: 2,        // Steps without progress before unstuck
        stuckCheckInterval: 50,     // Check every N steps
        minecraftPort: 55916,
        minecraftHost: 'localhost',
        ...overrides
    }
}