/**
 * Main entry point for the RL wood-chopping agent.
 * 
 * Usage:
 *   node src/rl/run_wood_rl.js [options]
 * 
 * Options (via environment variables):
 *   RL_POLICY=random|model|epsilon_greedy  (default: random)
 *   RL_EPISODES=100                        (default: 100)
 *   RL_TARGET_LOGS=5                       (default: 5)
 *   RL_MAX_STEPS=2000                      (default: 2000)
 *   RL_EPSILON=0.1                         (default: 0.1)
 *   RL_MODEL_URL=http://...               (default: http://localhost:5000/predict)
 *   RL_SCREENSHOTS=true|false              (default: false)
 *   RL_VIEWER_PORT=3000                    (default: 3000)
 *   RL_RESET_MODE=clear_inventory|relative|none (default: clear_inventory)
 *   RL_STUCK_THRESHOLD=100                 (default: 100)
 *   MC_PORT=55916                          (default: 55916)
 * 
 * Requires a running Minecraft server with LAN open.
 */

import { createBot } from 'mineflayer'
import { WoodChopEnvironment } from './environment.js'
import { RLAgent } from './agent.js'
import { loadConfig, saveTransitionsJSONL, saveEpisodeSummary } from './utils.js'
import { MetricsCollector } from '../metrics/metrics_collector.js'
import { argv } from 'process'

// Extract port from command line (node script.js --port 55916)
const portArgIndex = argv.indexOf('--port') || argv.indexOf('-p')
const cliPort = portArgIndex > -1 ? parseInt(argv[portArgIndex + 1]) : null

// Parse config from environment
const config = loadConfig({
    policy: process.env.RL_POLICY || 'random',
    numEpisodes: parseInt(process.env.RL_EPISODES || '100'),
    targetLogs: parseInt(process.env.RL_TARGET_LOGS || '5'),
    maxSteps: parseInt(process.env.RL_MAX_STEPS || '200'),
    epsilon: parseFloat(process.env.RL_EPSILON || '0.1'),
    modelUrl: process.env.RL_MODEL_URL || 'http://localhost:5000/predict',
    screenshotsEnabled: process.env.RL_SCREENSHOTS === 'true',
    viewerPort: parseInt(process.env.RL_VIEWER_PORT || '3000'),
    resetMode: process.env.RL_RESET_MODE || 'clear_inventory',
    stuckThreshold: parseInt(process.env.RL_STUCK_THRESHOLD || '5'),
    minecraftPort: cliPort || parseInt(process.env.MC_PORT || '55916'),
    minecraftHost: process.env.MC_HOST || 'localhost',
})

console.log('='.repeat(60))
console.log('RL WOOD CHOPPING AGENT')
console.log('='.repeat(60))
console.log(`Policy       : ${config.policy}`)
console.log(`Episodes     : ${config.numEpisodes}`)
console.log(`Target logs  : ${config.targetLogs}`)
console.log(`Max steps    : ${config.maxSteps}`)
console.log(`Reset mode   : ${config.resetMode}`)
console.log(`Stuck thresh : ${config.stuckThreshold} steps`)
console.log(`Epsilon      : ${config.epsilon}`)
console.log(`MC Server    : ${config.minecraftHost}:${config.minecraftPort}`)
console.log(`Output       : ${config.outputPath}`)
console.log('='.repeat(60))

// Create bot
const bot = createBot({
    host: config.minecraftHost,
    port: config.minecraftPort,
    username: 'RLWoodBot',
    version: '1.20.1',
})

bot.once('spawn', async () => {
    console.log('[RL] Bot spawned. Waiting for chunks...')
    await bot.waitForTicks(60) // Wait for world to load

    // Optional: metrics collector for screenshots + control tracking
    let metricsCollector = null
    if (config.metricsEnabled) {
        metricsCollector = new MetricsCollector('RLWoodBot', 'rl')
        await metricsCollector.initialize(
            'src/metrics/agent_metrics/',
            `RL Wood Chopping - ${config.policy}`,
            config.screenshotsEnabled,
            config.screenshotsEnabled ? config.viewerPort : null
        )
        metricsCollector.startControlTracking(
            bot, 50,
            config.screenshotsEnabled,
            config.screenshotsEnabled ? config.viewerPort : null
        )
    }

    // Create environment and agent
    const env = new WoodChopEnvironment(bot, {
        targetLogs: config.targetLogs,
        maxSteps: config.maxSteps,
        resetMode: config.resetMode,
        stuckThreshold: config.stuckThreshold,
        stuckCheckInterval: config.stuckCheckInterval,
        metricsCollector,
    })

    const agent = new RLAgent({
        policy: config.policy,
        modelUrl: config.modelUrl,
        epsilon: config.epsilon,
    })

    // Run episodes
    const allSummaries = []

    for (let ep = 0; ep < config.numEpisodes; ep++) {
        console.log(`\n${'─'.repeat(40)}`)
        console.log(`[RL] Episode ${ep + 1}/${config.numEpisodes}`)
        console.log('─'.repeat(40))

        const { observation, meta } = await env.reset()
        let obs = observation
        let obsMeta = meta
        let done = false
        let truncated = false

        while (!done && !truncated) {
            const actionIndex = await agent.selectAction(obs, obsMeta)
            const result = await env.step(actionIndex)
            obs = result.observation
            obsMeta = result.info
            done = result.done
            truncated = result.truncated
        }

        // Episode finished
        const summary = env.getEpisodeSummary()
        summary.episode = ep + 1
        allSummaries.push(summary)

        console.log(`[RL] Episode ${ep + 1} finished:`)
        console.log(`     Steps: ${summary.steps} | Reward: ${summary.total_reward} | ` +
            `Logs: ${summary.logs_collected}/${summary.target_logs} | ` +
            `Success: ${summary.success} | Unstuck: ${summary.stuck_recoveries}x`)

        // Save transitions for training
        const transitions = env.exportEpisodeData()
        saveTransitionsJSONL(transitions, config.outputPath)
        saveEpisodeSummary(summary)

        // Brief pause between episodes
        await bot.waitForTicks(20)
    }

    // Final report
    console.log('\n' + '='.repeat(60))
    console.log('TRAINING COMPLETE')
    console.log('='.repeat(60))

    const successes = allSummaries.filter(s => s.success).length
    const avgReward = allSummaries.reduce((s, ep) => s + ep.total_reward, 0) / allSummaries.length
    const avgSteps = allSummaries.reduce((s, ep) => s + ep.steps, 0) / allSummaries.length
    const totalUnstucks = allSummaries.reduce((s, ep) => s + ep.stuck_recoveries, 0)

    console.log(`Episodes      : ${config.numEpisodes}`)
    console.log(`Successes     : ${successes}/${config.numEpisodes} (${(successes / config.numEpisodes * 100).toFixed(1)}%)`)
    console.log(`Avg reward    : ${avgReward.toFixed(2)}`)
    console.log(`Avg steps     : ${avgSteps.toFixed(0)}`)
    console.log(`Total unstucks: ${totalUnstucks}`)
    console.log(`Agent stats   :`, JSON.stringify(agent.getStats(), null, 2))

    // Export metrics
    if (metricsCollector) {
        metricsCollector.completeTask(successes > 0)
        metricsCollector.stopControlTracking()
        await metricsCollector.export(bot)
    }

    bot.quit()
    process.exit(0)
})

bot.on('error', (err) => {
    console.error('[RL] Bot error:', err.message)
})

bot.on('kicked', (reason) => {
    console.error('[RL] Bot kicked:', reason)
    process.exit(1)
})