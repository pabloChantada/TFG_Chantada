/**
 * Mass recording automation for RL dataset generation.
 * 
 * Runs the HTN agent multiple times to collect diverse training episodes,
 * then automatically generates the unified JSONL dataset.
 * 
 * IMPORTANT: Requires a Minecraft server with LAN open before running.
 * 
 * Usage:
 *   node src/evaluation/mass_record.js                              # 10 episodes, default settings
 *   node src/evaluation/mass_record.js --episodes 50                # 50 episodes
 *   node src/evaluation/mass_record.js --episodes 20 --balance      # 20 episodes + auto-balance
 *   node src/evaluation/mass_record.js --minecraft-port 25565       # custom MC port
 * 
 * Pipeline:
 *   1. For each episode:  spawn HTN agent → run chop task → export metrics → agent exits
 *   2. After all episodes: run dataset.py to merge all metrics into train.jsonl
 *   3. Optionally:        run balance_dataset.py to filter repetitive frames
 */

import { spawn, execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'

const args = yargs(hideBin(process.argv))
    .option('episodes', {
        type: 'number',
        description: 'Number of recording episodes to run',
        default: 10
    })
    .option('minecraft-port', {
        type: 'number',
        description: 'Minecraft server port',
        default: 25565
    })
    .option('base-port', {
        type: 'number',
        description: 'Base viewer port for prismarine-viewer',
        default: 3000
    })
    .option('metrics-dir', {
        type: 'string',
        description: 'Directory for metrics output',
        default: 'src/metrics/agent_metrics'
    })
    .option('output', {
        type: 'string',
        description: 'Output JSONL path for the final dataset',
        default: 'data/train.jsonl'
    })
    .option('balance', {
        type: 'boolean',
        description: 'Run balance_dataset.py after generation',
        default: false
    })
    .option('max-repeat', {
        type: 'number',
        description: 'Max consecutive identical actions (for balancing)',
        default: 3
    })
    .option('upsample-rare', {
        type: 'boolean',
        description: 'Upsample rare actions during balancing',
        default: false
    })
    .option('pause', {
        type: 'number',
        description: 'Seconds to wait between episodes (server recovery)',
        default: 5
    })
    .option('clean', {
        type: 'boolean',
        description: 'Clean metrics directory before starting',
        default: false
    })
    .option('agent-type', {
        type: 'string',
        description: 'Agent type to record (htn or rl)',
        default: 'htn',
        choices: ['htn']
    })
    .help()
    .example('node src/evaluation/mass_record.js --episodes 20', 'Record 20 episodes')
    .example('node src/evaluation/mass_record.js --episodes 50 --balance --upsample-rare', 'Record + balance + upsample')
    .parse()

// ── Helpers ─────────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

function generateAgentName(episode) {
    // Short random suffix to avoid Minecraft username conflicts
    const suffix = Math.random().toString(36).substring(2, 8)
    return `Rec_${episode}_${suffix}`
}

/**
 * Run a single HTN agent episode and wait for it to finish.
 * Returns { success, duration_s }
 */
function runEpisode(episode, agentName, mcPort, viewerPort, metricsDir) {
    return new Promise((resolve) => {
        const metricsPath = path.join(metricsDir, `${agentName}_metrics.json`)

        const agentArgs = [
            'src/agents/add_agent.js',
            '--name', agentName,
            '--type', args['agent-type'],
            '--minecraft-port', String(mcPort),
            '--viewer-port', String(viewerPort),
            '--metrics-path', metricsPath,
        ]

        console.log(`\n${'─'.repeat(60)}`)
        console.log(`Episode ${episode}: agent=${agentName} viewer=:${viewerPort}`)
        console.log(`${'─'.repeat(60)}`)

        const startTime = Date.now()

        const child = spawn('node', agentArgs, {
            stdio: 'inherit',
            cwd: process.cwd(),
        })

        let settled = false

        child.on('close', (code) => {
            if (settled) return
            settled = true
            const duration = ((Date.now() - startTime) / 1000).toFixed(1)
            const success = code === 0

            if (success) {
                console.log(`✓ Episode ${episode} completed in ${duration}s`)
            } else {
                console.warn(`✗ Episode ${episode} exited with code ${code} after ${duration}s`)
            }

            // Check if metrics file was created
            const metricsExists = fs.existsSync(metricsPath)
            if (!metricsExists) {
                console.warn(`  ⚠ No metrics file found at ${metricsPath}`)
            }

            resolve({ success, duration_s: parseFloat(duration), metricsExists })
        })

        child.on('error', (err) => {
            if (settled) return
            settled = true
            console.error(`✗ Episode ${episode} spawn error: ${err.message}`)
            resolve({ success: false, duration_s: 0, metricsExists: false })
        })

        // Safety timeout: kill agent if it takes too long (5 minutes)
        const timeout = setTimeout(() => {
            if (!settled) {
                console.warn(`⚠ Episode ${episode} timed out, killing...`)
                child.kill('SIGTERM')
            }
        }, 5 * 60 * 1000)
        timeout.unref()
    })
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
    const { episodes, balance } = args
    const mcPort = args['minecraft-port']
    const basePort = args['base-port']
    const metricsDir = args['metrics-dir']
    const outputJsonl = args.output
    const pauseSec = args.pause

    console.log('╔══════════════════════════════════════════════════╗')
    console.log('║         MASS RECORDING FOR RL DATASET           ║')
    console.log('╠══════════════════════════════════════════════════╣')
    console.log(`║  Episodes:       ${String(episodes).padEnd(31)}║`)
    console.log(`║  MC port:        ${String(mcPort).padEnd(31)}║`)
    console.log(`║  Viewer port:    ${String(basePort).padEnd(31)}║`)
    console.log(`║  Metrics dir:    ${metricsDir.padEnd(31)}║`)
    console.log(`║  Output:         ${outputJsonl.padEnd(31)}║`)
    console.log(`║  Auto-balance:   ${String(balance).padEnd(31)}║`)
    console.log('╚══════════════════════════════════════════════════╝')
    console.log()

    // Clean metrics if requested
    if (args.clean) {
        console.log('[MASS] Cleaning metrics directory...')
        if (fs.existsSync(metricsDir)) {
            fs.rmSync(metricsDir, { recursive: true, force: true })
        }
    }
    fs.mkdirSync(metricsDir, { recursive: true })

    // ── Record episodes ────────────────────────────────────────────────
    const results = []
    for (let ep = 1; ep <= episodes; ep++) {
        const agentName = generateAgentName(ep)
        // Use the same viewer port for all (sequential, not parallel)
        const result = await runEpisode(ep, agentName, mcPort, basePort, metricsDir)
        results.push(result)

        // Pause between episodes to let the server recover
        if (ep < episodes) {
            console.log(`\n⏳ Waiting ${pauseSec}s before next episode...`)
            await sleep(pauseSec * 1000)
        }
    }

    // ── Summary ────────────────────────────────────────────────────────
    const succeeded = results.filter(r => r.success).length
    const withMetrics = results.filter(r => r.metricsExists).length
    const avgDuration = results.reduce((s, r) => s + r.duration_s, 0) / results.length

    console.log('\n' + '═'.repeat(60))
    console.log('RECORDING SUMMARY')
    console.log('═'.repeat(60))
    console.log(`  Episodes completed: ${succeeded}/${episodes}`)
    console.log(`  With metrics:       ${withMetrics}/${episodes}`)
    console.log(`  Avg duration:       ${avgDuration.toFixed(1)}s`)

    if (withMetrics === 0) {
        console.error('\n✗ No metrics files generated. Cannot build dataset.')
        process.exit(1)
    }

    // ── Build dataset ──────────────────────────────────────────────────
    console.log('\n' + '─'.repeat(60))
    console.log('BUILDING DATASET...')
    console.log('─'.repeat(60))

    try {
        execSync(
            `python src/evaluation/dataset.py --metrics_dir "${metricsDir}" --output_jsonl "${outputJsonl}"`,
            { stdio: 'inherit', cwd: process.cwd() }
        )
    } catch (err) {
        console.error(`✗ Dataset generation failed: ${err.message}`)
        process.exit(1)
    }

    // ── Balance dataset (optional) ─────────────────────────────────────
    if (balance) {
        const balancedPath = outputJsonl.replace('.jsonl', '_balanced.jsonl')
        console.log('\n' + '─'.repeat(60))
        console.log('BALANCING DATASET...')
        console.log('─'.repeat(60))

        let cmd = `python src/evaluation/balance_dataset.py --input "${outputJsonl}" --output "${balancedPath}" --max_repeat ${args['max-repeat']}`
        if (args['upsample-rare']) {
            cmd += ' --upsample_rare'
        }

        try {
            execSync(cmd, { stdio: 'inherit', cwd: process.cwd() })
            console.log(`\n✓ Balanced dataset: ${balancedPath}`)
        } catch (err) {
            console.error(`✗ Balancing failed: ${err.message}`)
        }
    }

    console.log('\n✓ Done! Dataset ready for training.')
    console.log(`  Raw:      ${outputJsonl}`)
    if (balance) {
        console.log(`  Balanced: ${outputJsonl.replace('.jsonl', '_balanced.jsonl')}`)
    }
    console.log(`\n  Train with: python src/rl/train.py --mode bc --jsonl ${balance ? outputJsonl.replace('.jsonl', '_balanced.jsonl') : outputJsonl}`)
}

main().catch(err => {
    console.error('Fatal error:', err)
    process.exit(1)
})
