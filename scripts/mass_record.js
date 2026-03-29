/**
 * Grabación masiva de episodios HTN para el dataset de IL.
 *
 * Lanza N episodios del agente HTN secuencialmente, recoge las métricas
 * y genera el JSONL de entrenamiento con dataset.py.
 *
 * Uso:
 *   node scripts/mass_record.js --episodes 20
 *   node scripts/mass_record.js --episodes 50 --minecraft-port 25565
 *   node scripts/mass_record.js --episodes 20 --clean
 *
 * Paso siguiente (limpieza manual del dataset):
 *   python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl
 */

import { spawn, execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'

const args = yargs(hideBin(process.argv))
    .option('episodes', {
        type: 'number',
        description: 'Número de episodios a grabar',
        default: 10
    })
    .option('minecraft-port', {
        type: 'number',
        description: 'Puerto del servidor Minecraft',
        default: 25565
    })
    .option('base-port', {
        type: 'number',
        description: 'Puerto base para prismarine-viewer',
        default: 3000
    })
    .option('metrics-dir', {
        type: 'string',
        description: 'Directorio de salida de métricas',
        default: 'src/metrics/agent_metrics'
    })
    .option('output', {
        type: 'string',
        description: 'Ruta JSONL de salida',
        default: 'data/train.jsonl'
    })
    .option('pause', {
        type: 'number',
        description: 'Segundos de espera entre episodios',
        default: 5
    })
    .option('clean', {
        type: 'boolean',
        description: 'Limpiar el directorio de métricas antes de empezar',
        default: false
    })
    .help()
    .parse()

// ── Helpers ──────────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

function generateAgentName(episode) {
    const suffix = Math.random().toString(36).substring(2, 8)
    return `Rec_${episode}_${suffix}`
}

/**
 * Encuentra el número de episodio más alto ya grabado en metricsDir.
 * Busca ficheros con el patrón Rec_N_* y devuelve el mayor N encontrado (0 si ninguno).
 */
function getLastEpisodeNumber(metricsDir) {
    if (!fs.existsSync(metricsDir)) return 0
    const files = fs.readdirSync(metricsDir)
    let max = 0
    for (const file of files) {
        const match = file.match(/^Rec_(\d+)_/)
        if (match) {
            const n = parseInt(match[1], 10)
            if (n > max) max = n
        }
    }
    return max
}

function runEpisode(episode, agentName, mcPort, viewerPort, metricsDir) {
    return new Promise((resolve) => {
        const metricsPath = path.join(metricsDir, `${agentName}_metrics.json`)

        const agentArgs = [
            'src/agents/add_agent.js',
            '--name', agentName,
            '--type', 'htn',
            '--minecraft-port', String(mcPort),
            '--viewer-port', String(viewerPort),
            '--metrics-path', metricsPath,
        ]

        console.log(`\n${'─'.repeat(60)}`)
        console.log(`Episodio ${episode}: ${agentName}  viewer=:${viewerPort}`)
        console.log(`${'─'.repeat(60)}`)

        const startTime = Date.now()
        const child = spawn('node', agentArgs, { stdio: 'inherit', cwd: process.cwd() })
        let settled = false

        child.on('close', (code) => {
            if (settled) return
            settled = true
            const duration = ((Date.now() - startTime) / 1000).toFixed(1)
            const success = code === 0

            if (success) {
                console.log(`Episodio ${episode} OK — ${duration}s`)
            } else {
                console.warn(`Episodio ${episode} salió con código ${code} tras ${duration}s`)
            }

            const metricsExists = fs.existsSync(metricsPath)
            if (!metricsExists) console.warn(`  Sin fichero de métricas en ${metricsPath}`)

            resolve({ success, duration_s: parseFloat(duration), metricsExists })
        })

        child.on('error', (err) => {
            if (settled) return
            settled = true
            console.error(`Episodio ${episode} error de spawn: ${err.message}`)
            resolve({ success: false, duration_s: 0, metricsExists: false })
        })

        // Timeout de seguridad: 5 minutos por episodio
        const timeout = setTimeout(() => {
            if (!settled) {
                console.warn(`Episodio ${episode} timeout, matando proceso...`)
                child.kill('SIGTERM')
            }
        }, 5 * 60 * 1000)
        timeout.unref()
    })
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
    const { episodes } = args
    const mcPort      = args['minecraft-port']
    const basePort    = args['base-port']
    const metricsDir  = args['metrics-dir']
    const outputJsonl = args.output
    const pauseSec    = args.pause

    console.log('╔══════════════════════════════════════════════════╗')
    console.log('║            MASS RECORDING — IL DATASET          ║')
    console.log('╠══════════════════════════════════════════════════╣')
    console.log(`║  Episodios:  ${String(episodes).padEnd(37)}║`)
    console.log(`║  Puerto MC:  ${String(mcPort).padEnd(37)}║`)
    console.log(`║  Métricas:   ${metricsDir.padEnd(37)}║`)
    console.log(`║  Salida:     ${outputJsonl.padEnd(37)}║`)
    console.log('╚══════════════════════════════════════════════════╝\n')

    if (args.clean) {
        console.log('[MASS] Limpiando directorio de métricas...')
        if (fs.existsSync(metricsDir)) fs.rmSync(metricsDir, { recursive: true, force: true })
    }
    fs.mkdirSync(metricsDir, { recursive: true })

    const startEp = getLastEpisodeNumber(metricsDir) + 1
    if (startEp > 1) console.log(`[MASS] Continuando desde episodio ${startEp} (${startEp - 1} ya grabados)\n`)

    // ── Grabar episodios ──────────────────────────────────────────────
    const results = []
    for (let ep = startEp; ep < startEp + episodes; ep++) {
        const agentName = generateAgentName(ep)
        const result = await runEpisode(ep, agentName, mcPort, basePort, metricsDir)
        results.push(result)

        if (ep < startEp + episodes - 1) {
            console.log(`\n Esperando ${pauseSec}s...`)
            await sleep(pauseSec * 1000)
        }
    }

    // ── Resumen ───────────────────────────────────────────────────────
    const succeeded   = results.filter(r => r.success).length
    const withMetrics = results.filter(r => r.metricsExists).length
    const avgDuration = results.reduce((s, r) => s + r.duration_s, 0) / results.length

    console.log('\n' + '═'.repeat(60))
    console.log('RESUMEN')
    console.log('═'.repeat(60))
    console.log(`  Episodios OK:   ${succeeded}/${episodes}`)
    console.log(`  Con métricas:   ${withMetrics}/${episodes}`)
    console.log(`  Duración media: ${avgDuration.toFixed(1)}s`)

    if (withMetrics === 0) {
        console.error('\nSin ficheros de métricas. No se puede generar dataset.')
        process.exit(1)
    }

    // ── Generar dataset ───────────────────────────────────────────────
    console.log('\n' + '─'.repeat(60))
    console.log('GENERANDO DATASET...')
    console.log('─'.repeat(60))

    try {
        execSync(
            `python scripts/dataset.py --metrics_dir "${metricsDir}" --output_jsonl "${outputJsonl}"`,
            { stdio: 'inherit', cwd: process.cwd() }
        )
    } catch (err) {
        console.error(`Error al generar el dataset: ${err.message}`)
        process.exit(1)
    }

    console.log('\nDataset generado:', outputJsonl)
    console.log('  Para limpiar antes de entrenar:')
    console.log(`  python scripts/prepare_dataset.py --input ${outputJsonl} --output data/train_clean.jsonl`)
}

main().catch(err => {
    console.error('Error fatal:', err)
    process.exit(1)
})
