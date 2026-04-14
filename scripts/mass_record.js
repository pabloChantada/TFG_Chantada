/**
 * Grabación masiva de episodios HTN para el dataset de IL.
 *
 * Lanza N episodios del agente HTN secuencialmente. Cada episodio guarda
 * un JSONL en data/recordings/. Al terminar, separa y concatena:
 *   - episodios exitosos (objetivo cumplido)
 *   - episodios fallidos/incompletos
 *
 * Uso:
 *   node scripts/mass_record.js --episodes 20
 *   node scripts/mass_record.js --episodes 50 --minecraft-port 25565
 *   node scripts/mass_record.js --episodes 20 --clean
 *
 * Paso siguiente:
 *   python scripts/prepare_dataset.py --input data/train.jsonl --output data/train_clean.jsonl
 *
 * SEED: 7145048257670320778
 */

import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import { validateSetup, startServer, stopServer, findAvailablePort } from './paper_server.js'

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
    .option('recordings-dir', {
        type: 'string',
        description: 'Directorio donde dataset_recorder guarda los JSONL y screenshots',
        default: 'data/recordings'
    })
    .option('output', {
        type: 'string',
        description: 'Ruta JSONL principal (por defecto: solo episodios exitosos)',
        default: 'data/train.jsonl'
    })
    .option('output-success', {
        type: 'string',
        description: 'Ruta JSONL para episodios exitosos',
        default: 'data/train_success.jsonl'
    })
    .option('output-failed', {
        type: 'string',
        description: 'Ruta JSONL para episodios fallidos/incompletos',
        default: 'data/train_failed.jsonl'
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
    .option('server-dir', {
        type: 'string',
        description: 'Directorio del servidor Paper',
        default: 'server'
    })
    .option('no-server', {
        type: 'boolean',
        description: 'No gestionar el servidor (usar servidor externo)',
        default: false
    })
    .option('fixed-seed', {
        type: 'boolean',
        description: 'Usar la seed fija de server/seed.txt en lugar de una aleatoria',
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
 * Encuentra el número de episodio más alto ya grabado en recordingsDir.
 * Busca ficheros con el patrón Rec_N_* y devuelve el mayor N encontrado (0 si ninguno).
 */
function getLastEpisodeNumber(recordingsDir) {
    if (!fs.existsSync(recordingsDir)) return 0
    const files = fs.readdirSync(recordingsDir)
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

function runEpisode(episode, agentName, mcPort, viewerPort) {
    return new Promise((resolve) => {
        const agentArgs = [
            'src/agents/add_agent.js',
            '--name', agentName,
            '--type', 'htn',
            '--minecraft-port', String(mcPort),
            '--viewer-port', String(viewerPort),
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

            resolve({ success, duration_s: parseFloat(duration) })
        })

        child.on('error', (err) => {
            if (settled) return
            settled = true
            console.error(`Episodio ${episode} error de spawn: ${err.message}`)
            resolve({ success: false, duration_s: 0 })
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

function listRecordingJsonlFiles(recordingsDir) {
    if (!fs.existsSync(recordingsDir)) return []
    return fs.readdirSync(recordingsDir)
        .filter(f => f.endsWith('.jsonl'))
        .map(f => path.join(recordingsDir, f))
        .sort()
}

function diffNewFiles(beforeFiles, afterFiles) {
    const beforeSet = new Set(beforeFiles)
    return afterFiles.filter(f => !beforeSet.has(f))
}

/**
 * Concatena una lista de ficheros .jsonl en outputPath.
 * Devuelve el total de líneas escritas.
 */
async function mergeJsonlFiles(jsonlFiles, outputPath) {
    if (jsonlFiles.length === 0) return 0

    fs.mkdirSync(path.dirname(outputPath), { recursive: true })

    const out = fs.createWriteStream(outputPath, { flags: 'w' })
    let totalLines = 0
    for (const file of jsonlFiles) {
        const content = fs.readFileSync(file, 'utf-8')
        const lines = content.split('\n').filter(l => l.trim())
        for (const line of lines) {
            out.write(line + '\n')
            totalLines++
        }
    }
    await new Promise((resolve, reject) => {
        out.on('error', reject)
        out.end(resolve)
    })

    return totalLines
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
    const { episodes } = args
    const mcPort         = args['minecraft-port']
    const basePort       = args['base-port']
    const recordingsDir  = args['recordings-dir']
    const outputJsonl    = args.output
    const outputSuccess  = args['output-success']
    const outputFailed   = args['output-failed']
    const pauseSec       = args.pause
    const serverDir      = path.resolve(args['server-dir'])
    const manageServer   = !args['no-server']
    const fixedSeed      = args['fixed-seed']

    console.log('╔══════════════════════════════════════════════════╗')
    console.log('║            MASS RECORDING — IL DATASET          ║')
    console.log('╠══════════════════════════════════════════════════╣')
    console.log(`║  Episodios:    ${String(episodes).padEnd(35)}║`)
    console.log(`║  Puerto MC:    ${String(mcPort).padEnd(35)}║`)
    console.log(`║  Servidor:     ${(manageServer ? serverDir : 'externo').padEnd(35)}║`)
    console.log(`║  Recordings:   ${recordingsDir.padEnd(35)}║`)
    console.log(`║  Salida:       ${outputJsonl.padEnd(35)}║`)
    console.log(`║  Éxitos:       ${outputSuccess.padEnd(35)}║`)
    console.log(`║  Fallidos:     ${outputFailed.padEnd(35)}║`)
    console.log('╚══════════════════════════════════════════════════╝\n')

    if (manageServer) {
        await validateSetup(serverDir)
        console.log('[MASS] Servidor Paper validado ✓\n')
    }

    if (args.clean) {
        console.log('[MASS] Limpiando directorio de grabaciones...')
        if (fs.existsSync(recordingsDir)) fs.rmSync(recordingsDir, { recursive: true, force: true })
    }
    fs.mkdirSync(recordingsDir, { recursive: true })

    const startEp = getLastEpisodeNumber(recordingsDir) + 1
    if (startEp > 1) console.log(`[MASS] Continuando desde episodio ${startEp} (${startEp - 1} ya grabados)\n`)

    // Handler para limpiar el servidor si se interrumpe el script
    let currentServer = null
    let cleaningUp = false
    const cleanup = async () => {
        if (cleaningUp) return
        cleaningUp = true

        process.off('SIGINT', cleanup)
        process.off('SIGTERM', cleanup)

        if (currentServer) {
            console.log('\n[MASS] Interrumpido — parando servidor...')
            await stopServer(currentServer)
            currentServer = null
        }
        process.exit(1)
    }
    process.on('SIGINT', cleanup)
    process.on('SIGTERM', cleanup)

    // ── Grabar episodios ──────────────────────────────────────────────
    const results = []
    const successFiles = []
    const failedFiles = []
    for (let ep = startEp; ep < startEp + episodes; ep++) {
        let episodeMcPort = mcPort

        if (manageServer) {
            episodeMcPort = await findAvailablePort(mcPort)
            if (episodeMcPort !== mcPort) {
                console.warn(`[MASS] Puerto ${mcPort} ocupado, usando ${episodeMcPort} para el episodio ${ep}.`)
            }
        }

        // Arrancar servidor con mundo nuevo
        if (manageServer) {
            try {
                const server = await startServer(serverDir, episodeMcPort, { useSeed: fixedSeed })
                currentServer = server.process
                console.log(`[MASS] Episodio ${ep} — seed=${server.seed} — mcPort=${episodeMcPort}`)
            } catch (e) {
                console.error(`[MASS] Error arrancando servidor para episodio ${ep}: ${e.message}`)
                results.push({ success: false, duration_s: 0 })
                break
            }
        }

        const beforeFiles = listRecordingJsonlFiles(recordingsDir)
        const agentName = generateAgentName(ep)
        const result = await runEpisode(ep, agentName, episodeMcPort, basePort)
        const afterFiles = listRecordingJsonlFiles(recordingsDir)
        const newFiles = diffNewFiles(beforeFiles, afterFiles)

        if (result.success) successFiles.push(...newFiles)
        else failedFiles.push(...newFiles)

        results.push({ ...result, jsonlFiles: newFiles })

        // Parar servidor
        if (manageServer && currentServer) {
            await stopServer(currentServer)
            currentServer = null
        }

        if (ep < startEp + episodes - 1) {
            const wait = manageServer ? 2 : pauseSec
            console.log(`\n Esperando ${wait}s...`)
            await sleep(wait * 1000)
        }
    }

    // ── Resumen ───────────────────────────────────────────────────────
    const succeeded   = results.filter(r => r.success).length
    const avgDuration = results.reduce((s, r) => s + r.duration_s, 0) / results.length

    console.log('\n' + '═'.repeat(60))
    console.log('RESUMEN')
    console.log('═'.repeat(60))
    console.log(`  Episodios OK:   ${succeeded}/${episodes}`)
    console.log(`  Duración media: ${avgDuration.toFixed(1)}s`)

    if (succeeded === 0) {
        console.error('\nNingún episodio completado correctamente.')
        process.exit(1)
    }

    // ── Concatenar JSONL de todas las grabaciones ─────────────────────
    console.log('\n' + '─'.repeat(60))
    console.log('COMBINANDO GRABACIONES...')
    console.log('─'.repeat(60))

    const totalSuccessLines = await mergeJsonlFiles(successFiles, outputSuccess)
    const totalFailedLines = await mergeJsonlFiles(failedFiles, outputFailed)

    if (totalSuccessLines === 0) {
        console.error('No se generaron episodios exitosos con frames.')
        process.exit(1)
    }

    // Mantener compatibilidad: output principal = solo episodios exitosos
    const outputMainPath = path.resolve(outputJsonl)
    const outputSuccessPath = path.resolve(outputSuccess)
    if (outputMainPath !== outputSuccessPath) {
        fs.copyFileSync(outputSuccess, outputJsonl)
    }

    console.log(`\nDataset principal (éxitos): ${outputJsonl}  (${totalSuccessLines} frames)`)
    console.log(`Dataset éxitos:              ${outputSuccess}  (${totalSuccessLines} frames)`)
    console.log(`Dataset fallidos:            ${outputFailed}  (${totalFailedLines} frames)`)
    console.log('\nPara limpiar y preparar el dataset antes de entrenar:')
    console.log(`  python scripts/prepare_dataset.py --input ${outputJsonl} --output data/train_clean.jsonl`)
}

main().catch(err => {
    console.error('Error fatal:', err)
    process.exit(1)
})
