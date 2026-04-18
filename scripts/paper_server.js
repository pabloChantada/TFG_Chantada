/**
 * Gestión del ciclo de vida de un servidor Paper para grabación automática.
 *
 * Exporta startServer / stopServer para que mass_record.js pueda:
 *   1. Arrancar un mundo nuevo con seed aleatoria antes de cada episodio
 *   2. Parar el servidor limpiamente después de cada episodio
 *
 * Requisitos:
 *   - java en PATH
 *   - server/paper-1.20.1.jar descargado manualmente
 */

import { spawn as cpSpawn } from 'child_process'
import fs   from 'fs'
import path from 'path'
import crypto from 'crypto'
import net from 'net'

const JAR_NAME   = 'paper-1.20.1.jar'
const WORLD_DIRS = ['world', 'world_nether', 'world_the_end']
const STOP_PROMISES = new WeakMap()
// ── Helpers internos ────────────────────────────────────────────────────────

function ensureEula(serverDir) {
    fs.writeFileSync(path.join(serverDir, 'eula.txt'), 'eula=true\n')
}

function deleteWorlds(serverDir) {
    for (const dir of WORLD_DIRS) {
        const p = path.join(serverDir, dir)
        if (fs.existsSync(p)) fs.rmSync(p, { recursive: true, force: true })
    }
}

/**
 * Crea un datapack que fuerza el overworld a generar un único bioma (forest)
 * con terreno normal (noise generator). Paper 1.20.1 no soporta
 * single_biome_surface vía server.properties, así que usamos esto.
 */
function installSingleBiomeDatapack(serverDir) {
    const dpDir  = path.join(serverDir, 'world', 'datapacks', 'single_biome_forest')
    const dimDir = path.join(dpDir, 'data', 'minecraft', 'dimension')
    const tagDir = path.join(dpDir, 'data', 'minecraft', 'tags', 'functions')
    const fnDir  = path.join(dpDir, 'data', 'training', 'functions')
    fs.mkdirSync(dimDir, { recursive: true })
    fs.mkdirSync(tagDir, { recursive: true })
    fs.mkdirSync(fnDir,  { recursive: true })

    // pack.mcmeta — pack_format 15 = MC 1.20–1.20.1
    fs.writeFileSync(path.join(dpDir, 'pack.mcmeta'), JSON.stringify({
        pack: {
            pack_format: 15,
            description: 'Training: forest biome + haste'
        }
    }, null, 2))

    // Override overworld dimension: normal noise terrain, fixed biome = forest
    fs.writeFileSync(path.join(dimDir, 'overworld.json'), JSON.stringify({
        type: 'minecraft:overworld',
        generator: {
            type: 'minecraft:noise',
            biome_source: {
                type: 'minecraft:fixed',
                biome: 'minecraft:forest'
            },
            settings: 'minecraft:overworld'
        }
    }, null, 2))

    // Tick function: haste 255 permanente + día forzado (sin depender de que el bot sea op)
    fs.writeFileSync(path.join(tagDir, 'tick.json'), JSON.stringify(
        { values: ['training:tick'] }, null, 2
    ))
    fs.writeFileSync(path.join(fnDir, 'tick.mcfunction'), [
        'effect give @a minecraft:saturation 2 255 true',
        'time set day',
    ].join('\n') + '\n')
}

function writeServerProperties(serverDir, seed, port = 25565) {
    const props = [
        `server-port=${port}`,
        'online-mode=false',
        `level-seed=${seed}`,
        'gamemode=survival',
        'difficulty=peaceful',
        'spawn-protection=0',
        'max-players=5',
        'view-distance=10',
        'enable-command-block=false',
        'motd=Training Server',
        'level-name=world',
        'generate-structures=true',
        'pvp=false',
        'allow-nether=false',
        'spawn-monsters=false',
        'spawn-animals=true',
    ].join('\n') + '\n'

    fs.writeFileSync(path.join(serverDir, 'server.properties'), props)
}

function generateSeed() {
    return crypto.randomInt(0, 2 ** 48 - 1)
}

function readSeedFile(serverDir) {
    const seedPath = path.join(serverDir, 'seed.txt')
    if (!fs.existsSync(seedPath)) {
        throw new Error(
            `--fixed-seed activo pero no existe ${seedPath}.\n` +
            `Crea ese fichero con un número entero, p.ej.: echo 7145048257670320778 > server/seed.txt`
        )
    }
    const raw = fs.readFileSync(seedPath, 'utf8').trim()
    if (!/^-?\d+$/.test(raw)) {
        throw new Error(`seed.txt contiene "${raw}", se esperaba un entero.`)
    }
    return raw   // devolver como string para evitar pérdida de precisión en BigInt seeds
}

function isPortInUse(port, host = '0.0.0.0') {
    return new Promise((resolve, reject) => {
        const tester = net.createServer()

        tester.once('error', (err) => {
            if (err.code === 'EADDRINUSE' || err.code === 'EACCES') {
                resolve(true)
            } else {
                reject(err)
            }
        })

        tester.once('listening', () => {
            tester.close(() => resolve(false))
        })

        tester.listen(port, host)
    })
}

async function findAvailablePort(startPort = 25565, maxAttempts = 30) {
    let port = startPort
    for (let i = 0; i < maxAttempts; i++) {
        let inUse = true
        try {
            // eslint-disable-next-line no-await-in-loop
            inUse = await isPortInUse(port)
        } catch {
            // Si no podemos comprobar este puerto por una causa inesperada,
            // lo tratamos como no utilizable y probamos el siguiente.
            inUse = true
        }
        if (!inUse) return port
        port++
    }

    throw new Error(`No se encontró puerto libre desde ${startPort} tras ${maxAttempts} intentos.`)
}

// ── Esperar a que el servidor esté listo ─────────────────────────────────────

function waitForReady(serverProcess, timeout = 120_000) {
    return new Promise((resolve, reject) => {
        let settled = false

        const failFastPatterns = [
            'FAILED TO BIND TO PORT',
            'Address already in use',
            'Failed to initialize server',
            'Encountered an unexpected exception'
        ]

        const cleanup = () => {
            clearTimeout(timer)
            serverProcess.stdout.off('data', onStdout)
            serverProcess.stderr.off('data', onStderr)
            serverProcess.off('close', onClose)
        }

        const finishResolve = () => {
            if (settled) return
            settled = true
            cleanup()
            resolve()
        }

        const finishReject = (err) => {
            if (settled) return
            settled = true
            cleanup()
            reject(err)
        }

        const timer = setTimeout(() => {
            finishReject(new Error(`Server start timeout (${timeout / 1000}s)`))
        }, timeout)

        const inspectLine = (text) => {
            if (text.includes('Done (')) {
                finishResolve()
                return
            }

            for (const pattern of failFastPatterns) {
                if (text.includes(pattern)) {
                    finishReject(new Error(`Server startup failure: ${pattern}`))
                    return
                }
            }
        }

        const onStdout = (chunk) => {
            const text = chunk.toString()
            inspectLine(text)
        }

        const onStderr = (chunk) => {
            const text = chunk.toString()
            inspectLine(text)
        }

        const onClose = (code) => {
            finishReject(new Error(`Server exited unexpectedly with code ${code}`))
        }

        serverProcess.stdout.on('data', onStdout)
        serverProcess.stderr.on('data', onStderr)

        serverProcess.on('close', onClose)
    })
}

// ── API pública ──────────────────────────────────────────────────────────────

/**
 * Valida que el entorno esté listo para arrancar el servidor.
 * Lanza Error si falta el .jar o java no está en PATH.
 */
async function validateSetup(serverDir) {
    const jarPath = path.join(serverDir, JAR_NAME)
    if (!fs.existsSync(jarPath)) {
        throw new Error(
            `No se encontró ${JAR_NAME} en ${serverDir}.\n` +
            `Descárgalo de https://papermc.io/downloads/paper y colócalo en ${serverDir}/`
        )
    }

    // Comprobar que java está en PATH
    await new Promise((resolve, reject) => {
        const proc = cpSpawn('java', ['-version'], { stdio: 'ignore' })
        proc.on('close', (code) => {
            if (code === 0) resolve()
            else reject(new Error('java no encontrado en PATH. Instala Java 17+ y asegúrate de que está en PATH.'))
        })
        proc.on('error', () => {
            reject(new Error('java no encontrado en PATH. Instala Java 17+ y asegúrate de que está en PATH.'))
        })
    })
}

/**
 * Arranca un servidor Paper con un mundo nuevo.
 * @param {string} serverDir  - Ruta a la carpeta del servidor
 * @param {number} port       - Puerto del servidor (default 25565)
 * @param {object} opts
 * @param {boolean} opts.useSeed - Si true, lee la seed de server/seed.txt; si no, genera una aleatoria
 * @returns {Promise<{process: ChildProcess, seed: string|number}>}
 */
async function startServer(serverDir, port = 25565, { useSeed = false } = {}) {
    const seed = useSeed ? readSeedFile(serverDir) : generateSeed()

    const portBusy = await isPortInUse(port)
    if (portBusy) {
        throw new Error(`El puerto ${port} está en uso antes de arrancar Paper. Usa otro puerto o cierra el proceso que lo ocupa.`)
    }

    console.log(`[SERVER] Preparando mundo nuevo — seed=${seed}`)

    deleteWorlds(serverDir)
    installSingleBiomeDatapack(serverDir)
    ensureEula(serverDir)
    writeServerProperties(serverDir, seed, port)

    const jarPath = JAR_NAME  // relativo al cwd del proceso hijo
    const serverProcess = cpSpawn('java', [
        '-Xmx1G', '-Xms512M',
        '-jar', jarPath,
        '--nogui'
    ], {
        cwd: serverDir,
        stdio: ['pipe', 'pipe', 'pipe']
    })

    // Mostrar logs del servidor en consola (stderr incluye warnings de Paper)
    serverProcess.stdout.on('data', (chunk) => {
        const lines = chunk.toString().split('\n').filter(l => l.trim())
        for (const line of lines) {
            console.log(`[MC] ${line}`)
        }
    })
    serverProcess.stderr.on('data', (chunk) => {
        const lines = chunk.toString().split('\n').filter(l => l.trim())
        for (const line of lines) {
            console.warn(`[MC:err] ${line}`)
        }
    })

    console.log(`[SERVER] Esperando a que el servidor esté listo...`)
    await waitForReady(serverProcess)
    console.log(`[SERVER] Servidor listo en puerto ${port}`)

    return { process: serverProcess, seed }
}

/**
 * Para el servidor Paper de forma limpia.
 * @param {ChildProcess} serverProcess - Proceso del servidor
 * @param {number} timeout - Timeout en ms para esperar cierre limpio (default 30s)
 */
async function stopServer(serverProcess, timeout = 30_000) {
    if (!serverProcess || serverProcess.exitCode !== null) return

    if (STOP_PROMISES.has(serverProcess)) {
        return STOP_PROMISES.get(serverProcess)
    }

    console.log(`[SERVER] Enviando comando stop...`)

    const stopPromise = new Promise((resolve) => {
        let settled = false

        const finish = () => {
            if (settled) return
            settled = true
            STOP_PROMISES.delete(serverProcess)
            resolve()
        }

        const timer = setTimeout(() => {
            console.warn(`[SERVER] Timeout esperando cierre limpio, forzando kill...`)
            try {
                serverProcess.kill()
            } catch {
                // ignore
            }
            finish()
        }, timeout)

        serverProcess.once('close', () => {
            clearTimeout(timer)
            console.log(`[SERVER] Servidor parado`)
            finish()
        })

        if (serverProcess.stdin && !serverProcess.stdin.destroyed && serverProcess.stdin.writable) {
            serverProcess.stdin.write('stop\n', (err) => {
                if (!err) return
                if (err.code !== 'EPIPE') {
                    console.warn(`[SERVER] Error enviando stop: ${err.message}`)
                }
            })
        } else {
            // stdin ya no está disponible, intentar terminar proceso directamente
            try {
                serverProcess.kill()
            } catch {
                // ignore
            }
        }
    })

    STOP_PROMISES.set(serverProcess, stopPromise)
    return stopPromise
}

export { validateSetup, startServer, stopServer, findAvailablePort }
