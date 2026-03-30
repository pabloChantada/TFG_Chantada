/**
 * Lanzador de agentes Minecraft.
 *
 * Uso:
 *   node scripts/run.js                                      # un agente HTN
 *   node scripts/run.js --agents htn,htn --names Bot1,Bot2  # dos agentes
 *   node scripts/run.js --minecraft-port 25566              # puerto personalizado
 */

import { spawn } from 'child_process'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import fs from 'fs'
import path from 'path'

const args = yargs(hideBin(process.argv))
    .option('agents', {
        type: 'string',
        description: 'Tipos de agente separados por coma (htn, rl)',
        default: 'htn'
    })
    .option('names', {
        type: 'string',
        description: 'Nombres de agente separados por coma (debe coincidir con --agents)',
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
    .option('metrics-clean', {
        alias: 'mc',
        type: 'boolean',
        description: 'Limpiar métricas antes de arrancar',
        default: true
    })
    .option('memory-clean', {
        alias: 'm',
        type: 'boolean',
        description: 'Limpiar memoria de agentes antes de arrancar',
        default: true
    })
    .help()
    .parse()

const agentTypes = args.agents.split(',').map(s => s.trim())
const agentNames = args.names
    ? args.names.split(',').map(s => s.trim())
    : agentTypes.map((_, i) => `Agent${i + 1}`)

if (agentTypes.length !== agentNames.length) {
    console.error('Error: el número de tipos debe coincidir con el número de nombres')
    process.exit(1)
}

if (args['memory-clean']) {
    const memoryDir = path.join(process.cwd(), 'src', 'agents', 'memories')
    if (fs.existsSync(memoryDir)) {
        fs.readdirSync(memoryDir)
            .filter(f => f.endsWith('.json'))
            .forEach(f => fs.unlinkSync(path.join(memoryDir, f)))
        console.log('[INFO] Memoria limpiada')
    }
}

console.log(`[INFO] Lanzando ${agentTypes.length} agente(s)...\n`)

const processes = []

for (let i = 0; i < agentTypes.length; i++) {
    const agentType  = agentTypes[i]
    const agentName  = agentNames[i]
    const viewerPort = args['base-port'] + i

    const cmd = [
        'src/agents/add_agent.js',
        '--name', agentName,
        '--type', agentType,
        '--minecraft-port', String(args['minecraft-port']),
        '--viewer-port', String(viewerPort),
        '--metrics-path', metricsPath,
    ]

    console.log(`[INFO] Agente ${i + 1}/${agentTypes.length}: ${agentName} (${agentType}) → puerto ${viewerPort}`)

    const agentProcess = spawn('node', cmd, { stdio: 'inherit' })
    processes.push(agentProcess)

    await new Promise(resolve => setTimeout(resolve, 1000))
}

process.on('SIGINT', () => {
    console.log('\n[INFO] Cerrando agentes...')
    processes.forEach(p => p.kill())
    process.exit(0)
})
