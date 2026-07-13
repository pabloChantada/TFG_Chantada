/**
 * Regenera el mundo del servidor Paper y lo deja corriendo.
 *
 * Hace lo mismo que mass_record por episodio, pero de forma manual:
 *   1. Borra world / world_nether / world_the_end
 *   2. Reinstala el datapack de bioma único (bosque)  ← clave: vive dentro de world/
 *   3. Reescribe server.properties con la seed elegida
 *   4. Arranca el servidor y espera a que esté listo
 *
 * Uso (PowerShell):
 *   node scripts/reset_world.js                 # mundo de EVALUACIÓN (seed.txt), bosque, limpio
 *   node scripts/reset_world.js --random        # mundo nuevo aleatorio (bosque)
 *   node scripts/reset_world.js --no-datapack   # mundo NORMAL de la seed (sin forzar bioma)
 *   node scripts/reset_world.js --port 25566    # otro puerto
 *
 * Deja el servidor en primer plano. Ctrl+C lo para limpiamente.
 * NOTA: no debe haber otro servidor ocupando el puerto (cierra el .bat antes).
 */
import path from 'path'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import { startServer, stopServer, validateSetup } from './paper_server.js'

const args = yargs(hideBin(process.argv))
    .option('server-dir', { type: 'string', default: 'server', description: 'Directorio del servidor Paper' })
    .option('port',       { type: 'number', default: 25565, description: 'Puerto del servidor' })
    .option('random',     { type: 'boolean', default: false, description: 'Seed aleatoria (mundo nuevo) en vez de la de seed.txt' })
    .option('datapack',   { type: 'boolean', default: true,  description: 'Forzar bioma bosque con datapack (--no-datapack para mundo normal)' })
    .help()
    .parse()

const serverDir = path.resolve(args['server-dir'])

let serverProcess = null
async function shutdown() {
    if (serverProcess) {
        console.log('\n[RESET] Parando servidor...')
        await stopServer(serverProcess)
    }
    process.exit(0)
}
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)

;(async () => {
    await validateSetup(serverDir)
    const { seed } = await startServer(serverDir, args.port, {
        useSeed: !args.random,
        singleBiome: args.datapack,
    })
    console.log(`\n[RESET] Mundo regenerado y servidor listo (seed=${seed}, bosque=${args.datapack}).`)
    console.log('[RESET] Lanza ahora el agente en otra terminal. Ctrl+C aquí para parar el servidor.\n')
})().catch((err) => {
    console.error(`[RESET] Error: ${err.message}`)
    process.exit(1)
})
