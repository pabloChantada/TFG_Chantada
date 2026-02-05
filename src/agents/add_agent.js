/**
 * Add agents to a running server via Socket.IO
 * Usage: node src/agents/add_agent.js --name AgentName --type htn --port 8080
 */

import { io } from 'socket.io-client';
import { buildAgentSettings } from './create_agent.js';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const args = yargs(hideBin(process.argv))
    .option('name', {
        alias: 'n',
        type: 'string',
        description: 'Agent name',
        demandOption: true
    })
    .option('type', {
        alias: 't',
        type: 'string',
        description: 'Agent type (htn, rl, custom)',
        default: 'htn',
        choices: ['htn', 'rl', 'custom']
    })
    .option('port', {
        alias: 'p',
        type: 'number',
        description: 'Server port',
        default: 8080
    })
    .option('minecraft-host', {
        alias: 'mh',
        type: 'string',
        description: 'Minecraft server host',
        default: '127.0.0.1'
    })
    .option('minecraft-port', {
        alias: 'mp',
        type: 'number',
        description: 'Minecraft server port',
        default: 25565
    })
    .option('load-memory', {
        type: 'boolean',
        description: 'Load previous memory',
        default: false
    })
    .option('metrics-path', {
        type: 'string',
        description: 'Path to export metrics JSON',
        default: 'src/metrics/agent_metrics'
    })
    .help()
    .example('node src/agents/add_agent.js --name HTNTest --type htn', 'Add HTN agent to running server')
    .example('node src/agents/add_agent.js -n RL1 -t rl -p 8081', 'Add RL agent to server on port 8081')
    .parse();

const socket = io(`http://localhost:${args.port}`);

socket.on('connect', () => {
    console.log(`[INFO] Connected to server on port ${args.port}`);
    
    const settings = buildAgentSettings(args.name, args.type, {
        minecraftHost: args['minecraft-host'],
        minecraftPort: args['minecraft-port'],
        loadMemory: args['load-memory'],
        metricsPath: args['metrics-path']
    });
    
    console.log(`[INFO] Creating ${args.type} agent: ${args.name}...`);
    
    socket.emit('create-agent', settings, (response) => {
        if (response.success) {
            console.log(`[INFO] Agent ${args.name} created successfully`);
        } else {
            console.error(`[ERROR] Failed to create agent: ${response.error}`);
        }
        socket.disconnect();
        process.exit(response.success ? 0 : 1);
    });
});

socket.on('connect_error', (error) => {
    console.error(`[ERROR] Failed to connect to server on port ${args.port}`);
    console.error(error.message);
    process.exit(1);
});

// El bot no se desconecta, lo que hace add_agent.js es simplemente enviar la solicitud al servidor para que cree el agente.
// Pero el archivo no se encarga de gestionar la desconexión del socket.
// socket.on('disconnect', () => {
//    console.log('[INFO] Disconnected from server');
// });
