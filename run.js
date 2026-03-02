/**
 * Simple agent launcher - replaces server.js
 * 
 * Usage:
 *   Basic (uses defaults)
 *     node run.js
 *
 *   # Custom name and port
 *     node run.js --name Bot1 --type htn --viewer-port 3000
 *
 *   # Clean metrics before running
 *     node run.js -n Bot1 -t htn --metrics-clean
 *
 *   # RL agent
 *     node run.js --name Bot1 --type rl
 *   # Custom server
 *     node run.js --name Bot1 --type htn --minecraft-host example.com --minecraft-port 25566
 */

import { spawn } from 'child_process';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const args = yargs(hideBin(process.argv))
    .option('agents', {
        type: 'string',
        description: 'Comma-separated agent types (e.g., "htn,rl")',
        default: 'htn'
    })
    .option('names', {
        type: 'string',
        description: 'Comma-separated agent names (must match agents count)'
    })
    .option('base-port', {
        type: 'number',
        description: 'Base viewer port (each agent gets +1)',
        default: 3000
    })
    .option('minecraft-port', {
        type: 'number',
        description: 'Minecraft server port',
        default: 25565
    })
    .option('metrics-path', {
        type: 'string',
        description: 'Metrics export path',
        default: './src/metrics/agent_metrics/${agentName}_metrics.json'
    })
    .option('metrics-clean', {
        type: 'boolean',
        description: 'Clean metrics directory before running',
        default: false
    })
    .option('memory-clean', {
        type: 'boolean',
        description: 'Clean agent memory before running',
        default: true
    })
    .help()
    .example('node run.js --agents htn --names Bot1', 'Launch single HTN agent')
    .example('node run.js --agents htn,htn --names Bot1,Bot2', 'Launch 2 HTN agents')
    .example('node run.js -n Bot1 -t htn --metrics-clean', 'Launch HTN agent with clean metrics')
    .parse();

const agentTypes = args.agents.split(',').map(s => s.trim());
const agentNames = args.names 
    ? args.names.split(',').map(s => s.trim())
    : agentTypes.map((_, i) => `Agent${i + 1}`);

if (agentTypes.length !== agentNames.length) {
    console.error('Error: Number of agent types must match number of names');
    process.exit(1);
}

console.log(`[INFO] Launching ${agentTypes.length} agent(s)...\n`);

const processes = [];

for (let i = 0; i < agentTypes.length; i++) {
    const agentType = agentTypes[i];
    const agentName = agentNames[i];
    const viewerPort = args['base-port'] + i;

    const cmd = [
        'src/agents/add_agent.js',
        '--name', agentName,
        '--type', agentType,
        '--minecraft-port', String(args['minecraft-port']),
        '--viewer-port', String(viewerPort),
        '--metrics-path', args['metrics-path'].replace('${agentName}', agentName),
        
    ];

    console.log(`[INFO] Agent ${i + 1}/${agentTypes.length}: ${agentName} (${agentType}) on port ${viewerPort}`);
    
    const agentProcess = spawn('node', cmd, {
        stdio: 'inherit'
    });

    processes.push(agentProcess);

    // Stagger agent startup
    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n[INFO] Shutting down all agents...');
    processes.forEach(p => p.kill());
    process.exit(0);
});