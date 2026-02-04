/**
 * Main entry point for the Evaluation Server
 * Launch server and optionally create agents from command line
 * 
 * Usage:
 *   node src/server/server.js                                    # Launch server only
 *   node src/server/server.js --agent htn --name Agent1          # Launch server + HTN agent
 *   node src/server/server.js --agent htn --name Agent1 --no-ui         # Launch server + HTN agent
 *   node src/server/server.js --agents htn,rl --names A1,A2      # Launch server + multiple agents
 */

import * as EvalServer from './mindcraft.js';
import { buildAgentSettings, parseAgentsConfig } from '../agents/create_agent.js';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const args = yargs(hideBin(process.argv))
    .option('port', {
        alias: 'p',
        type: 'number',
        description: 'Server port',
        default: 8080
    })
    .option('host-public', {
        type: 'boolean',
        description: 'Make server accessible from external IPs (0.0.0.0)',
        default: false
    })
    .option('no-ui', {
        type: 'boolean',
        description: 'Don\'t auto-open browser UI',
        default: false
    })
    .option('no-viewer', {
        type: 'boolean',
        description: 'Don\'t open 3D bot viewer',
        default: false
    })
    .option('agent', {
        alias: 'a',
        type: 'string',
        description: 'Agent type to create on startup (htn, rl, custom)',
        choices: ['htn', 'rl', 'custom']
    })
    .option('name', {
        alias: 'n',
        type: 'string',
        description: 'Agent name (required if --agent is used)',
    })
    .option('agents', {
        type: 'string',
        description: 'Comma-separated list of agent types (e.g., "htn,rl,htn")'
    })
    .option('names', {
        type: 'string',
        description: 'Comma-separated list of agent names (must match --agents count)'
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
    .option('minecraft-version', {
        alias: 'mv',
        type: 'string',
        description: 'Minecraft version (or "auto")',
        default: 'auto'
    })
    .option('clean-metrics', {
        type: 'boolean',
        description: 'Clean metrics directory (excluding example_*.json/csv files)',
        default: false
    })
    .option('config', {
        alias: 'c',
        type: 'string',
        description: 'Path to JSON config file with agent definitions'
    })
    .help()
    .alias('help', 'h')
    .example('node server.js', 'Launch server only')
    .example('node server.js --agent htn --name MyHTN', 'Launch server + HTN agent')
    .example('node server.js --agents htn,rl --names Agent1,Agent2', 'Launch server + 2 agents')
    .example('node server.js --config agents.json', 'Launch server with agents from config file')
    .parse();

const disableUi = args.noUi === true || args.ui === false;
if (disableUi) {
    process.env.MINDCRAFT_NO_UI = '1';
    console.log('[INFO] UI auto-open disabled via --no-ui flag');
}

// Initialize server
console.log('[INFO] Starting Evaluation Server...');
// Clean memory directory if it exists and is not empty
const memoryPath = "src/agents/memories/";
try {
    const fs = await import('fs');
    if (fs.existsSync(memoryPath)) {
        const files = fs.readdirSync(memoryPath);
        if (files.length > 0) {
            console.log(`[INFO] Cleaning memory directory: ${memoryPath}`);
            for (const file of files) {
                const filePath = `${memoryPath}/${file}`;
                if (fs.statSync(filePath).isFile()) {
                    fs.unlinkSync(filePath);
                }
            }
            console.log(`[INFO] Removed ${files.length} file(s) from memory directory`);
        }
    } else {
        // Create directory if it doesn't exist
        fs.mkdirSync(memoryPath, { recursive: true });
        console.log(`[INFO] Created memory directory: ${memoryPath}`);
    }
} catch (error) {
    console.error('[ERROR] Failed to clean memory directory:', error.message);
}

// Clean metrics directory if requested
if (args.cleanMetrics) {
    const metricsPath = "src/metrics/agent_metrics/";
    try {
        const fs = await import('fs');
        if (fs.existsSync(metricsPath)) {
            const files = fs.readdirSync(metricsPath);
            let deletedCount = 0;
            
            for (const file of files) {
                // Skip example files
                if (file.startsWith('example_')) {
                    continue;
                }
                
                const filePath = `${metricsPath}/${file}`;
                if (fs.statSync(filePath).isFile()) {
                    fs.unlinkSync(filePath);
                    deletedCount++;
                }
            }
            
            if (deletedCount > 0) {
                console.log(`[INFO] Removed ${deletedCount} file(s) from metrics directory`);
            }
        } else {
            // Create directory if it doesn't exist
            fs.mkdirSync(metricsPath, { recursive: true });
            console.log(`[INFO] Created metrics directory: ${metricsPath}`);
        }
    } catch (error) {
        console.error('[ERROR] Failed to clean metrics directory:', error.message);
    }
}

await EvalServer.init(args.hostPublic, args.port, !disableUi);

// Create agents from CLI arguments
if (args.agent && args.name) {
    console.log(`[INFO] Creating ${args.agent} agent: ${args.name}`);
    await createAgentFromCLI(args.agent, args.name, args);
}
else if (args.agents && args.names) {
    const agentTypes = args.agents.split(',').map(s => s.trim());
    const agentNames = args.names.split(',').map(s => s.trim());
    
    if (agentTypes.length !== agentNames.length) {
        console.error('Error: Number of agent types must match number of names');
        process.exit(1);
    }
    
    for (let i = 0; i < agentTypes.length; i++) {
        console.log(`Creating ${agentTypes[i]} agent: ${agentNames[i]}`);
        await createAgentFromCLI(agentTypes[i], agentNames[i], args);
        // Small delay between agents to avoid race conditions
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}
else if (args.config) {
    console.log(`Loading agents from config: ${args.config}`);
    await createAgentsFromConfig(args.config, args);
}

console.log(`\n[INFO] Evaluation Server running on http://localhost:${args.port}`);
console.log('Press Ctrl+C to shutdown\n');

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n[INFO] Shutting down...');
    EvalServer.shutdown();
});

/**
 * Create a single agent from CLI arguments
 */
async function createAgentFromCLI(agentType, agentName, cliArgs) {
    const settings = buildAgentSettings(agentName, agentType, {
        minecraftHost: cliArgs.minecraftHost,
        minecraftPort: cliArgs.minecraftPort,
        minecraftVersion: cliArgs.minecraftVersion,
        noViewer: cliArgs.noViewer
    });

    const result = await EvalServer.createAgent(settings);
    
    if (result.success) {
        console.log(`[INFO] Agent ${agentName} created successfully`);
    } else {
        console.error(`[ERROR] Failed to create agent ${agentName}: ${result.error}`);
    }
}

/**
 * Create multiple agents from a JSON config file
 */
async function createAgentsFromConfig(configPath, cliArgs) {
    try {
        const defaultOptions = {
            minecraftHost: cliArgs.minecraftHost,
            minecraftPort: cliArgs.minecraftPort,
            minecraftVersion: cliArgs.minecraftVersion,
            noViewer: cliArgs.noViewer
        };
        
        const agentSettingsList = await parseAgentsConfig(configPath, defaultOptions);
        
        for (const settings of agentSettingsList) {
            console.log(`[INFO] Creating ${settings.agent_type} agent: ${settings.profile.name}`);
            const result = await EvalServer.createAgent(settings);
            
            if (result.success) {
                console.log(`[INFO] Agent ${settings.profile.name} created successfully`);
            } else {
                console.error(`[ERROR] Failed to create agent ${settings.profile.name}: ${result.error}`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    } catch (error) {
        console.error('[ERROR] Error loading config file:', error.message);
        process.exit(1);
    }
}
