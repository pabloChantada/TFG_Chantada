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
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const args = yargs(hideBin(process.argv))
    .option('port', {
        alias: 'p',
        type: 'number',
        description: 'Server port',
        default: 8080
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
    .option('minecraft-port', {
        alias: 'mp',
        type: 'number',
        description: 'Minecraft server port',
        default: 25565
    })
    .option('clean-metrics', {
        alias: 'cm',
        type: 'boolean',
        description: 'Clean metrics directory',
        default: false
    })
    .help()
    .alias('help', 'h')
    .example('node server.js', 'Launch server only')
    .example('node server.js --agent htn --name MyHTN', 'Launch server + HTN agent')
    .example('node server.js --agents htn,rl --names Agent1,Agent2', 'Launch server + 2 agents')
    .example('node server.js --config agents.json', 'Launch server with agents from config file')
    .parse();

// Configure UI based on CLI arguments and environment variables
const disableUi = configureUi(args);

// Remove old memory files on startup to prevent interference with new agents
console.log('[INFO] Starting Evaluation Server...');
await cleanMemoryDirectory();
await cleanMetricsDirectory(args.cleanMetrics);

// Start the MindCraft server and create agents based on CLI arguments
await EvalServer.init(args.hostPublic, args.port, !disableUi);
await createAgentsFromArgs(args);

logServerReady(args.port);
registerShutdownHandler();

/**
 * Build a standardized settings object for agent creation
 * 
 * @param {string} agentName - Agent name
 * @param {string} agentType - Agent type (htn, rl, llm)
 * @param {Object} options - Additional configuration options
 * @returns {Object} Settings object ready for agent.start()
 */
function buildAgentSettings(agentName, agentType, options = {}) {
    return {
        profile: {
            name: agentName,
            agent_type: agentType
        },
        host: options.minecraftHost || options.host || '127.0.0.1',
        port: options.minecraftPort || options.port || 25565,
        auth: options.auth || 'offline',
        minecraft_version: options.minecraftVersion || options.minecraft_version || 'auto',
        load_memory: options.loadMemory || options.load_memory || false,
        init_message: options.initMessage || options.init_message || null,
        render_bot_view: options.noViewer ? false : (options.renderBotView !== false && options.render_bot_view !== false),
        spawn_timeout: options.spawnTimeout || options.spawn_timeout || 30,
        metrics_enabled: options.metricsEnabled !== false && options.metrics_enabled !== false,
        metrics_export_path: options.metricsPath || options.metrics_export_path || null,
        mindserver_port: options.mindserverPort || options.mindserver_port || 8080,
        task: options.task || null
    };
}

/**
 * Create a single agent from CLI arguments
 * @param {string} agentType - The type of agent to create (htn, rl, custom)
 * @param {string} agentName - The name of the agent to create
 * @param {Object} cliArgs - The original CLI arguments for additional configuration
 */
async function createAgentFromCLI(agentType, agentName, cliArgs) {
    const settings = buildAgentSettings(agentName, agentType, {
        minecraftHost: cliArgs.minecraftHost,
        minecraftPort: cliArgs.minecraftPort,
        minecraftVersion: cliArgs.minecraftVersion,
        noViewer: cliArgs.noViewer,
        mindserverPort: cliArgs.port,
        metricsPath: `src/metrics/agent_metrics/${agentName}_metrics.json`
    });

    try {
        const result = await EvalServer.createAgent(settings);
        if (!result.success) {
            console.error(`[ERROR] Failed to create agent ${agentName}: ${result.error}`);
            return;
        }
        console.log(`[INFO] Agent ${agentName} created successfully`);
    } catch (error) {
        console.error(`[ERROR] Failed to create agent ${agentName}: ${error.message}`);
    }
}


/**
 * Creates a new agent with the given settings and starts it.
 * @param {Object} settings - The settings for the agent, including profile information and connection details.
 * @returns {Object} An object indicating success or failure, and any error message if applicable.
 */
function configureUi(cliArgs) {
    const disableUi = cliArgs.noUi === true || cliArgs.ui === false;
    if (disableUi) {
        process.env.MINDCRAFT_NO_UI = '1';
        console.log('[INFO] UI auto-open disabled via --no-ui flag');
    }
    return disableUi;
}

/**
 * Cleans the memory directory by removing all files to prevent interference with new agents. If the directory does not exist, it is created.
 */
async function cleanMemoryDirectory(shouldClean = true) {
    if (!shouldClean) {
        return;
    }
    const memoryPath = 'src/agents/memories/';
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
            return;
        }
        fs.mkdirSync(memoryPath, { recursive: true });
        console.log(`[INFO] Created memory directory: ${memoryPath}`);
    } catch (error) {
        console.error('[ERROR] Failed to clean memory directory:', error.message);
    }
}

/**
 * Cleans the metrics directory by removing all files and subdirectories. If the directory does not exist, it is created.
 * @param {boolean} shouldClean - Whether to perform the cleaning operation. If false, the function will return without doing anything.
 */
async function cleanMetricsDirectory(shouldClean = true) {
    if (!shouldClean) {
        return;
    }

    const metricsPath = 'src/metrics/agent_metrics/';
    try {
        const fs = await import('fs');
        if (fs.existsSync(metricsPath)) {
            // Remove entire directory recursively and recreate it
            fs.rmSync(metricsPath, { recursive: true, force: true });
            console.log(`[INFO] Removed metrics directory and all contents`);
        }
        fs.mkdirSync(metricsPath, { recursive: true });
        console.log(`[INFO] Created clean metrics directory: ${metricsPath}`);
    } catch (error) {
        console.error('[ERROR] Failed to clean metrics directory:', error.message);
    }
}

/**
 * Creates agents based on CLI arguments. Supports creating a single agent with --agent and --name, multiple agents with --agents and --names, or loading agents from a config file with --config.
 * @param {Object} cliArgs - The parsed CLI arguments from yargs.
 * @returns {Promise<void>} A promise that resolves when all agents have been created.
 */
async function createAgentsFromArgs(cliArgs) {
    if (cliArgs.agent && cliArgs.name) {
        console.log(`[INFO] Creating ${cliArgs.agent} agent: ${cliArgs.name}`);
        await createAgentFromCLI(cliArgs.agent, cliArgs.name, cliArgs);
        return;
    }

    if (cliArgs.agents && cliArgs.names) {
        const agentTypes = cliArgs.agents.split(',').map(s => s.trim());
        const agentNames = cliArgs.names.split(',').map(s => s.trim());

        if (agentTypes.length !== agentNames.length) {
            console.error('Error: Number of agent types must match number of names');
            process.exit(1);
        }

        for (let i = 0; i < agentTypes.length; i++) {
            console.log(`Creating ${agentTypes[i]} agent: ${agentNames[i]}`);
            await createAgentFromCLI(agentTypes[i], agentNames[i], cliArgs);
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        return;
    }

    if (cliArgs.config) {
        console.log(`Loading agents from config: ${cliArgs.config}`);
        await createAgentsFromConfig(cliArgs.config, cliArgs);
    }
}

/**
 * Logs the server URL and shutdown instructions to the console when the server is ready.
 */
function logServerReady(port) {
    console.log(`\n[INFO] Evaluation Server running on http://localhost:${port}`);
    console.log('Press Ctrl+C to shutdown\n');
}

/**
 * Registers a shutdown handler to gracefully shut down the server when the process receives a SIGINT signal (e.g., when the user presses Ctrl+C).
 */
function registerShutdownHandler() {
    process.on('SIGINT', () => {
        console.log('\n[INFO] Shutting down...');
        EvalServer.shutdown();
    });
}