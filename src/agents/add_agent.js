/**
 * Universal agent launcher via CLI
 * Can be used both as standalone CLI and spawned by AgentProcess
 * 
 * Usage (Direct):
 *   node src/agents/add_agent.js --name AgentName --type htn --port 8080
 * 
 * Usage (Via AgentProcess):
 *   spawned with settings from mindserver
 */

import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import { setSettings } from '../llm/src/agent/settings.js';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

(async () => {
    try {
        // Parse command line arguments
        const args = yargs(hideBin(process.argv))
            .option(`name`, {
                alias: `n`,
                type: `string`,
                description: `Agent name`,
                default: `Agent_${Math.floor(Math.random() * 10)}`
            })
            .option(`type`, {
                alias: `t`,
                type: `string`,
                description: `Agent type (htn, rl, llm)`,
                default: `htn`,
                choices: [`htn`, `rl`, `llm`]
            })
            .option(`port`, {
                alias: `p`,
                type: `number`,
                description: `MindServer port`,
                default: 8080
            })
            .option(`minecraft-port`, {
                alias: `mp`,
                type: `number`,
                description: `Minecraft server port`,
                default: 25565
            })
            .option(`count`, {
                alias: `c`,
                type: `number`,
                description: `Agent index for multi-agent scenarios`,
                default: 0
            })
            .option(`load-memory`, {
                alias: `l`,
                type: `boolean`,
                description: `Load previous memory`,
                default: false
            })
            .option(`init-message`, {
                alias: `m`,
                type: `string`,
                description: `Initial message for agent`,
                default: null
            })
            .option(`metrics-path`, {
                type: `string`,
                description: `Path to export metrics JSON`,
                default: `src/metrics/agent_metrics`
            })
            .help()
            .example(`node src/agents/add_agent.js --name HTNTest --type htn`, `Add HTN agent`)
            .example(`node src/agents/add_agent.js -n Agent1 -t llm -c 0`, `Add LLM agent with index 0`)
            .parse();

        const agentName = args.name;
        const agentType = args.type;
        const serverPort = args.port;
        const agentIndex = args.count;
        const viewerPort = 3000 + agentIndex;
        
        console.log(`[${agentName}] Connecting to MindServer on port ${serverPort}...`);
        await serverProxy.connect(agentName, serverPort, { skipSettings: true, allowMissingSettings: true });
        
        // Request settings from server (if running via AgentProcess)
        // Otherwise, build settings from CLI args
        let settings;
        try {
            settings = await requestSettingsFromServer(agentName);
        } catch (e) {
            // Server not responding, build settings from CLI args
            console.log(`[${agentName}] Building settings from CLI arguments...`);
            settings = buildAgentSettings(agentName, agentType, {
                minecraftPort: args[`minecraft-port`],
                loadMemory: args[`load-memory`],
                metricsPath: args[`metrics-path`],
                initMessage: args[`init-message`]
            });
            await registerAgentOnServer(agentName, settings, viewerPort);
        }

        setSettings(settings);
        serverProxy.getSocket().emit('connect-agent-process', agentName);

        const finalAgentType = settings.agent_type || settings.profile?.agent_type || agentType;
        console.log(`[${agentName}] Loading ${finalAgentType} agent...`);
        
        // Dynamically import correct agent class
        let AgentClass;
        if (finalAgentType === `htn`) {
            const module = await import(`./htn_agent.js`);
            AgentClass = module.HTNAgent;
        } else if (finalAgentType === `llm`) {
            // For LLM agents, use the Mindcraft Agent
            const module = await import(`../llm/src/agent/agent.js`);
            AgentClass = module.Agent;
        } else {
            throw new Error(`Unknown agent type: ${finalAgentType}`);
        }
        
        // Create and start agent
        const agent = new AgentClass(agentName);
        serverProxy.setAgent(agent);
        
        const loadMemory = args[`load-memory`] || settings.load_memory || false;
        const initMessage = args[`init-message`] || settings.init_message || null;
        
        await agent.start(
            settings,
            viewerPort,
            loadMemory,
            initMessage
        );
        
    } catch (error) {
        console.error(`[ERROR] Failed to start agent:`);
        console.error(error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();

function requestSettingsFromServer(agentName) {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error('Timeout waiting for settings'));
        }, 2000);

        serverProxy.getSocket().emit('get-settings', agentName, (response) => {
            clearTimeout(timeout);
            if (response?.error) {
                reject(new Error(response.error));
                return;
            }
            resolve(response.settings);
        });
    });
}

function registerAgentOnServer(agentName, settings, viewerPort) {
    return new Promise((resolve, reject) => {
        serverProxy.getSocket().emit('register-agent', settings, viewerPort, (response) => {
            if (!response?.success) {
                reject(new Error(response?.error || `Failed to register ${agentName}`));
                return;
            }
            resolve();
        });
    });
}

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
        host: options.minecraftHost || options.host || `127.0.0.1`,
        port: options.minecraftPort || options.port || 25565,
        auth: options.auth || `offline`,
        minecraft_version: options.minecraftVersion || options.minecraft_version || `auto`,
        load_memory: options.loadMemory || options.load_memory || false,
        init_message: options.initMessage || options.init_message || null,
        render_bot_view: options.noViewer ? false : (options.renderBotView !== false && options.render_bot_view !== false),
        spawn_timeout: options.spawnTimeout || options.spawn_timeout || 30,
        metrics_enabled: options.metricsEnabled !== false && options.metrics_enabled !== false,
        metrics_export_path: options.metricsPath || options.metrics_export_path || null,
        task: options.task || null
    };
}