import { createMindServer, registerAgent, numStateListeners } from './mindserver.js';
import { AgentProcess } from '../llm/src/process/agent_process.js';
import { getServer } from './mcserver.js';
import open from 'open';

let mindserver;
let connected = false;
let agent_processes = {};
let agent_count = 0;
let mindserver_port = 8080;


/**
 * Initializes the MindCraft server and optionally opens the UI in the default browser.
 * @param {boolean} host_public - Whether to host the server publicly or on localhost.
 * @param {number} port - The port to run the MindCraft server on.
 * @param {boolean} auto_open_ui - Whether to automatically open the UI in the browser after initialization.
 */
export async function init(host_public=false, port=8080, auto_open_ui=true) {
    if (connected) {
        console.error('Already initialized!');
        return;
    }
    mindserver = createMindServer(host_public, port);
    mindserver_port = port;
    connected = true;
    const disableAutoOpen = process.env.MINDCRAFT_NO_UI === '1' || process.env.NO_UI === '1';
    
    if (auto_open_ui && !disableAutoOpen) {
        setTimeout(() => {
            // check if browser listener is already open
            if (numStateListeners() === 0) {
                open('http://localhost:'+port);
            }
        }, 3000);
    }
}

/**
 * Creates a new agent with the given settings and starts it.
 * @param {Object} settings - The settings for the agent, including profile information and connection details.
 * @returns {Object} An object indicating success or failure, and any error message if applicable.
 */
export async function createAgent(settings) {
    if (!settings.profile.name) {
        console.error('Agent name is required in profile');
        return {
            success: false,
            error: 'Agent name is required in profile'
        };
    }
    // Obtain data from settings, ensuring we don't modify the original object
    settings = JSON.parse(JSON.stringify(settings));
    let agent_name = settings.profile.name;
    const agentIndex = agent_count++;
    const viewer_port = 3000 + agentIndex;
    console.log(`Creating agent ${agent_name} with viewer port ${viewer_port}`);
    registerAgent(settings, viewer_port);
    let load_memory = settings.load_memory || false;
    let init_message = settings.init_message || null;

    try {
        try {
            const server = await getServer(settings.host, settings.port, settings.minecraft_version);
            settings.host = server.host;
            settings.port = server.port;
            settings.minecraft_version = server.version;
        } catch (error) {
            console.warn(`Error getting server:`, error);
            if (settings.minecraft_version === "auto") {
                settings.minecraft_version = null;
            }
            console.warn(`Attempting to connect anyway...`);
        }

        const script = 'src/agents/add_agent.js';

        console.log(`Starting ${settings.agent_type || 'custom'} agent with script: ${script}`);
        const agentProcess = new AgentProcess(agent_name, mindserver_port, script);
        agentProcess.start(load_memory, init_message, agentIndex);
        agent_processes[settings.profile.name] = agentProcess;
    } catch (error) {
        console.error(`Error creating agent ${agent_name}:`, error);
        destroyAgent(agent_name);
        return {
            success: false,
            error: error.message
        };
    }
    return {
        success: true,
        error: null
    };
}

export function getAgentProcess(agentName) {
    return agent_processes[agentName];
}

export function startAgent(agentName) {
    if (agent_processes[agentName]) {
        agent_processes[agentName].forceRestart();
    }
    else {
        console.error(`Cannot start agent ${agentName}; not found`);
    }
}

export function stopAgent(agentName) {
    if (agent_processes[agentName]) {
        agent_processes[agentName].stop();
    }
}

export function destroyAgent(agentName) {
    if (agent_processes[agentName]) {
        agent_processes[agentName].stop();
        delete agent_processes[agentName];
    }
}

export function shutdown() {
    console.log('Shutting down');
    for (let agentName in agent_processes) {
        agent_processes[agentName].stop();
    }
    setTimeout(() => {
        process.exit(0);
    }, 2000);
}
