/**
 * Generic agent launcher - used by AgentProcess
 * Dynamically loads the correct agent class based on settings
 * 
 * This is invoked by AgentProcess with settings from server
 * Usado por server.js y mindcraft.js (servidor web) para añadir los agentes al server
 * Siempre podriamos eliminar este archivo y usar add_agent.js directamente, 
 * pero de esta forma mantenemos el servidor web y los agentes "separados".
 */

import { serverProxy } from '../llm/src/agent/mindserver_proxy.js';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

(async () => {
    try {
        // Parse command line arguments
        const args = yargs(hideBin(process.argv))
            .option('n', {
                alias: 'name',
                type: 'string',
                description: 'Agent name'
            })
            .option('p', {
                alias: 'port',
                type: 'number',
                description: 'Server port',
                default: 8080
            })
            .option('c', {
                alias: 'count',
                type: 'number',
                description: 'Agent index',
                default: 0
            })
            .option('l', {
                alias: 'load-memory',
                type: 'boolean',
                description: 'Load memory'
            })
            .option('m', {
                alias: 'message',
                type: 'string',
                description: 'Init message'
            })
            .argv;

        // Get settings from server via mindserver_proxy
        const agentName = args.n || args.name || process.env.AGENT_NAME || 'Agent';
        const serverPort = args.p || args.port || parseInt(process.env.MINDSERVER_PORT || '8080');
        const agentIndex = args.c || args.count || parseInt(process.env.AGENT_INDEX || '0');
        
        console.log(`[${agentName}] Connecting to MindServer on port ${serverPort}...`);
        await serverProxy.connect(agentName, serverPort);
        
        // Request settings from server
        const settings = await new Promise((resolve, reject) => {
            serverProxy.socket.emit('get-settings', agentName, (response) => {
                if (response.error) {
                    reject(new Error(response.error));
                } else {
                    resolve(response.settings);
                }
            });
        });
        
        
        const agentType = settings.agent_type || settings.profile?.agent_type || 'htn';
        console.log(`[${agentName}] Loading ${agentType} agent...`);
        
        // Dynamically import correct agent class
        let AgentClass;
        if (agentType === 'htn') {
            const module = await import('./htn_agent.js');
            AgentClass = module.HTNAgent;
        } else if (agentType === 'rl') {
            const module = await import('./rl_agent.js');
            AgentClass = module.RLAgent;
        } else {
            throw new Error(`Unknown agent type: ${agentType}`);
        }
        
        // Create and start agent
        const agent = new AgentClass(agentName);
        serverProxy.setAgent(agent);
        
        const viewerPort = 3000 + agentIndex;
        const loadMemory = args.l || args.loadMemory || settings.load_memory || false;
        const initMessage = args.m || args.message || settings.init_message || null;
        
        await agent.start(
            settings,
            viewerPort,
            agentIndex,
            loadMemory,
            initMessage
        );
        
    } catch (error) {
        console.error('[ERROR] Failed to start agent:');
        console.error(error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();
