/**
 * Universal agent launcher via CLI
 * 
 * Usage:
 *   node src/agents/add_agent.js --name MyBot --type htn --minecraft-port 25565
 */

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { logInfo, logError } from './logging.js';

(async () => {
    try {
        // Parse command line arguments
        const args = yargs(hideBin(process.argv))
            .option(`name`, {
                alias: `n`,
                type: `string`,
                description: `Agent name`,
                default: `Agent_${Math.floor(Math.random() * 10000)}`
            })
            .option(`type`, {
                alias: `t`,
                type: `string`,
                description: `Agent type (htn, rl, il)`,
                default: `htn`,
                choices: [`htn`, `rl`, `il`]
            })
            .option(`minecraft-host`, {
                alias: `h`,
                type: `string`,
                description: `Minecraft server host`,
                default: `localhost`
            })
            .option(`minecraft-port`, {
                alias: `mp`,
                type: `number`,
                description: `Minecraft server port`,
                default: 25565
            })
            .option(`viewer-port`, {
                alias: `vp`,
                type: `number`,
                description: `Port for browser viewer`,
                default: 3000
            })
            .option(`inference-url`, {
                type: `string`,
                description: `URL del servidor de inferencia IL (sólo para --type il)`,
                default: `http://127.0.0.1:8765`
            })
            .option(`minecraft-version`, {
                type: `string`,
                description: `Minecraft version`,
                default: `auto`
            })
            .option(`record`, {
                type: `boolean`,
                description: `Grabar screenshots y log de inferencia (sólo --type il)`,
                default: false,
            })
            .help()
            .example(`node src/agents/add_agent.js --name MyBot --type htn`, `Launch HTN agent`)
            .example(`node src/agents/add_agent.js -n TestBot -t htn -mp 25565 -vp 3000`, `Launch with custom ports`)
            .parse();

        const agentName = args.name;
        const agentType = args.type;
        
        logInfo(agentName, `Initializing ${agentType.toUpperCase()} agent...`);

        // Build settings from CLI arguments
        const settings = buildAgentSettings(agentName, agentType, {
            minecraftHost: args[`minecraft-host`],
            minecraftPort: args[`minecraft-port`],
            minecraftVersion: args[`minecraft-version`],
            inferenceUrl: args[`inference-url`]
        });

        logInfo(agentName, `Settings: ${JSON.stringify(settings, null, 2)}`);

        // Dynamically import correct agent class
        let AgentClass;
        if (agentType === `htn`) {
            const { HTNAgent } = await import(`./types/htn_agent.js`);
            AgentClass = HTNAgent;
        } else if (agentType === `rl`) {
            const { RLAgent } = await import(`./types/rl_agent.js`);
            AgentClass = RLAgent;
        } else if (agentType === `il`) {
            const { ILAgent } = await import(`./types/il_agent.js`);
            AgentClass = ILAgent;
        } else if (agentType === `llm`) {
            // TODO: Implement LLMAgent and import here, its easy but not the focus right now
            logError(agentName, new Error(`LLM agent type not implemented yet`));
            process.exit(1); 
        }
        else {
            throw new Error(`Unknown agent type: ${agentType}`);
        }

        // Create and start agent
        logInfo(agentName, `Starting ${agentType.toUpperCase()} agent...`);
        const agent = (agentType === `il`)
            ? new AgentClass(agentName, undefined, args.record)
            : new AgentClass(agentName);

        await agent.start(settings, args[`viewer-port`]);

    } catch (error) {
        console.error(`[ERROR] Failed to start agent:`);
        console.error(error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();

/**
 * Build a standardized settings object for agent creation
 * 
 * @param {string} agentName - Agent name
 * @param {string} agentType - Agent type (htn, rl)
 * @param {Object} options - Configuration options
 * @returns {Object} Settings object ready for agent.start()
 */
function buildAgentSettings(agentName, agentType, options = {}) {
    return {
        username: agentName,
        host: options.minecraftHost || `localhost`,
        port: options.minecraftPort || 25565,
        auth: `offline`,
        version: options.minecraftVersion || `auto`,
        inference_url: options.inferenceUrl || `http://127.0.0.1:8765`,
        task: {
            goal: `Complete ${agentType.toUpperCase()} progression`
        }
    };
}