/**
 * Shared utility to create agent settings objects
 * Used by both server.js and add_agent.js to avoid duplication
 */

/**
 * Build a standardized settings object for agent creation
 * 
 * @param {string} agentName - Agent name
 * @param {string} agentType - Agent type (htn, rl, custom)
 * @param {Object} options - Additional configuration options
 * @returns {Object} Settings object ready for createAgent()
 */
export function buildAgentSettings(agentName, agentType, options = {}) {
    return {
        profile: {
            name: agentName,
            agent_type: agentType
        },
        agent_type: agentType,
        host: options.minecraftHost || options.host || '127.0.0.1',
        port: options.minecraftPort || options.port || 25565,
        minecraft_version: options.minecraftVersion || options.minecraft_version || 'auto',
        auth: options.auth || 'offline',
        load_memory: options.loadMemory || options.load_memory || false,
        init_message: options.initMessage || options.init_message || null,
        render_bot_view: options.noViewer ? false : (options.renderBotView !== false && options.render_bot_view !== false),
        chat_ingame: options.chatIngame !== false && options.chat_ingame !== false,
        narrate_behavior: options.narrateBehavior !== false && options.narrate_behavior !== false,
        spawn_timeout: options.spawnTimeout || options.spawn_timeout || 30,
        metrics_enabled: options.metricsEnabled !== false && options.metrics_enabled !== false,
        metrics_export_path: options.metricsPath || options.metrics_export_path || null,
        task: options.task || null
    };
}

/**
 * Parse agents configuration from JSON file
 * 
 * @param {string} configPath - Path to config JSON file
 * @param {Object} defaultOptions - Default options to merge with config
 * @returns {Array} Array of agent settings objects
 */
export async function parseAgentsConfig(configPath, defaultOptions = {}) {
    const fs = await import('fs');
    const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    
    if (!configData.agents || !Array.isArray(configData.agents)) {
        throw new Error('Config file must have "agents" array');
    }
    
    return configData.agents.map(agentConfig => {
        if (!agentConfig.name || !agentConfig.type) {
            throw new Error('Each agent must have "name" and "type"');
        }
        
        // Merge config with defaults
        const options = {
            ...defaultOptions,
            ...agentConfig
        };
        
        return buildAgentSettings(agentConfig.name, agentConfig.type, options);
    });
}
