export function logInfo(agentName, message) {
    console.log(`[INFO] [${agentName}] ${message}`);
}

export function logError(agentName, error) {
    console.error(`[ERROR] [${agentName}] ${error.message}`);
}