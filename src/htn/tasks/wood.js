import { collectResource, ensureCraftingTable } from '../primitive_task.js'

// =========================================================
// --- TAREAS DE NIVEL MEDIO: MADERA/MESA ---
// =========================================================

async function chopTree(bot, mcData, logs = 4) {
    await collectResource(bot, mcData, 'oak_log', logs)
}

export {
    ensureCraftingTable,
    chopTree
}
