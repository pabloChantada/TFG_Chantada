// Fábrica de bots de Mineflayer para los agentes (HTN / IL / RL).
//
// Crea una instancia de bot conectada al servidor de Minecraft y le carga los
// plugins que necesitan los agentes: pathfinder (navegación), pvp (combate),
// collectblock (recolección), auto-eat (comida) y armor-manager (equipar armadura).
//
// Extraído del módulo LLM vendorizado (mindcraft) para dejar el core del proyecto
// autocontenido. Solo se conserva `initBot`, que es lo único que consumían los agentes.
import { createBot } from 'mineflayer';
import { pathfinder } from 'mineflayer-pathfinder';
import { plugin as pvp } from 'mineflayer-pvp';
import { plugin as collectblock } from 'mineflayer-collectblock';
import armorManager from 'mineflayer-armor-manager';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

/**
 * Crea y configura un bot de Mineflayer.
 *
 * @param {string} username - Nombre del bot en el servidor.
 * @param {Object} [settings={}] - Configuración de conexión.
 * @param {string} [settings.host='localhost'] - Host del servidor.
 * @param {number} [settings.port=25565] - Puerto del servidor.
 * @param {string} [settings.auth='offline'] - Modo de autenticación.
 * @param {string} [settings.version='auto'] - Versión de Minecraft ('auto' la detecta).
 * @returns {import('mineflayer').Bot} El bot ya con los plugins cargados.
 */
export function initBot(username, settings = {}) {
    const options = {
        username,
        host: settings.host || 'localhost',
        port: settings.port || 25565,
        auth: settings.auth || 'offline',
        version: settings.version,
    };

    // 'auto' (o sin valor) → dejar que Mineflayer detecte la versión del servidor.
    if (!options.version || options.version === 'auto') {
        delete options.version;
    }

    const bot = createBot(options);
    bot.loadPlugin(pathfinder);
    bot.loadPlugin(pvp);
    bot.loadPlugin(collectblock);

    // auto-eat mezcla CJS/ESM según versión; se carga dinámicamente para evitar
    // fallos de import.
    try {
        const autoEatModule = require('mineflayer-auto-eat');
        const autoEat = autoEatModule.plugin || autoEatModule.default || autoEatModule;
        if (typeof autoEat === 'function') {
            bot.loadPlugin(autoEat);
        }
    } catch (e) {
        console.warn('Could not load mineflayer-auto-eat plugin:', e.message);
    }

    bot.loadPlugin(armorManager); // equipar armadura automáticamente

    bot.once('resourcePack', () => {
        bot.acceptResourcePack();
    });

    return bot;
}
