import settings from '../settings.js';
import prismarineViewer from 'prismarine-viewer';
const mineflayerViewer = prismarineViewer.mineflayer;

export function addBrowserViewer(bot, port, viewDistance = 10, firstPerson = true) {
    if (settings.render_bot_view)
        mineflayerViewer(bot, { port: port, firstPerson: firstPerson, viewDistance: viewDistance});
}
