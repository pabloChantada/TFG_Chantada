import { initBot } from '../src/utils/mcdata.js';
import { addBrowserViewer } from '../src/agent/vision/browser_viewer.js';
import { serverProxy } from '../src/agent/mindserver_proxy.js';
import settings from '../src/agent/settings.js';
import { startHTN } from '../../htn/main_htn.js';

/**
 * CustomAgent class implementing specific bot logic. Here we define
 * how the agent connects to the Minecraft server and its behavior.
 * We can either use HTN, RL, or custom scripted logic.
 * 
 * Basically we need to refactor de files in src/htn.
 */
export class CustomAgent {
    async start(load_mem=false, init_message=null, count_id=0, nameOverride=null) {
        // Si no se carga la memoria, reiniciamos el archivo memory.json
        if (!load_mem) {
            try {
                const fs = await import('fs');
                fs.writeFileSync('my_agent/memory.json', JSON.stringify({}));
                console.log('Memoria (memory.json) reiniciada.');
            } catch (e) {
                console.error('Error limpiando memoria:', e);
            }
        }

        this.name = nameOverride || settings.profile.name;
        settings.profile.name = this.name;
        this.count_id = count_id;
        
        // Mock components required by getFullState
        this.actions = { currentActionLabel: 'Custom Logic' };
        this.modes = { getMiniDocs: () => 'Custom Mode' };
        
        console.log(this.name, 'logging into minecraft...');
        console.log('Settings used for connection:', JSON.stringify(settings, null, 2));
        this.bot = initBot(this.name);
        this.bot.modes = { getMiniDocs: () => 'Custom Mode' };

        this.bot.on('login', () => {
            console.log(this.name, 'logged in!');
            serverProxy.login();
        });

        this.bot.on('error', (err) => {
            console.error('Bot error:', err);
        });

        this.bot.on('kicked', (reason) => {
            console.error('Bot kicked:', reason);
        });

        this.bot.once('spawn', async () => {
            addBrowserViewer(this.bot, count_id);
            console.log(`${this.name} spawned.`);
            
            // CUSTOM LOGIC
            this.runCustomLogic();
        });
    }

    runCustomLogic() {
        console.log("Custom logic started!");
        startHTN(this.bot);
    }

    isIdle() {
        return false;
    }

    cleanKill() {
        if (this.bot) this.bot.quit();
    }

    respondFunc(from, message) {
        console.log(`Received message from ${from}: ${message}`);
        if (this.bot) this.bot.chat(`I am a custom bot. I received: ${message}`);
    }
}
