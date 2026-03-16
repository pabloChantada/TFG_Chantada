/**
 * Bot control state interception and monitoring
 */

const PRE_ACTION_SCREENSHOT_WAIT_MS = 180

export class ControlInterceptor {
    constructor(bot, controlStates, recordCallback, rlActionCallback = null, metricsCollector = null) {
        this.bot = bot
        this.controlStates = controlStates
        this.recordCallback = recordCallback
        this.rlActionCallback = rlActionCallback
        this.metrics = metricsCollector

        this.originalSetControlState = null
        this.originalPlaceBlock = null
        this.originalActivateItem = null
        this.originalActivateBlock = null
        this.originalEquip = null
        this.originalAttack = null
        this.originalLook = null
        this.originalDig = null
        this.originalCraft = null
        this.originalOpenFurnace = null

        this.monitoringInterval = null
        this.isEnabled = false
        this.lastWindowType = null
    }

    /**
     * Wait briefly before executing an action so the recording loop can capture
     * a representative screenshot of the intended action state.
     */
    async preActionScreenshotWait() {
        if (!this.metrics?.captureScreenshots?.()) return
        await new Promise(resolve => setTimeout(resolve, PRE_ACTION_SCREENSHOT_WAIT_MS))
    }

    /**
     * Intercept bot.setControlState() calls and key high-level actions.
     */
    setupControlInterception() {
        if (!this.bot?.setControlState) return

        this.originalSetControlState = this.bot.setControlState.bind(this.bot)
        this.bot.setControlState = (control, state) => {
            this.originalSetControlState(control, state)

            if (this.controlStates.hasOwnProperty(control)) {
                const oldState = this.controlStates[control]
                this.controlStates[control] = state
                if (oldState !== state) {
                    this.recordCallback(control, state)
                }
            }
        }

        // Intercept placeBlock
        if (this.bot?.placeBlock) {
            this.originalPlaceBlock = this.bot.placeBlock.bind(this.bot)
            this.bot.placeBlock = async (referenceBlock, faceVector) => {
                let blockName = 'unknown'
                try {
                    const heldItem = this.bot.heldItem
                    if (heldItem) blockName = heldItem.name
                } catch {
                    // Silent fail
                }

                this.controlStates.place = true
                this.recordCallback('place', true, blockName)
                if (this.rlActionCallback) this.rlActionCallback('place', blockName)

                await this.preActionScreenshotWait()

                const result = await this.originalPlaceBlock(referenceBlock, faceVector)

                setTimeout(() => {
                    this.controlStates.place = false
                    this.recordCallback('place', false, blockName)
                }, 50)

                return result
            }
        }

        // Intercept activateItem (open/activate interactions)
        if (this.bot?.activateItem) {
            this.originalActivateItem = this.bot.activateItem.bind(this.bot)
            this.bot.activateItem = async () => {
                await this.preActionScreenshotWait()
                const result = await this.originalActivateItem()
                if (this.bot.currentWindow) {
                    const windowType = this.bot.currentWindow.type
                    this.recordCallback('activateItem', true, windowType)
                }
                return result
            }
        }

        // Intercept activateBlock (block interactions like table/furnace opening)
        if (this.bot?.activateBlock) {
            this.originalActivateBlock = this.bot.activateBlock.bind(this.bot)
            this.bot.activateBlock = async (...args) => {
                await this.preActionScreenshotWait()
                return await this.originalActivateBlock(...args)
            }
        }

        // Intercept equip
        if (this.bot?.equip) {
            this.originalEquip = this.bot.equip.bind(this.bot)
            this.bot.equip = async (item, destination) => {
                const itemName = item?.name || 'unknown'
                this.controlStates.equip = itemName
                this.recordCallback('equip', true, itemName)
                if (this.rlActionCallback) this.rlActionCallback('equip', itemName)

                await this.preActionScreenshotWait()

                const result = await this.originalEquip(item, destination)

                return result
            }
        }

        // Intercept look (camera actions)
        if (this.bot?.look) {
            this.originalLook = this.bot.look.bind(this.bot)
            this.bot.look = async (...args) => {
                await this.preActionScreenshotWait()
                return await this.originalLook(...args)
            }
        }

        // Intercept dig/mining
        if (this.bot?.dig) {
            this.originalDig = this.bot.dig.bind(this.bot)
            this.bot.dig = async (...args) => {
                if (this.rlActionCallback) this.rlActionCallback('attack', true)
                await this.preActionScreenshotWait()
                const result = await this.originalDig(...args)
                setTimeout(() => {
                    if (this.rlActionCallback) this.rlActionCallback('attack', false)
                }, 100)
                return result
            }
        }

        // Intercept crafting
        if (this.bot?.craft) {
            this.originalCraft = this.bot.craft.bind(this.bot)
            this.bot.craft = async (...args) => {
                await this.preActionScreenshotWait()
                return await this.originalCraft(...args)
            }
        }

        // Intercept furnace opening for smelting interactions
        if (this.bot?.openFurnace) {
            this.originalOpenFurnace = this.bot.openFurnace.bind(this.bot)
            this.bot.openFurnace = async (...args) => {
                await this.preActionScreenshotWait()
                return await this.originalOpenFurnace(...args)
            }
        }

        // Intercept direct attack
        if (this.bot?.attack) {
            this.originalAttack = this.bot.attack.bind(this.bot)
            this.bot.attack = async (...args) => {
                if (this.rlActionCallback) this.rlActionCallback('attack', true)
                await this.preActionScreenshotWait()
                const result = await this.originalAttack(...args)
                setTimeout(() => {
                    if (this.rlActionCallback) this.rlActionCallback('attack', false)
                }, 100)
                return result
            }
        }
    }

    /**
     * Start monitoring
     */
    start(pollInterval = 50) {
        if (this.isEnabled) return
        this.isEnabled = true
        this.setupControlInterception()

        this.monitoringInterval = setInterval(() => {
            this.checkForMineActivated()
        }, pollInterval)
    }

    /**
     * Stop monitoring and restore original bot methods
     */
    stop() {
        if (!this.isEnabled) return
        this.isEnabled = false

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval)
            this.monitoringInterval = null
        }

        if (this.originalSetControlState && this.bot) {
            this.bot.setControlState = this.originalSetControlState
        }
        if (this.originalPlaceBlock && this.bot) {
            this.bot.placeBlock = this.originalPlaceBlock
        }
        if (this.originalActivateItem && this.bot) {
            this.bot.activateItem = this.originalActivateItem
        }
        if (this.originalActivateBlock && this.bot) {
            this.bot.activateBlock = this.originalActivateBlock
        }
        if (this.originalEquip && this.bot) {
            this.bot.equip = this.originalEquip
        }
        if (this.originalLook && this.bot) {
            this.bot.look = this.originalLook
        }
        if (this.originalDig && this.bot) {
            this.bot.dig = this.originalDig
        }
        if (this.originalCraft && this.bot) {
            this.bot.craft = this.originalCraft
        }
        if (this.originalOpenFurnace && this.bot) {
            this.bot.openFurnace = this.originalOpenFurnace
        }
        if (this.originalAttack && this.bot) {
            this.bot.attack = this.originalAttack
        }
    }

    /**
     * Check for mine/window events and infer attack/craft/smelt transitions.
     */
    checkForMineActivated() {
        if (!this.isEnabled || !this.bot) return

        const wasWindowOpen = !!this.controlStates.openWindow
        const hasWindowOpen = !!this.bot.currentWindow
        const windowType = this.bot.currentWindow?.type ?? null

        if (hasWindowOpen !== wasWindowOpen) {
            this.controlStates.openWindow = hasWindowOpen
            this.recordCallback('openWindow', hasWindowOpen, windowType)

            if (hasWindowOpen) {
                this.lastWindowType = windowType
            } else {
                this.detectCraftingOrSmelting(this.lastWindowType)
                this.lastWindowType = null
            }
        }

        const isMining = !!this.bot.targetDigBlock
        if (isMining !== this.controlStates.mine) {
            this.controlStates.mine = isMining
            this.recordCallback('mine', isMining)
            if (this.rlActionCallback) this.rlActionCallback('attack', isMining)
        }
    }

    /**
     * Detect crafting or smelting at window close.
     */
    detectCraftingOrSmelting(windowType) {
        if (!windowType || !this.rlActionCallback) return

        if (windowType === 'minecraft:crafting' || windowType === 'minecraft:crafting_table') {
            this.rlActionCallback('craft', 'unknown')
        }

        if (windowType === 'minecraft:furnace') {
            this.rlActionCallback('smelt', 'iron_ingot')
        }
    }
}