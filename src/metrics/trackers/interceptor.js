/**
 * Bot control state interception and monitoring
 */

export class ControlInterceptor {
    constructor(bot, controlStates, recordCallback, rlActionCallback = null) {
        this.bot = bot
        this.controlStates = controlStates
        this.recordCallback = recordCallback
        this.rlActionCallback = rlActionCallback

        this.originalSetControlState = null
        this.originalPlaceBlock = null
        this.originalActivateItem = null
        this.originalEquip = null
        this.originalAttack = null

        this.monitoringInterval = null
        this.isEnabled = false
        this.lastWindowType = null
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

                const result = await this.originalPlaceBlock(referenceBlock, faceVector)

                this.controlStates.place = true
                this.recordCallback('place', true, blockName)
                if (this.rlActionCallback) this.rlActionCallback('place', blockName)

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
                const result = await this.originalActivateItem()
                if (this.bot.currentWindow) {
                    const windowType = this.bot.currentWindow.type
                    this.recordCallback('activateItem', true, windowType)
                }
                return result
            }
        }

        // Intercept equip
        if (this.bot?.equip) {
            this.originalEquip = this.bot.equip.bind(this.bot)
            this.bot.equip = async (item, destination) => {
                const itemName = item?.name || 'unknown'
                const result = await this.originalEquip(item, destination)

                this.controlStates.equip = itemName
                this.recordCallback('equip', true, itemName)
                if (this.rlActionCallback) this.rlActionCallback('equip', itemName)

                return result
            }
        }

        // Intercept direct attack
        if (this.bot?.attack) {
            this.originalAttack = this.bot.attack.bind(this.bot)
            this.bot.attack = async (...args) => {
                if (this.rlActionCallback) this.rlActionCallback('attack', true)
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
        if (this.originalEquip && this.bot) {
            this.bot.equip = this.originalEquip
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