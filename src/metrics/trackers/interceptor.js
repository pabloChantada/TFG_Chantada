/**
 * Bot control state interception and monitoring
 */

export class ControlInterceptor {
    constructor(bot, controlStates, recordCallback) {
        this.bot = bot
        this.controlStates = controlStates
        this.recordCallback = recordCallback
        this.originalSetControlState = null
        this.monitoringInterval = null
        this.isEnabled = false
    }

    /**
     * Intercept bot.setControlState() calls. Used for tracking custom
     * controls WASD/jump/sprint and detecting mine/place actions.
     */
    setupControlInterception() {
        if (!this.bot?.setControlState) return

        this.originalSetControlState = this.bot.setControlState.bind(this.bot)

        this.bot.setControlState = (control, state) => {
            this.originalSetControlState(control, state)
            // Update our tracked state and record the event
            if (this.controlStates.hasOwnProperty(control)) {
                const oldState = this.controlStates[control]
                this.controlStates[control] = state
                // Only record if the state actually changed
                if (oldState !== state) {
                    this.recordCallback(control, state)
                }
            }
        }

        // Intercept placeBlock
        if (this.bot?.placeBlock) {
            this.originalPlaceBlock = this.bot.placeBlock.bind(this.bot)
            this.bot.placeBlock = async (...args) => {
                const result = await this.originalPlaceBlock(...args)
                this.controlStates.place = true
                this.recordCallback('place', true)
                setTimeout(() => {
                    this.controlStates.place = false
                    this.recordCallback('place', false)
                }, 50)
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

        // Monitor mine/place events
        this.monitoringInterval = setInterval(() => {
            this.checkForMineActivated()
        }, pollInterval)

        // Initial check in case bot is already mining/placing when tracking starts
        if (this.originalPlaceBlock && this.bot) {
            this.bot.placeBlock = this.originalPlaceBlock
        }
    }

    /**
     * Stop monitoring
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
    }

    /**
     * Check for mine/place events
     */
    checkForMineActivated() {
        if (!this.isEnabled || !this.bot) return
        // Detect if the bot is using a furnace or crafting table by checking if a window is open
        const hasWindowOpen = !!this.bot.currentWindow
        if (hasWindowOpen !== this.controlStates.openWindow) {
            this.controlStates.openWindow = hasWindowOpen
            const windowType = this.bot.currentWindow?.type ?? null
            this.recordCallback('openWindow', hasWindowOpen, windowType)
        }
        // Detect mining by checking if the bot has a target block it's digging
        const isMining = !!this.bot.targetDigBlock
        if (isMining !== this.controlStates.mine) {
            this.controlStates.mine = isMining
            this.recordCallback('mine', isMining)
        }

    }
}