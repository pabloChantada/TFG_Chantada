/**
 * ActionTracker - Automatic action tracking based on bot state
 * Monitors bot.pathfinder state to detect and track actions automatically
 * Eliminates need for manual trackActionStart/End calls in primitives
 */
export class ActionTracker {
    constructor(bot, metricsCollector) {
        this.bot = bot;
        this.metrics = metricsCollector;
        this.currentAction = null;
        this.startTime = null;
        this.monitoringInterval = null;
        this.isEnabled = false;
    }

    /**
     * Start automatic action monitoring
     * @param {number} pollInterval - How often to check state (ms, default 100ms)
     */
    start(pollInterval = 100) {
        if (this.isEnabled) return;
        this.isEnabled = true;

        this.monitoringInterval = setInterval(() => {
            this.checkStateChange();
        }, pollInterval);

        console.log('[ActionTracker] Automatic action tracking started');
    }

    /**
     * Stop automatic action monitoring
     */
    stop() {
        if (!this.isEnabled) return;
        this.isEnabled = false;

        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }

        // End any ongoing action
        if (this.currentAction) {
            this.endCurrentAction(false);
        }

        console.log('[ActionTracker] Automatic action tracking stopped');
    }

    /**
     * Detect current bot action based on pathfinder state
     * @returns {string|null} Action name or null if idle
     */
    detectAction() {
        if (!this.bot || !this.bot.pathfinder) return null;

        // Priority order: mining > building > moving > idle
        if (this.bot.pathfinder.isMining?.()) return 'mine';
        if (this.bot.pathfinder.isBuilding?.()) return 'build';
        if (this.bot.pathfinder.isMoving?.()) return 'move';

        return null;
    }

    /**
     * Check if bot state changed and update tracking
     * @private
     */
    checkStateChange() {
        if (!this.isEnabled || !this.bot) return;

        const newAction = this.detectAction();

        // No state change
        if (newAction === this.currentAction) return;

        // Action ended
        if (this.currentAction && !newAction) {
            this.endCurrentAction(true);
            return;
        }

        // Action changed
        if (this.currentAction && newAction && newAction !== this.currentAction) {
            this.endCurrentAction(true);
            this.startNewAction(newAction);
            return;
        }

        // New action started
        if (!this.currentAction && newAction) {
            this.startNewAction(newAction);
            return;
        }
    }

    /**
     * Start tracking a new action
     * @private
     */
    startNewAction(actionName) {
        this.currentAction = actionName;
        this.startTime = Date.now();
        this.metrics.trackActionStart(actionName, this.bot);
    }

    /**
     * End current action tracking
     * @private
     */
    async endCurrentAction(success) {
        if (!this.currentAction) return;

        const actionName = this.currentAction;
        const duration = (Date.now() - this.startTime) / 1000;

        this.currentAction = null;
        this.startTime = null;

        await this.metrics.trackActionEnd(success, this.bot);

        console.log(`[ActionTracker] Action '${actionName}' ended (${duration.toFixed(2)}s, success: ${success})`);
    }

    /**
     * Manually mark current action as failed
     * Used when primitives detect errors
     */
    async markActionFailed() {
        if (this.currentAction) {
            await this.endCurrentAction(false);
        }
    }

    /**
     * Get current tracked action info
     * @returns {Object|null} Current action info or null if idle
     */
    getCurrentAction() {
        if (!this.currentAction) return null;

        return {
            name: this.currentAction,
            duration: (Date.now() - this.startTime) / 1000,
            startTime: new Date(this.startTime)
        };
    }

    /**
     * Force end current action (for error handling)
     */
    async forceEnd(success = false) {
        if (this.currentAction) {
            await this.endCurrentAction(success);
        }
    }
}
