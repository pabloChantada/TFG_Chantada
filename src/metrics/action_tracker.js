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
        this.startPosition = null;
        this.startInventoryCount = 0;
        this.monitoringInterval = null;
        this.isEnabled = false;
        
        // Thresholds for success detection
        this.STUCK_THRESHOLD_MS = 8000; // Movement longer than 8s considered potentially stuck
        this.MIN_MOVEMENT_DISTANCE = 1.0; // Minimum distance to consider movement successful
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

        console.log(`[ActionTracker] [${this.bot.username}] Automatic action tracking started`);
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

        console.log(`[ActionTracker] [${this.bot.username}] Automatic action tracking stopped`);
    }

    /**
     * Detect current bot action based on pathfinder state
     * @returns {string|null} Action name or null if idle
     */
    detectAction() {
        if (!this.bot || !this.bot.pathfinder) return null;

        // Priority order: mining > building > moving > idle
        if (this.bot.targetDigBlock) return 'mine';
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

        // Check for stuck movement while action is ongoing
        if (this.currentAction === 'move' && newAction === 'move') {
            const duration = Date.now() - this.startTime;
            if (duration > this.STUCK_THRESHOLD_MS && this.startPosition) {
                const distance = this.startPosition.distanceTo(this.bot.entity.position);
                if (distance < this.MIN_MOVEMENT_DISTANCE) {
                    console.warn(`[ActionTracker] [${this.bot.username}] Bot appears stuck during move (${(duration/1000).toFixed(1)}s, ${distance.toFixed(2)} blocks)`);
                    // Don't end the action here - let it complete naturally but log the warning
                }
            }
        }

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
        
        // Capture initial state for success evaluation
        if (this.bot?.entity?.position) {
            this.startPosition = this.bot.entity.position.clone();
        }
        
        // Count total inventory items for mining success detection
        this.startInventoryCount = this.bot?.inventory?.items()?.reduce((sum, item) => sum + item.count, 0) || 0;
        
        this.metrics.trackActionStart(actionName, this.bot);
    }

    /**
     * Evaluate if the action was successful based on state changes
     * @private
     * @returns {boolean} Whether the action appears to have succeeded
     */
    evaluateActionSuccess() {
        if (!this.currentAction || !this.bot) return false;
        
        const duration = Date.now() - this.startTime;
        
        switch (this.currentAction) {
            case 'move': {
                // Success if: moved significant distance AND didn't take too long
                if (!this.startPosition || !this.bot.entity?.position) return false;
                const distance = this.startPosition.distanceTo(this.bot.entity.position);
                const tookTooLong = duration > this.STUCK_THRESHOLD_MS;
                const movedEnough = distance >= this.MIN_MOVEMENT_DISTANCE;
                
                if (tookTooLong && !movedEnough) {
                    console.log(`[ActionTracker] [${this.bot.username}] Move appears stuck: ${distance.toFixed(2)} blocks in ${(duration/1000).toFixed(1)}s`);
                    return false;
                }
                return movedEnough;
            }
            
            case 'mine': {
                // Success if: inventory increased (got items from mining)
                const currentCount = this.bot.inventory?.items()?.reduce((sum, item) => sum + item.count, 0) || 0;
                return currentCount > this.startInventoryCount;
            }
            
            case 'build': {
                // Build actions are harder to verify - assume success if completed without error
                // The actual placement success should be verified by the calling code
                return true;
            }
            
            default:
                return true;
        }
    }

    /**
     * End current action tracking
     * @private
     */
    async endCurrentAction(naturalEnd) {
        if (!this.currentAction) return;

        const actionName = this.currentAction;
        const duration = (Date.now() - this.startTime) / 1000;

        let success = false;
        if (naturalEnd) {
            if (actionName === 'mine') {
                // Brief non-blocking wait for inventory update
                await new Promise(resolve => setTimeout(resolve, 300));
            }

            // Evaluate actual success based on state changes, not just natural ending
            success = this.evaluateActionSuccess();
        }

        this.currentAction = null;
        this.startTime = null;
        this.startPosition = null;
        this.startInventoryCount = 0;

        // Log immediately (before awaiting metrics queue) for responsive feedback
        // console.log(`[ActionTracker] [${this.bot.username}] Action '${actionName}' ended (${duration.toFixed(2)}s, success: ${success})`);
        
        await this.metrics.trackActionEnd(success, this.bot);
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
     * @param {boolean} success - Whether to mark as successful (default false for errors)
     */
    async forceEnd(success = false) {
        if (this.currentAction) {
            // For forced ends, bypass the evaluation
            const actionName = this.currentAction;
            const duration = (Date.now() - this.startTime) / 1000;

            this.currentAction = null;
            this.startTime = null;
            this.startPosition = null;
            this.startInventoryCount = 0;

            console.log(`[ActionTracker] [${this.bot.username}] Action '${actionName}' force-ended (${duration.toFixed(2)}s, success: ${success})`);
            await this.metrics.trackActionEnd(success, this.bot);
        }
    }
}
