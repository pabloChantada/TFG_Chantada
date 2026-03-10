/**
 * CameraTracker - Bot camera/orientation helper
 * Provides current bot orientation (yaw/pitch) for integration with other trackers
 */

export class CameraTracker {
    constructor(bot) {
        this.bot = bot;
    }
    /**
     * Get current bot orientation (yaw/pitch)
     * @returns {Object} Current orientation {yaw, pitch} or null if unavailable
     */
    getCurrentOrientation() {
        if (!this.bot || !this.bot.entity) return null;
        
        return {
            // Mineflayer exposes yaw in radians. Round to 4 decimals for precision/space balance
            yaw: Math.round(this.bot.entity.yaw * 10000) / 10000,
            // Pitch is -90° when looking straight up, 0° when looking at the horizon, and +90° when looking straight down
            // In radians: -1.5708 (up) to 0 (horizon) to 1.5708 (down)
            pitch: Math.round(this.bot.entity.pitch * 10000) / 10000
        };
    }
}
