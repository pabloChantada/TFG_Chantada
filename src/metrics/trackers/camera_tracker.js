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
            // Radians  ->  Degrees
            // 0        ->  0°    (forward)
            //  1.5708  ->  90°   (left)
            //  -1.5708  ->  -90°   (right)
            //  3.1416  ->  180°  (backwards)
            // -3.1416  -> -180°  (backwards, same as 180°)
            yaw: this.bot.entity.yaw.toFixed(3),      
            // Pitch is -90° when looking straight up, 0° when looking at the horizon, and +90° when looking straight down
            // In radians: -1.5708 (up) to 0 (horizon) to 1.5708 (down)
            pitch: this.bot.entity.pitch.toFixed(3)
        };
    }
}
