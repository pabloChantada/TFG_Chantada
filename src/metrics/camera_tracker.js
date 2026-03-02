/**
 * CameraTracker - Bot camera/orientation helper
 * Provides current bot orientation (yaw/pitch) for integration with other trackers
 */

function round(num, decimals) {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

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
            yaw: round(this.bot.entity.yaw, 4),
            pitch: round(this.bot.entity.pitch, 4)
        };
    }

    /**
     * Get current camera state with heading description
     * @returns {Object} {yaw, pitch, heading} or null
     */
    getOrientationWithHeading() {
        const orientation = this.getCurrentOrientation();
        if (!orientation) return null;

        // Convert yaw to compass heading
        let heading = (-orientation.yaw + Math.PI) * 180 / Math.PI;
        heading = ((heading % 360) + 360) % 360;

        const directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
        const index = Math.round(heading / 45) % 8;

        return {
            yaw: orientation.yaw,
            pitch: orientation.pitch,
            heading: heading.toFixed(1),
            direction: directions[index]
        };
    }
}
