/**
 * Control data analysis and statistics
 */

export class ControlAnalytics {
    constructor(eventRecorder) {
        this.eventRecorder = eventRecorder
    }

    /**
     * Get control statistics
     */
    getControlStats() {
        const stats = {
            totalPresses: 0,
            controlCounts: {},
            mostUsedControl: null,
            leastUsedControl: null
        }

        // Initialize counts
        const sequence = this.eventRecorder.getControlSequence()
        const controlNames = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak', 'mine', 'place', 'openWindow']
        
        for (const control of controlNames) {
            stats.controlCounts[control] = 0
        }

        // Count presses only
        for (const event of sequence) {
            if (event.action === 'pressed' && stats.controlCounts[event.control] !== undefined) {
                stats.controlCounts[event.control]++
                stats.totalPresses++
            }
        }

        // Find most and least used
        let maxCount = 0, minCount = Infinity
        for (const control in stats.controlCounts) {
            const count = stats.controlCounts[control]
            if (count > maxCount) {
                maxCount = count
                stats.mostUsedControl = control
            }
            if (count < minCount && count >= 0) {
                minCount = count
                stats.leastUsedControl = control
            }
        }

        return stats
    }

    /**
     * Analyze movement patterns
     */
    getMovementPatterns() {
        const patterns = {
            total_move_time_ms: 0,
            forward_presses: 0,
            backward_presses: 0,
            left_presses: 0,
            right_presses: 0,
            jump_events: 0,
            sprint_events: 0,
            common_combinations: []
        }

        const sequence = this.eventRecorder.getControlSequence()
        let currentMovingControls = new Set()

        for (const event of sequence) {
            if (event.action !== 'pressed') continue

            switch (event.control) {
                case 'forward': patterns.forward_presses++; break
                case 'back': patterns.backward_presses++; break
                case 'left': patterns.left_presses++; break
                case 'right': patterns.right_presses++; break
                case 'jump': patterns.jump_events++; break
                case 'sprint': patterns.sprint_events++; break
            }

            // Track movement combinations
            if (['forward', 'back', 'left', 'right'].includes(event.control)) {
                if (event.action === 'pressed') {
                    currentMovingControls.add(event.control)
                } else {
                    currentMovingControls.delete(event.control)
                }

                if (currentMovingControls.size > 0) {
                    // Create a sorted combination string for consistent tracking
                    const combo = Array.from(currentMovingControls).sort().join('+')
                    const existing = patterns.common_combinations.find(c => c.combination === combo)
                    if (existing) {
                        existing.count++
                    } else {
                        patterns.common_combinations.push({ combination: combo, count: 1 })
                    }
                }
            }
        }

        patterns.common_combinations.sort((a, b) => b.count - a.count)
        return patterns
    }

    /**
     * Export all data
     */
    exportControlData() {
        return {
            statistics: this.getControlStats()
        }
    }
}