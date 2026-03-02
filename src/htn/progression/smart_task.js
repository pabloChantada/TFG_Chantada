import { getBotLabel } from './utils.js'
import { exploreRandom } from '../primitives/movement.js'

/**
 * Executes a task with retry and recovery logic.
 * @param {Bot} bot - Bot instance
 * @param {string} taskName - Name for logs
 * @param {Function} conditionFn - Function that returns true if task is done (post-condition)
 * @param {Function} actionFn - Main action to execute
 * @param {Function} recoveryFn - (Optional) Action to execute if actionFn fails
 * @param {Object} options - Additional options
 * @returns {Promise<void>}
 */
async function runSmartTask(bot, taskName, conditionFn, actionFn, recoveryFn = null, options = {}) {
    const exploreOnFail = options.exploreOnFail !== false
    
    // Check if task is already done before starting
    if (await conditionFn()) {
        console.log(`[SKIP] [${getBotLabel(bot)}] ${taskName} completed. Skipping...`)
        return
    }

    console.log(`[START] [${getBotLabel(bot)}] ${taskName}`)
    let attempts = 0
    const maxAttempts = 5

    while (!await conditionFn() && attempts < maxAttempts) {
        try {
            // Execute main action
            await actionFn()
            await bot.waitForTicks(100)
        } catch (err) {
            attempts++
            console.error(`[ERROR] [${getBotLabel(bot)}] ${taskName}: ${err.message}`)
            console.log(`[FAIL] [${getBotLabel(bot)}] ${taskName} - Try ${attempts}/${maxAttempts}`)

            // Check if we've exhausted attempts
            if (attempts >= maxAttempts) {
                console.error(`[ABORT] [${getBotLabel(bot)}] Max attempts reached for ${taskName}`)
                break
            }

            // Replan
            if (recoveryFn) {
                console.log(`[REPLAN] Executing recovery for ${taskName}...`)
                try {
                    await recoveryFn(err)
                    console.log(`[REPLAN] Recovery finished. Retrying main task.`)
                } catch (recoveryErr) {
                    console.error(`[FATAL] Recovery failed: ${recoveryErr.message}`)
                    if (exploreOnFail) {
                        try {
                            await exploreRandom(bot, 10)
                        } catch (exploreError) {
                            console.warn(`[REPLAN] [${getBotLabel(bot)}] Explore failed: ${exploreError.message}`)
                        }
                    }
                }
            } else if (exploreOnFail) {
                try {
                    await exploreRandom(bot, 10)
                } catch (exploreError) {
                    console.warn(`[REPLAN] [${getBotLabel(bot)}] Explore failed: ${exploreError.message}`)
                }
            }
        }
    }

    if (!await conditionFn()) {
        throw new Error(`[ABORT] [${getBotLabel(bot)}] Could not complete ${taskName} after ${maxAttempts} attempts.`)
    }
    console.log(`[COMPLETED] [${getBotLabel(bot)}] ${taskName}`)
}

export { runSmartTask }