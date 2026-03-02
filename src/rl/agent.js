/**
 * RL Agent for wood chopping.
 * 
 * Supports two modes:
 *   1. Random policy (for data collection / baseline)
 *   2. Model-based policy (loads trained model via HTTP inference server)
 * 
 * The model server is expected to be the one from src/evaluation/llm.py
 * or any server that accepts POST with {state, image} and returns {action}.
 */

import { NUM_ACTIONS, ACTION_NAMES } from './actions.js'

export class RLAgent {
    /**
     * @param {object} options
     * @param {'random'|'model'|'epsilon_greedy'} options.policy - Policy type
     * @param {string} options.modelUrl - URL for model inference (if policy=model or epsilon_greedy)
     * @param {number} options.epsilon - Exploration rate for epsilon_greedy (default 0.1)
     */
    constructor(options = {}) {
        this.policy = options.policy || 'random'
        this.modelUrl = options.modelUrl || 'http://localhost:5000/predict'
        this.epsilon = options.epsilon ?? 0.1
        this.actionCounts = new Array(NUM_ACTIONS).fill(0)
        this.totalSteps = 0
    }

    /**
     * Select an action given the current observation.
     * 
     * @param {number[]} observation - State vector from extractState()
     * @param {object} meta - Additional metadata (cursorBlockName, etc.)
     * @returns {Promise<number>} Action index
     */
    async selectAction(observation, meta = {}) {
        this.totalSteps++
        let actionIndex

        switch (this.policy) {
            case 'random':
                actionIndex = this._randomAction()
                break

            case 'model':
                actionIndex = await this._modelAction(observation, meta)
                break

            case 'epsilon_greedy':
                if (Math.random() < this.epsilon) {
                    actionIndex = this._randomAction()
                } else {
                    actionIndex = await this._modelAction(observation, meta)
                }
                break

            default:
                actionIndex = this._randomAction()
        }

        this.actionCounts[actionIndex]++
        return actionIndex
    }

    /**
     * Random action selection with weighted probabilities.
     * Biases toward useful actions (forward, mine, look) over noop.
     * @returns {number}
     */
    _randomAction() {
        // Weighted distribution favoring movement and mining
        const weights = [
            0.02,  // noop
            0.15,  // forward
            0.05,  // back
            0.05,  // left
            0.05,  // right
            0.05,  // jump
            0.18,  // mine
            0.12,  // forward_mine
            0.06,  // look_up
            0.06,  // look_down
            0.06,  // turn_left
            0.06,  // turn_right
            0.04,  // forward_jump
            0.04,  // sprint_forward
            0.01,  // stop_all
        ]

        const totalWeight = weights.reduce((a, b) => a + b, 0)
        let rand = Math.random() * totalWeight
        for (let i = 0; i < weights.length; i++) {
            rand -= weights[i]
            if (rand <= 0) return i
        }
        return 0
    }

    /**
     * Query the trained model for an action prediction.
     * Falls back to random on failure.
     * 
     * @param {number[]} observation 
     * @param {object} meta 
     * @returns {Promise<number>}
     */
    async _modelAction(observation, meta) {
        try {
            const response = await fetch(this.modelUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state: observation,
                    meta: meta,
                }),
            })

            if (!response.ok) {
                console.warn(`[RL-Agent] Model server returned ${response.status}, falling back to random`)
                return this._randomAction()
            }

            const data = await response.json()
            const actionIndex = data.action ?? data.action_index ?? 0

            if (actionIndex < 0 || actionIndex >= NUM_ACTIONS) {
                console.warn(`[RL-Agent] Invalid action ${actionIndex} from model, falling back to random`)
                return this._randomAction()
            }

            return actionIndex
        } catch (err) {
            console.warn(`[RL-Agent] Model inference failed: ${err.message}, falling back to random`)
            return this._randomAction()
        }
    }

    /**
     * Get action distribution statistics.
     * @returns {object}
     */
    getStats() {
        const stats = {}
        for (let i = 0; i < NUM_ACTIONS; i++) {
            stats[ACTION_NAMES[i]] = {
                count: this.actionCounts[i],
                percentage: this.totalSteps > 0
                    ? ((this.actionCounts[i] / this.totalSteps) * 100).toFixed(1) + '%'
                    : '0%'
            }
        }
        return {
            policy: this.policy,
            total_steps: this.totalSteps,
            epsilon: this.epsilon,
            action_distribution: stats
        }
    }

    /**
     * Reset counters (for new training run).
     */
    resetStats() {
        this.actionCounts.fill(0)
        this.totalSteps = 0
    }
}