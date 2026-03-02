/**
 * Used as an ID for the JSON file, so we can check it with the hash of the commit
 */

import fs from 'fs'

export class VersionManager {
    /**
     * Get current git commit hash
     */
    static async getGitVersion() {
        try {
            // Obtain the current git branch
            const refPath = '.git/' + fs.readFileSync('.git/HEAD', 'utf-8').trim().split(' ')[1]
            // Obtain the commit hash from the ref file
            const refContent = fs.readFileSync(refPath, 'utf-8')
            return refContent.toString().trim()
        } catch (error) {
            console.warn(`[VersionManager] Could not get git version: ${error.message}`)
            return 'unknown'
        }
    }
}