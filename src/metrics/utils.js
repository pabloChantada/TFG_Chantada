export function getBotName(bot) {
    return bot?.username || bot?.name || 'unknown'
}

export function getBotPosition(bot) {
    if (!bot?.entity?.position) return null
    const pos = bot.entity.position
    return {
        // Round to 3 decimal places for consistency
        // Also minecraft usually has 3 decimal places for position
        x: pos.x.toFixed(3),
        y: pos.y.toFixed(3),
        z: pos.z.toFixed(3)
    }
}

export function captureInventoryState(bot) {
    if (!bot) return []
    return bot.inventory.items().map(item => ({
        name: item.name,
        id: item.type,
        count: item.count
    }))
}

export function compareInventory(startInv, endInv) {
    const changes = []
    const endMap = new Map(endInv.map(item => [item.name, item.count]))
    
    // Check for changed or removed items
    for (const startItem of startInv) {
        const endCount = endMap.get(startItem.name) || 0
        if (endCount !== startItem.count) {
            changes.push({
                item: startItem.name,
                delta: endCount - startItem.count
            })
        }
    }
    
    // Check for new items
    const startMap = new Map(startInv.map(item => [item.name, item.count]))
    for (const endItem of endInv) {
        if (!startMap.has(endItem.name)) {
            changes.push({
                item: endItem.name,
                delta: endItem.count
            })
        }
    }
    
    return changes.length > 0 ? changes : null
}