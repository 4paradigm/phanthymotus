// Shared by WebSocket activity and manual-card completion polling. Presentation
// only: never suppress an ACP receipt, barrier notification or LLM event.
const shown = new Set();

export function claimCompletionActivity(actionId) {
  if (typeof actionId !== 'string' || !actionId) return true;
  if (shown.has(actionId)) return false;
  shown.add(actionId);
  if (shown.size > 2048) shown.delete(shown.values().next().value);
  return true;
}

export function activityCompletionId(event) {
  if (event?.type !== 'trigger') return null;
  let value = event.payload;
  if (typeof value?.text === 'string') {
    // Collector may prefix a single JSON event with [source]. Do not discard
    // batched triggers containing other events merely because one ID repeats.
    const match = value.text.trim().match(/^(?:\[[^\]\r\n]*\]\s*)?(\{[\s\S]*\})\s*$/);
    if (!match) return null;
    try { value = JSON.parse(match[1]); } catch { return null; }
  }
  return value?.type === 'action_complete' && typeof value.action_id === 'string'
    && ['completed', 'error', 'failed', 'cancelled'].includes(value.status)
    ? value.action_id : null;
}
