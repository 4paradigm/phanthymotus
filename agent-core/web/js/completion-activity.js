// Shared by WebSocket activity and manual-card completion polling. Presentation
// only: never suppress an ACP receipt, barrier notification or LLM event.
const activities = new Map();

function activity(actionId) {
  let value = activities.get(actionId);
  if (!value) {
    value = { manual: false, shown: null };
    activities.set(actionId, value);
    if (activities.size > 2048) activities.delete(activities.keys().next().value);
  }
  return value;
}

// Manual cards supply the detailed completion row. A collector receipt can
// arrive before the HTTP response identifies that card's action, so callers
// also remove its provisional row when this function returns true.
export function preferManualCompletionActivity(actionId) {
  if (typeof actionId !== 'string' || !actionId) return false;
  const value = activity(actionId);
  const removeCollector = value.shown === 'collector';
  value.manual = true;
  if (removeCollector) value.shown = null;
  return removeCollector;
}

export function claimCompletionActivity(actionId, source = 'canvas') {
  if (typeof actionId !== 'string' || !actionId) return true;
  const value = activity(actionId);
  if ((source === 'collector' && value.manual) || value.shown) return false;
  value.shown = source;
  return true;
}

export function activityCompletionId(event) {
  if (event?.type !== 'trigger') return null;
  let value = event.payload;
  if (typeof value?.text === 'string') {
    // Real collector text has an <event ...> envelope. Legacy sources may
    // instead prefix bare JSON with [source]. Parse exactly one event: mixed
    // messages or sibling envelopes must remain visible in the activity log.
    const text = value.text.trim().replace(/^\[[^\]\r\n]*\]\s*/, '');
    const envelope = text.match(/^<event(?:\s[^>]*)?>\s*(\{[\s\S]*\})\s*<\/event>$/);
    try { value = JSON.parse(envelope ? envelope[1] : text); } catch { return null; }
  }
  return value?.type === 'action_complete' && typeof value.action_id === 'string'
    && ['completed', 'error', 'failed', 'cancelled'].includes(value.status)
    ? value.action_id : null;
}
