/** Read only the Core receipt for a returned action_id, never call the device.
 * Polling tolerates early callbacks and a dropped WebSocket without changing ACP.
 */
export function completionRequest(data, schema, action) {
  const spec = schema?.['x-completion'];
  if (!spec || (spec.actions?.length && !spec.actions.includes(action))) return null;
  const parts = Array.isArray(data) ? data : [data];
  for (let part of parts) {
    if (part?.type === 'text') part = part.text;
    if (typeof part === 'string') {
      try { part = JSON.parse(part); } catch { continue; }
    }
    if (!part || typeof part.action_id !== 'string' || !part.action_id || part.action_id.length > 256) continue;
    if (['completed', 'failed', 'error', 'cancelled'].includes(part.state)) continue;
    const seconds = typeof spec.timeout === 'number' && Number.isFinite(spec.timeout) ? spec.timeout : 120;
    return { actionId: part.action_id, timeoutMs: Math.min(600, Math.max(1, seconds)) * 1000 + 5000 };
  }
  return null;
}

export async function watchCompletion({ actionId, timeoutMs, isCurrent, onResult,
  fetchStatus = (url, options) => fetch(url, options),
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), now = () => performance.now() }) {
  const deadline = now() + timeoutMs;
  while (isCurrent() && now() < deadline) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(5000, Math.max(1, deadline - now())));
    try {
      const response = await fetchStatus(`/api/mcp/actions/${encodeURIComponent(actionId)}`,
        { cache: 'no-store', signal: controller.signal });
      if (!isCurrent()) return;
      if (response.ok) {
        const json = await response.json(), value = json.data;
        if (!isCurrent()) return;
        if (json.code === 200 && value?.action_id === actionId) {
          if (value.state === 'settled') {
            const state = ({ completed: 'completed', error: 'failed', failed: 'failed', cancelled: 'cancelled' })[value.status];
            if (state) {
              onResult({ state, action_id: actionId, result: value.result }, state !== 'completed');
              return;
            }
          } else if (['unknown', 'unconfirmed'].includes(value.state)) {
            onResult({ state: 'unconfirmed', action_id: actionId,
              message: value.message || 'Completion record unavailable. Check actual device state; do not assume success.' }, true);
            return;
          }
        }
      }
    } catch { /* A temporary connection error is not a robot failure. */ }
    finally { clearTimeout(timer); }
    if (isCurrent()) await sleep(500);
  }
  if (isCurrent()) onResult({ state: 'unconfirmed', action_id: actionId,
    message: 'Completion was not confirmed before the display timeout. Check actual device state.' }, true);
}
