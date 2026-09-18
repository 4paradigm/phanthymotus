/**
 * json-util.js — shared value handling for the JSON-ish renderers.
 *
 * Bus payloads routinely carry a second layer: a field whose value is a JSON
 * *string* rather than a JSON value. It happens wherever something was
 * serialised twice on its way here — a driver forwarding a response verbatim,
 * an LLM tool call whose `arguments` the API defines as a string, a producer
 * using a `json.dumps` that defaults to `ensure_ascii=True`.
 *
 * Displayed as-is it reaches the panel as source text rather than data, escapes
 * and all: `你好` arrives as `你好` and a nested object shows its
 * backslash-escaped quotes. That is what made the decision_core card unreadable,
 * but nothing about it is specific to that topic — any producer can double-encode
 * — so the decoding lives here and every JSON renderer gets it.
 */

const MAX_DEPTH = 4;

/**
 * Decode a value, unwrapping any strings that are themselves JSON.
 *
 * Only strings that are structurally an object or an array are unwrapped. A
 * bare quoted string or a numeric string is left alone: `"123"` must not become
 * `123`, and prose that happens to sit in quotes must not silently lose them.
 *
 * Depth-bounded — a malformed producer can nest these arbitrarily, and the
 * point is to make the payload readable, not to chase it to the bottom.
 */
export function decodeNestedJson(value, depth = 0) {
  if (typeof value === 'string') {
    if (depth >= MAX_DEPTH) return value;
    const inner = _parseStructured(value);
    return inner === undefined ? value : decodeNestedJson(inner, depth + 1);
  }
  // Walking into an array or an object costs nothing against the budget. The
  // budget exists to stop a chain of strings-encoding-strings, and spending it
  // on ordinary structure meant the real payload never got decoded at all:
  // decision_core nests decisions → round → tool_calls → call → args, which
  // exhausted a depth of 4 one level before reaching the string that needed it.
  if (Array.isArray(value)) return value.map(v => decodeNestedJson(v, depth));
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = decodeNestedJson(v, depth);
    return out;
  }
  return value;
}

function _parseStructured(s) {
  const t = s.trim();
  if (t.length < 2) return undefined;
  const open = t[0], close = t[t.length - 1];
  if (!((open === '{' && close === '}') || (open === '[' && close === ']'))) return undefined;
  try {
    const parsed = JSON.parse(t);
    return (parsed && typeof parsed === 'object') ? parsed : undefined;
  } catch {
    return undefined;
  }
}

/** A decoded value as compact, readable text — no quotes, no escapes. */
export function plainValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === 'boolean') return String(v);
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) return v.map(plainValue).join(', ');
  if (typeof v === 'object') {
    return Object.entries(v).map(([k, x]) => `${k}=${plainValue(x)}`).join(' ');
  }
  return String(v);
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
