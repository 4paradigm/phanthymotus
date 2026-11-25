/**
 * x-options-action — a config field whose choices only the tool can find out.
 *
 * The printers on a PC a driver reaches over SSH, the pages open in a browser
 * there: nothing the dashboard can list itself. Such a field names the action
 * that finds them, and the config dialog puts a button beside it
 * (sidebar.js). The action replies with
 *
 *   { "options": { "<field>": [{const, title}, ...] },   new choices for selects
 *     "help":    { "<field>": "<text>" } }                new help under fields
 *
 * The rules that decide what the dialog shows are here, as plain functions;
 * the DOM work in sidebar.js stays too thin to be worth testing.
 */

/** The object a tool returned: MCP text content parsed, or the data as is. */
export function parseToolReply(data) {
  const text = Array.isArray(data) ? data.find(c => c && c.type === 'text')?.text : null;
  if (text == null) return data && typeof data === 'object' && !Array.isArray(data) ? data : {};
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === 'object' ? parsed : { message: text };
  } catch {
    return { message: text };
  }
}

/** What a select keeps after new choices arrive: the current pick while still offered. */
export function keptSelection(current, choices) {
  return choices.some(c => String(c.const) === String(current)) ? current : '';
}

/** The reply to show when the call itself failed: its reason, under the field. */
export function failureReply(key, err) {
  return { help: { [key]: String((err && err.message) || err) } };
}
