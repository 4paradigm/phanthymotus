/** activity.js — Default renderer: shows MCP request/response timeline */
export const ActivityRenderer = {
  name: 'activity',
  canRender: (hint) => !hint
    || hint.startsWith('sensor/') || hint.startsWith('control/')
    || hint.startsWith('state/')  || hint.startsWith('data/'),
  _el: null,

  mount(container, mcpId) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-activity';
    container.appendChild(this._el);
    this._mcpId = mcpId;
  },

  onData(buffer, fmt) {
    if (!this._el) return;
    try {
      const text = new TextDecoder().decode(buffer);
      const obj = JSON.parse(text);
      if (obj.type === 'ping' || obj.type === 'meta') return;
      const row = document.createElement('div');
      row.className = 'log-entry';
      const t = obj.asr_complete_ts
        ? new Date(obj.asr_complete_ts * 1000).toLocaleTimeString()
        : new Date().toLocaleTimeString();
      const msg = obj.text
        ? `<span style="color:var(--text-primary)">${obj.text}</span>`
        : `<span style="color:var(--text-muted)">${_truncate(text, 120)}</span>`;
      row.innerHTML = `<span class="log-time">${t}</span><span class="log-type asr_result">asr</span><span class="log-msg">${msg}</span>`;
      this._el.appendChild(row);
      this._el.scrollTop = this._el.scrollHeight;
    } catch {
      // not JSON — ignore
    }
  },

  onEvent(event) {
    if (!this._el) return;
    const row = document.createElement('div');
    row.className = 'log-entry';
    const t = new Date(event.ts * 1000).toLocaleTimeString();
    row.innerHTML = `
      <span class="log-time">${t}</span>
      <span class="log-type ${event.type}">${event.type}</span>
      <span class="log-msg">${_summarize(event)}</span>
    `;
    this._el.appendChild(row);
    this._el.scrollTop = this._el.scrollHeight;
  },

  unmount() {
    this._el?.remove();
    this._el = null;
  },
};

function _summarize(event) {
  const p = event.payload || {};
  switch (event.type) {
    case 'mcp_call':     return `${p.tool}(${JSON.stringify(p.args || {})})`;
    case 'mcp_result':   return `← ${_truncate(JSON.stringify(p.result), 120)}`;
    case 'agent_thought': return p.text || '';
    case 'render':       return `renderer=${p.renderer}`;
    case 'status':       return p.online ? '⬤ online' : '○ offline';
    default:             return JSON.stringify(p).slice(0, 80);
  }
}

function _truncate(str, n) {
  return str.length > n ? str.slice(0, n) + '…' : str;
}
