/**
 * activity-log.js — Activity log strip at the bottom.
 * Subscribes to /ws/motus and appends log entries.
 */

import { onMotusEvent } from './motus-stream.js';

export function initActivityLog() {
  onMotusEvent(null, _append);
  _initCollapse();
}

function _initCollapse() {
  const strip = document.getElementById('activity-strip');
  const btn = document.getElementById('activity-collapse-btn');
  if (!strip || !btn) return;

  // restore persisted state
  if (localStorage.getItem('activity-collapsed') === '1') {
    strip.classList.add('collapsed');
  }

  document.getElementById('activity-toggle').addEventListener('click', () => {
    strip.classList.toggle('collapsed');
    localStorage.setItem('activity-collapsed', strip.classList.contains('collapsed') ? '1' : '0');
  });
}

function _append(event) {
  if (event.type === 'ping') return;

  const log = document.getElementById('activity-log');
  if (!log) return;

  const atBottom = log.scrollHeight - log.scrollTop <= log.clientHeight + 40;

  const row = document.createElement('div');
  row.className = 'log-entry';

  const t        = new Date(event.ts * 1000).toLocaleTimeString();
  const mcpTag   = event.mcp_id ? `<span style="color:var(--accent);margin-right:4px">[${event.mcp_id}]</span>` : '';
  const msg      = _summarize(event);

  row.innerHTML = `
    <span class="log-time">${t}</span>
    <span class="log-type ${event.type}">${event.type}</span>
    <span class="log-msg">${mcpTag}${msg}</span>
  `;
  log.appendChild(row);

  while (log.children.length > 500) log.removeChild(log.firstChild);

  if (atBottom) log.scrollTop = log.scrollHeight;
}

function _summarize(event) {
  const p = event.payload || {};
  switch (event.type) {
    case 'mcp_call':       return `${p.tool || ''}(${_trunc(JSON.stringify(p.args || {}), 80)})`;
    case 'mcp_result':     return `← ${_trunc(JSON.stringify(p.result), 100)}`;
    case 'agent_thought':  return p.text || '';
    case 'asr_result':     return `"${p.text || ''}"`;
    case 'trigger':        return p.text || _trunc(JSON.stringify(p), 60);
    case 'render':         return `renderer=${p.renderer}`;
    case 'status':
      return 'connected' in p
        ? (p.connected ? '● 已连接' : '○ 断开')
        : (p.online ? '⬤ online' : '○ offline') + (p.mcp_id ? ` [${p.mcp_id}]` : '');
    default:               return _trunc(JSON.stringify(p), 80);
  }
}

function _trunc(str, n) {
  return (str || '').length > n ? str.slice(0, n) + '…' : (str || '');
}
