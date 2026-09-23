/**
 * activity-log.js — Activity log strip at the bottom.
 * Subscribes to /ws/motus and appends log entries.
 */

import { onMotusEvent } from './motus-stream.js';
import { toggleLog } from './mobile.js';

export function initActivityLog() {
  onMotusEvent(null, _append);
  _initCollapse();
  _initResize();
}

// Kept in step with `.activity-strip`'s own min-height / max-height, so a drag
// cannot leave the element at a size the stylesheet then clamps back.
const _MIN_H = 80;
function _maxH() { return Math.round(window.innerHeight * 0.6); }

/**
 * Drag the strip's top edge to set its height.
 *
 * This replaces CSS `resize: vertical`, whose handle is mouse-only — on a
 * tablet the log was stuck at whichever of collapsed/expanded it was in, with
 * no way to see more of it. Pointer events cover mouse, pen and touch through
 * one path, and unlike the native handle the chosen height can be persisted.
 */
function _initResize() {
  const strip = document.getElementById('activity-strip');
  const handle = document.getElementById('activity-resize-handle');
  if (!strip || !handle) return;

  const saved = parseInt(localStorage.getItem('activity-height') || '', 10);
  if (Number.isFinite(saved)) {
    strip.style.height = `${Math.min(Math.max(saved, _MIN_H), _maxH())}px`;
  }

  let startY = 0;
  let startH = 0;

  handle.addEventListener('pointerdown', (e) => {
    if (strip.classList.contains('collapsed')) return;
    startY = e.clientY;
    startH = strip.getBoundingClientRect().height;
    // Mark the drag as live before anything that can throw: setPointerCapture
    // is redundant for touch, which captures implicitly, and raises on some
    // pointer ids. Letting it abort here would leave the strip without the
    // 'resizing' class, and every later move would bail on that check.
    strip.classList.add('resizing');
    try { handle.setPointerCapture(e.pointerId); } catch { /* implicit capture */ }
    e.preventDefault();
  });

  handle.addEventListener('pointermove', (e) => {
    if (!strip.classList.contains('resizing')) return;
    const dy = startY - e.clientY;      // drag up = taller
    strip.style.height = `${Math.min(Math.max(startH + dy, _MIN_H), _maxH())}px`;
  });

  const end = () => {
    if (!strip.classList.contains('resizing')) return;
    strip.classList.remove('resizing');
    localStorage.setItem('activity-height', String(Math.round(strip.getBoundingClientRect().height)));
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);

  // The grip lives inside the header, and the header is the collapse toggle, so
  // both the click ending a drag and a stray tap on the grip would otherwise
  // bubble up and toggle the bar.
  handle.addEventListener('click', (e) => e.stopPropagation());
}

function _initCollapse() {
  const strip = document.getElementById('activity-strip');
  const toggle = document.getElementById('activity-toggle');
  if (!strip || !toggle) return;

  // Restore the bar's persisted state, but only where a bar is what gets shown.
  // Below 768px the log is a drawer and `collapsed` has no styling behind it, so
  // carrying the flag in only left the class on the element misrepresenting a
  // state the user could not see or change.
  if (localStorage.getItem('activity-collapsed') === '1' &&
      !window.matchMedia('(max-width: 768px)').matches) {
    strip.classList.add('collapsed');
  }

  // Delegated to the shared toggle so the header, the floating button and the
  // monitor-header button all mean the same thing — see mobile.js toggleLog.
  toggle.addEventListener('click', () => toggleLog());
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

  // Mirror onto the ledge, which shows this instead of the panel's own name
  // once the bar is shut. Only the type is dropped — it is a coloured chip that
  // would dominate a single line, and the message usually names the tool anyway.
  const latest = document.getElementById('activity-latest');
  if (latest) {
    latest.innerHTML = `<span class="log-time">${t}</span><span>${mcpTag}${msg}</span>`;
    document.getElementById('activity-strip')?.classList.add('has-latest');
  }

  while (log.children.length > 1000) log.removeChild(log.firstChild);

  if (atBottom) log.scrollTop = log.scrollHeight;
}

function _summarize(event) {
  const p = event.payload || {};
  switch (event.type) {
    case 'mcp_call':       return `${p.tool || ''}(${_trunc(JSON.stringify(p.args || {}), 200)})`;
    case 'mcp_result':     return `← ${_trunc(JSON.stringify(p.result), 400)}`;
    case 'agent_thought':  return p.text || '';
    // 单独一类，不并进 agent_thought：本功能最坏的失败模式是汇报里编造一个不存在的
    // 发现，得能和模型自己的想法分开看才抓得住。
    // 轮数维度已删（纯时间触发），payload 里只剩秒数 —— 再读 silent_rounds 会渲染成
    // 「静默undefined轮」。in_turn 区分"turn 还在跑"和"turn 结束后子代理还在干"。
    case 'narration':      return `📢 ${p.text || ''} (静默${p.silent_seconds}s${p.in_turn ? '' : '·后台'})`;
    case 'asr_result':     return `"${p.text || ''}"`;
    case 'trigger':        return p.text || _trunc(JSON.stringify(p), 400);
    case 'peer_pair_request': return `${p.display_name || p.peer_id?.slice(0, 12) || 'peer'} 请求配对 · 验证码 ${p.code}`;
    case 'peer_tool_call':   return `${p.peer || 'peer'} → ${p.tool}${p.action ? `(${p.action})` : ''}`;
    case 'peer_tool_result': return p.ok
      ? `${p.peer || 'peer'} ← ${p.tool} 完成${p.elapsed_ms != null ? ` ${p.elapsed_ms}ms` : ''}${p.action_id ? ` [${p.action_id}]` : ''}`
      : `${p.peer || 'peer'} ← ${p.tool} 失败: ${_trunc(String(p.error || ''), 200)}`;
    case 'render':         return `renderer=${p.renderer}`;
    case 'llm_usage':      return `tokens: in=${p.prompt_tokens} out=${p.completion_tokens} cached=${p.cached_tokens}`
                              + (p.elapsed_s != null ? ` · ${p.elapsed_s}s` : '');
    case 'turn_end':
      if (p.usage) return `✓ ${p.rounds}轮 ${p.duration_s}s | tokens: in=${p.usage.prompt_tokens} out=${p.usage.completion_tokens} cached=${p.usage.cached_tokens}`;
      return '✓ turn complete';
    case 'status':
      return 'connected' in p
        ? (p.connected ? '● 已连接' : '○ 断开')
        : (p.online ? '⬤ online' : '○ offline') + (p.mcp_id ? ` [${p.mcp_id}]` : '');
    default:               return _trunc(JSON.stringify(p), 300);
  }
}

// **Cuts out of the middle, not off the end.**
//
// Every payload here is a JSON blob whose boilerplate comes first and whose
// point comes last: an ACP completion spends its first 180 characters on
// `type` / `action_id` / `status` before reaching `reason`, so a head-only cut
// reliably kept the part that could be guessed and dropped the part that could
// not. On the robot this showed up as an activity row that always ended in
// `"reaso`. The middle of a blob is the cheapest thing to lose.
function _trunc(str, n) {
  str = str || '';
  if (str.length <= n) return str;
  const tail = Math.min(Math.floor(n / 3), 200);
  return `${str.slice(0, n - tail)}…（省略 ${str.length - n} 字）…${str.slice(-tail)}`;
}
