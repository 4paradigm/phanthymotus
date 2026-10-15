/**
 * activity-log.js — Activity log strip at the bottom.
 * Subscribes to /ws/motus and appends log entries.
 */

import { onMotusEvent } from './motus-stream.js';
import { summarizeEvent } from './activity-summary.js';
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
  const msg      = summarizeEvent(event);

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
