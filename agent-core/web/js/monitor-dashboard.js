/**
 * monitor-dashboard.js — Monitor mode dashboard grid.
 * Apple-widget-style layout: fixed cell grid, cards snap to cells,
 * drag to reposition, resize by snapping to cell boundaries.
 */

import { ActivityRenderer } from './renderers/activity.js';
import { TextRenderer }     from './renderers/text.js';
import { VideoRenderer }    from './renderers/video.js';
import { ImageRenderer }    from './renderers/image.js';
import { AudioRenderer }    from './renderers/audio.js';
import { LidarRenderer }    from './renderers/lidar.js';
import { ControlRenderer }  from './renderers/control.js';
import { PointCloudRenderer } from './renderers/pointcloud.js';
import { MappingRenderer }   from './renderers/mapping.js';
import { SkeletonRenderer } from './renderers/skeleton.js';
import { KvLatestRenderer } from './renderers/kv-latest.js';
import { CameraRenderer, DepthRenderer, DepthZlibRenderer } from './renderers/camera.js';
import { CostmapRenderer, OdometryRenderer, PathRenderer } from './renderers/navigation.js';
import { resolveDerivedTopics, syncConnectionTopics } from './topic-derive.js';

const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, DepthZlibRenderer, ImageRenderer, AudioRenderer, PointCloudRenderer, MappingRenderer, CostmapRenderer, OdometryRenderer, PathRenderer, LidarRenderer, SkeletonRenderer, ControlRenderer, TextRenderer, ActivityRenderer];
const STORAGE_KEY = 'monitor-dashboard-layout-v2';
// Column count by viewport width. Desktop used to be a flat 5 whatever the
// window was, which is only right near 1080p: at 1280 a column is 237px and a
// KV panel fits one key per row, at 800 it is 131px and video/skeleton cards
// are unreadable, and on a 3440 ultrawide it is 665px of mostly empty card.
// Bands target ~300px per column, which given the grid's 48px of padding and
// 12px gaps puts every threshold from three columns up exactly 300 apart.
// A column narrower than ~280px is where the KV panel drops to one key per row
// and video stops being worth looking at, so that is the floor the ladder keeps.
const COL_BANDS = [
  [480, 1], [900, 2], [1200, 3], [1500, 4], [1800, 5],
  [2100, 6], [2400, 7], [2700, 8], [3000, 9],
];
const MAX_COLS = 10;    // above the last band
const EDGE = 48;        // px from a grid edge where a drag starts auto-scrolling
const EDGE_SPEED = 14;  // px per frame of auto-scroll
let _topicMcpMap = {};  // topic → mcpId, populated on fetch

let _grid = null;
let _cards = new Map(); // topicPath → { el, renderer, ws, format, mode, col, row, colSpan, rowSpan }
let _totalCols = MAX_COLS;

function _getResponsiveCols() {
  const w = window.innerWidth;
  for (const [maxWidth, cols] of COL_BANDS) if (w <= maxWidth) return cols;
  return MAX_COLS;
}

export function activate() {
  _grid = document.getElementById('monitor-dashboard-grid');
  _grid.innerHTML = '';
  _cards.clear();
  _totalCols = _getResponsiveCols();
  _applyGridStyle();
  _fetchAndBuild();
  window.addEventListener('resize', _onResize);
}

export function deactivate() {
  for (const card of _cards.values()) {
    card.ws?.close();
    card.renderer?.unmount?.();
  }
  _cards.clear();
  if (_grid) _grid.innerHTML = '';
  window.removeEventListener('resize', _onResize);
}

function _onResize() {
  const cols = _getResponsiveCols();
  if (cols === _totalCols) return;
  _totalCols = cols;
  _applyGridStyle();
  _reflowIntoColumns();
}

/**
 * Bring every card inside the current column count.
 *
 * Rewriting `grid-template-columns` alone was not enough: the cards kept the
 * column indices they were given at five columns, so CSS grid created implicit
 * columns to hold them and squeezed the one real column down to nothing. On a
 * 420px viewport two cards ended up 2px wide — effectively invisible, with no
 * way to recover them short of clearing the layout.
 *
 * Cards are re-packed in reading order so the result is stable and predictable
 * rather than dependent on which ones happened to overflow.
 */
function _reflowIntoColumns() {
  const ordered = [..._cards.values()].sort((a, b) => a.row - b.row || a.col - b.col);
  const settled = [];
  for (const card of ordered) {
    card.colSpan = _clampSpan(card.colSpan);
    const slot = _findFreeSlot(settled, card.colSpan, card.rowSpan);
    card.col = slot.col;
    card.row = slot.row;
    settled.push({ col: card.col, row: card.row, colSpan: card.colSpan, rowSpan: card.rowSpan });
    _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
  }
  _saveLayout();
}

function _applyGridStyle() {
  if (!_grid) return;
  _grid.style.gridTemplateColumns = `repeat(${_totalCols}, 1fr)`;
}

// ── Grid geometry ─────────────────────────────────────────────────────────────
//
// Measured from the DOM, never re-derived. The stylesheet owns these numbers and
// changes them per breakpoint — rows step 140/160/190/220px by viewport height,
// with the gap dropping 12px → 8px below 768px wide. An earlier version
// hardcoded `rect.height * 0.2 + 12` to match a `20%` row height, and was wrong
// two ways: the percentage resolved against the content box while
// getBoundingClientRect includes padding (5px of drift per row), and below 768px
// it bore no relation to the real pitch at all, so dragging by one row moved the
// card two. Reading a real card's box is exact and survives any band change
// here without a matching constant.

function _gaps() {
  const cs = getComputedStyle(_grid);
  return { row: parseFloat(cs.rowGap) || 0, col: parseFloat(cs.columnGap) || 0 };
}

/** Height of one row plus the gap below it. */
function _rowPitch() {
  const gap = _gaps().row;
  for (const card of _cards.values()) {
    const h = card.el.getBoundingClientRect().height;
    if (h > 0 && card.rowSpan > 0) return (h - (card.rowSpan - 1) * gap) / card.rowSpan + gap;
  }
  // No card to measure yet. `gridAutoRows` is only useful when the stylesheet
  // gave it in px; a percentage comes back verbatim and has to be resolved here.
  const cs = getComputedStyle(_grid);
  const declared = parseFloat(cs.gridAutoRows);
  if (isFinite(declared) && cs.gridAutoRows.endsWith('px')) return declared + gap;
  const pad = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
  const pct = (parseFloat(cs.gridAutoRows) || 20) / 100;
  return Math.max(1, (_grid.clientHeight - pad) * pct) + gap;
}

/** Width of one column plus the gap beside it. */
function _colPitch() {
  const gap = _gaps().col;
  for (const card of _cards.values()) {
    const w = card.el.getBoundingClientRect().width;
    if (w > 0 && card.colSpan > 0) return (w - (card.colSpan - 1) * gap) / card.colSpan + gap;
  }
  const cs = getComputedStyle(_grid);
  const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
  return Math.max(1, (_grid.clientWidth - pad - gap * (_totalCols - 1)) / _totalCols) + gap;
}

/** Pointer position expressed in grid cells, scroll included. */
function _pointerCell(e) {
  const r = _grid.getBoundingClientRect();
  const cs = getComputedStyle(_grid);
  const x = e.clientX - r.left + _grid.scrollLeft - (parseFloat(cs.paddingLeft) || 0);
  const y = e.clientY - r.top + _grid.scrollTop - (parseFloat(cs.paddingTop) || 0);
  return { x, y };
}

async function _fetchAndBuild() {
  let mcps = [];
  let topicDetails = {};
  let layout = {};
  try {
    const [mcpRes, topicRes, layoutRes] = await Promise.all([
      fetch('/api/mcp').then(r => r.json()),
      fetch('/api/topics').then(r => r.json()),
      fetch('/api/canvas/layout').then(r => r.json()),
    ]);
    mcps = mcpRes.data || [];
    const items = topicRes.data || [];
    for (const t of items) topicDetails[t.topic] = t;
    layout = layoutRes.data || {};
  } catch { /* silent */ }

  const canvasCards = layout.cards || [];
  const connections = layout.connections || [];
  const canvasTools = new Set(canvasCards.map(c => `${c.mcpId}:${c.toolName}`));

  // The layout is not a reliable record of derived topics — see topic-derive.js.
  // multiInstance marks the tools whose topic is derived rather than declared,
  // so those are re-asked even when the layout already has a topic for them: a
  // card that has lost its inbound connection still carries the topic derived
  // from it, and nothing else would ever correct it.
  const isDerived = (card) => {
    const mcp = mcps.find(m => m.id === card.mcpId);
    const tool = (mcp?.tools || []).find(t => (typeof t === 'string' ? t : t.name) === card.toolName);
    return !!(typeof tool === 'object' && tool.multiInstance);
  };
  await resolveDerivedTopics(canvasCards, connections, { isDerived });
  // A connection's fromTopic is read below as a topic to subscribe to, and it is
  // just as stale as the card topic it was copied from — sync it before use.
  syncConnectionTopics(canvasCards, connections);

  const topicSet = new Set();
  const hiddenTopics = new Set();
  _topicMcpMap = {};  // reset
  const addMonitorTopic = (item, mcpId) => {
    if (!item?.topic) return;
    if (item.monitor === false) {
      hiddenTopics.add(item.topic);
      topicSet.delete(item.topic);
      return;
    }
    if (!hiddenTopics.has(item.topic)) {
      topicSet.add(item.topic);
      _topicMcpMap[item.topic] = mcpId;
    }
  };
  // First pass: collect all topic_out from static tool definitions (non-multiInstance)
  for (const mcp of mcps) {
    const mcpOnCanvas = canvasCards.some(c => c.mcpId === mcp.id);
    if (!mcpOnCanvas) continue;
    for (const tool of (mcp.tools || [])) {
      if (tool.multiInstance) continue;  // handled via card instances below
      if (!canvasTools.has(`${mcp.id}:${tool.name}`)) continue;
      for (const t of (tool.topic_out || [])) addMonitorTopic(t, mcp.id);
    }
  }
  // Second pass: add topic_in from static tools only if not already covered
  for (const mcp of mcps) {
    const mcpOnCanvas = canvasCards.some(c => c.mcpId === mcp.id);
    if (!mcpOnCanvas) continue;
    for (const tool of (mcp.tools || [])) {
      if (tool.multiInstance) continue;
      if (!canvasTools.has(`${mcp.id}:${tool.name}`)) continue;
      for (const t of (tool.topic_in || [])) addMonitorTopic(t, mcp.id);
    }
  }
  // Instance-specific topics from each canvas card (covers multiInstance tools like ASR/TTS)
  for (const card of canvasCards) {
    for (const t of (card.topicOut || [])) addMonitorTopic(t, card.mcpId);
    for (const t of (card.topicIn || [])) addMonitorTopic(t, card.mcpId);
  }
  // Dynamic connection topics remain visible unless either endpoint declares them hidden.
  for (const conn of connections) {
    if (conn.fromTopic && !hiddenTopics.has(conn.fromTopic)) topicSet.add(conn.fromTopic);
  }

  // Fallback: fill _topicMcpMap from /api/topics mcp_id for any topic not yet mapped
  for (const t of Object.values(topicDetails)) {
    if (t.topic && t.mcp_id && !_topicMcpMap[t.topic]) {
      _topicMcpMap[t.topic] = t.mcp_id;
    }
  }

  if (topicSet.size === 0) {
    _grid.innerHTML = `<div class="monitor-dashboard-empty">
      <div class="placeholder-icon">◎</div>
      <p>暂无活跃的数据流</p>
      <p style="font-size:11px;opacity:0.5">部署驱动并启动监控后，数据流将在此显示</p>
    </div>`;
    return;
  }

  // Apply grid style
  _applyGridStyle();

  // Load persisted layout
  const savedLayout = _loadLayout();
  const topics = [...topicSet];

  // Two passes, because a single pass could not see what it was about to
  // collide with. `_findFreeSlot` only knows about cards already placed, and the
  // loop ran in topic order — so a new topic took the first free slot without
  // knowing that a *later* topic held a saved position there, and the saved card
  // was then restored right on top of it. Placing every saved card first means
  // new cards are fitted around all of them.
  const placed = [];
  const fresh = [];
  for (const topicPath of topics) {
    const saved = savedLayout[topicPath];
    const colSpan = _clampSpan(saved?.colSpan || 1);
    const rowSpan = saved?.rowSpan || 2;
    if (saved?.col != null && saved?.row != null) {
      // Clamp into the current column count. A layout arranged at five columns
      // and reopened at one left cards sitting on columns that no longer exist,
      // and CSS grid answered by inventing implicit ones: measured on a 420px
      // viewport the single real column collapsed to 2px and two cards with it.
      const col = Math.max(0, Math.min(_totalCols - colSpan, saved.col));
      placed.push({ topicPath, col, row: saved.row, colSpan, rowSpan, mode: saved.mode || 'log' });
    } else {
      fresh.push({ topicPath, colSpan, rowSpan, mode: 'log' });
    }
  }
  // Saved positions can also collide with each other — a layout written before
  // the clamp above, or two cards resized into one another. Resolve rather than
  // render them stacked.
  const settled = [];
  let rearranged = fresh.length > 0;
  for (const c of placed) {
    if (_overlapsAny(c, settled)) {
      Object.assign(c, _findFreeSlot(settled, c.colSpan, c.rowSpan));
      rearranged = true;
    }
    settled.push(c);
  }
  for (const c of fresh) {
    Object.assign(c, _findFreeSlot(settled, c.colSpan, c.rowSpan));
    settled.push(c);
  }

  for (const c of settled) {
    const detail = topicDetails[c.topicPath];
    _createCard(c.topicPath, detail?.format || 'activity', detail?.status || 'offline',
                c.col, c.row, c.colSpan, c.rowSpan, c.mode);
  }
  // Persist what was just decided. Auto-placement used to be recomputed on every
  // load against whatever order the API happened to return, so a card nobody had
  // ever dragged moved around by itself between refreshes.
  if (rearranged) _saveLayout();
}

function _clampSpan(span) {
  return Math.max(1, Math.min(_totalCols, span));
}

function _overlapsAny(c, others) {
  return others.some(o =>
    c.col < o.col + o.colSpan && c.col + c.colSpan > o.col &&
    c.row < o.row + o.rowSpan && c.row + c.rowSpan > o.row);
}

function _createCard(topicPath, format, status, col, row, colSpan, rowSpan, savedMode) {
  const el = document.createElement('div');
  el.className = 'monitor-card';
  el.dataset.topic = topicPath;

  // Apply grid placement
  _applyPlacement(el, col, row, colSpan, rowSpan);

  const fmtClass = _formatClass(format);
  const shortName = topicPath.split('/').filter(Boolean).pop() || topicPath;
  const isJson = format === 'data/json' || format?.startsWith('text/');
  const defaultMode = (isJson && savedMode) ? savedMode : 'log';

  const modeHtml = isJson
    ? `<div class="monitor-card-modes">
        <button class="monitor-card-mode-btn${defaultMode === 'log' ? ' active' : ''}" data-mode="log">日志</button>
        <button class="monitor-card-mode-btn${defaultMode === 'kv' ? ' active' : ''}" data-mode="kv">最新</button>
       </div>`
    : '';

  el.innerHTML = `
    <div class="monitor-card-header">
      <div class="monitor-card-dot ${_esc(status)}"></div>
      <div class="monitor-card-names">
        <span class="monitor-card-topic">${_esc(shortName)}</span>
        <span class="monitor-card-path" title="${_esc(topicPath)}">${_esc(_displayPath(topicPath))}</span>
      </div>
      ${modeHtml}
      <span class="monitor-card-format ${fmtClass}">${_esc(_formatLabel(format))}</span>
    </div>
    <div class="monitor-card-body"></div>
    <div class="monitor-card-empty">
      <span class="monitor-card-empty-title">尚未收到数据</span>
      <span class="monitor-card-empty-hint">发布方启动后会自动出现</span>
    </div>
    <div class="monitor-card-resize"></div>
  `;

  const body = el.querySelector('.monitor-card-body');
  const renderer = _createRenderer(format, defaultMode);
  renderer.mount(body, _topicMcpMap[topicPath] || 'dashboard');

  const ws = _connectWs(topicPath, format, renderer);
  const card = { el, renderer, ws, format, mode: defaultMode, col, row, colSpan, rowSpan };
  _cards.set(topicPath, card);

  // Mode toggle
  if (isJson) {
    const modeBtns = el.querySelectorAll('.monitor-card-mode-btn');
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => _switchMode(topicPath, btn.dataset.mode, modeBtns));
    });
  }

  // The whole card is the drag handle, minus the controls that need their own
  // clicks. Only the 49px header used to be grabbable — 12% of a 230px card —
  // which is both hard to hit and the reason a missed grab landed on the body
  // and started selecting log text instead of moving anything.
  el.addEventListener('pointerdown', (e) => _beginPointer(e, topicPath, 'move'));

  const resizeHandle = el.querySelector('.monitor-card-resize');
  resizeHandle.addEventListener('pointerdown', (e) => _beginPointer(e, topicPath, 'resize'));

  _grid.appendChild(el);
}

function _applyPlacement(el, col, row, colSpan, rowSpan) {
  el.style.gridColumn = `${col + 1} / span ${colSpan}`;
  el.style.gridRow = `${row + 1} / span ${rowSpan}`;
}

function _connectWs(topicPath, format, renderer) {
  // An unresolved topic used to build `/ws/bus` with nothing after it. That
  // matches no route (the endpoint is `/ws/bus/{topic:path}`), so Starlette
  // failed the handshake and uvicorn logged `ASGI callable returned without
  // completing handshake` + a 500 — noise that reads like a server fault when
  // the real state is simply "this card has no topic yet".
  if (!topicPath || topicPath === '/') {
    console.debug('[monitor] no topic yet, not opening a bus WS');
    return null;
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${location.host}/ws/bus${topicPath}`;

  function create() {
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = _wsHandler(topicPath, () => _cards.get(topicPath)?.renderer || renderer, format);
    ws.onerror = () => {};
    ws.onclose = () => {
      // Auto-reconnect after 5s if card still exists
      const card = _cards.get(topicPath);
      if (card && card.ws === ws) {
        setTimeout(() => {
          const c = _cards.get(topicPath);
          if (c && c.ws === ws) {
            c.ws = create();
          }
        }, 5000);
      }
    };
    return ws;
  }

  return create();
}

/**
 * Deliver one frame to a renderer. Returns whether it was real data.
 *
 * The distinction matters to the empty-state placeholder: the bus also carries
 * keepalive pings and metadata, and a card fed nothing but those would have
 * retired its placeholder and gone back to being the blank rectangle the
 * placeholder exists to explain.
 */
function _handleWsMessage(ev, renderer, format) {
  if (ev.data instanceof ArrayBuffer) {
    if (ev.data.byteLength === 0) return false;
    renderer.onData?.(ev.data, format);
    return true;
  }
  try {
    const parsed = JSON.parse(ev.data);
    if (parsed.type === 'ping' || parsed.type === 'meta' || parsed.type === 'error') return false;
  } catch { /* not JSON — treat as payload */ }
  renderer.onData?.(new TextEncoder().encode(ev.data).buffer, format);
  return true;
}

/**
 * The socket's message handler, shared by the initial connection and by a
 * re-wire after a mode switch — which used to install a plain _handleWsMessage
 * and so stopped tracking whether the card had ever received anything.
 * `getRenderer` is resolved per frame because a mode switch replaces it.
 */
function _wsHandler(topicPath, getRenderer, format) {
  return (ev) => {
    if (!_handleWsMessage(ev, getRenderer(), format)) return;
    const card = _cards.get(topicPath);
    if (card && !card.gotData) {
      card.gotData = true;
      card.el.classList.add('has-data');
    }
  };
}

function _refreshRenderer(topicPath) {
  const card = _cards.get(topicPath);
  if (!card) return;
  const body = card.el.querySelector('.monitor-card-body');
  card.renderer?.unmount?.();
  body.innerHTML = '';
  const renderer = _createRenderer(card.format, card.mode);
  renderer.mount(body, _topicMcpMap[topicPath] || 'dashboard');
  card.renderer = renderer;
  // Re-wire WS. Optional: _connectWs returns null for a card whose topic is
  // not resolved yet, and a mode switch on such a card must not throw.
  if (card.ws) {
    card.ws.onmessage = _wsHandler(topicPath, () => _cards.get(topicPath)?.renderer, card.format);
  }
}

function _switchMode(topicPath, newMode, modeBtns) {
  const card = _cards.get(topicPath);
  if (!card || card.mode === newMode) return;
  card.mode = newMode;
  modeBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === newMode));
  _refreshRenderer(topicPath);
  _saveLayout();
}

function _createRenderer(format, mode) {
  if (mode === 'kv') {
    return Object.assign(Object.create(Object.getPrototypeOf(KvLatestRenderer)), KvLatestRenderer);
  }
  const hint = format || 'activity';
  const Renderer = RENDERERS.find(r => r.canRender(hint)) || ActivityRenderer;
  return Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
}

// ── Auto-placement helper ──────────────────────────────────────────────────────

function _findFreeSlot(placed, colSpan, rowSpan) {
  for (let row = 0; ; row++) {
    for (let col = 0; col <= _totalCols - colSpan; col++) {
      const ok = placed.every(p =>
        col >= p.col + p.colSpan || col + colSpan <= p.col ||
        row >= p.row + p.rowSpan || row + rowSpan <= p.row
      );
      if (ok) return { col, row };
    }
  }
}

// ── Drag to reposition, resize to span (pointer-based, snap to cell) ──────────
//
// Pointer events rather than mouse events: the grid already lays itself out for
// phones and tablets (1 and 2 columns), and the tabbar offers a monitor tab, but
// `mousedown` meant none of it could be rearranged by touch at all.

let _drag = null;     // { kind: 'move' | 'resize', ... }
let _edgeTimer = null;
let _hold = null;     // 触摸时的"按住不动"计时，见 _armHold

// 按住多久算"拿起卡片"，以及这期间允许的手指晃动。
const HOLD_MS = 450;
const HOLD_SLOP = 8;

/** Controls inside a card that must keep their own click behaviour. */
function _isInteractive(target) {
  return !!target.closest('.monitor-card-modes, .monitor-card-resize, button, a, input, select, textarea');
}

function _beginPointer(e, topicPath, kind) {
  if (e.button != null && e.button !== 0) return;
  if (kind === 'move' && _isInteractive(e.target)) return;
  const card = _cards.get(topicPath);
  if (!card) return;

  // 触摸：手指按下先一律当成滚动，按住不动 450ms 才算拿起卡片。
  //
  // 鼠标能把"滚动"和"拖动"分给两个不同的动作（滚轮 / 按住左键），触摸只有一个
  // 手指，两件事的起手式一模一样。原来这里在 pointerdown 上就 preventDefault，
  // 而在触摸上那等于取消本次手势的原生滚动 —— 结果是手指落在任何一张卡片上都
  // 开始搬卡片，整个看板只能从卡片之间的缝隙里滚。默认必须是滚动：在手机上看
  // 数据是常态，调位置是偶尔为之。
  if (e.pointerType && e.pointerType !== 'mouse') {
    _armHold(e, topicPath, card, kind);
    return;
  }
  e.preventDefault();
  e.stopPropagation();
  _startDrag(e, topicPath, card, kind);
}

/** 手指按住不动到点，才真正进入拖动。中途移动或抬起则作罢，让页面照常滚。 */
function _armHold(e, topicPath, card, kind) {
  _cancelHold();
  const startX = e.clientX, startY = e.clientY;
  let last = e;

  const onMove = (ev) => {
    last = ev;
    if (Math.abs(ev.clientX - startX) > HOLD_SLOP ||
        Math.abs(ev.clientY - startY) > HOLD_SLOP) _cancelHold();
  };
  const onEnd = () => _cancelHold();

  const timer = setTimeout(() => {
    _cancelHold();
    _startDrag(last, topicPath, card, kind);
    // 手势已经开始但还没滚动（手指没动过），此时拦 touchmove 才拦得住页面滚动 ——
    // pointermove 上的 preventDefault 对滚动没有作用。
    document.addEventListener('touchmove', _blockTouchScroll, { passive: false });
    card.el.classList.add('lifted');
    navigator.vibrate?.(12);   // 拿起的确认；iOS Safari 没有，静默跳过
  }, HOLD_MS);

  _hold = { timer, onMove, onEnd };
  document.addEventListener('pointermove', onMove);
  document.addEventListener('pointerup', onEnd);
  document.addEventListener('pointercancel', onEnd);
  // 页面滚起来了就说明这是一次滚动，不是要搬卡片。
  _grid.addEventListener('scroll', onEnd, { once: true });
}

function _cancelHold() {
  if (!_hold) return;
  clearTimeout(_hold.timer);
  document.removeEventListener('pointermove', _hold.onMove);
  document.removeEventListener('pointerup', _hold.onEnd);
  document.removeEventListener('pointercancel', _hold.onEnd);
  _grid.removeEventListener('scroll', _hold.onEnd);
  _hold = null;
}

function _blockTouchScroll(e) {
  if (_drag) e.preventDefault();
}

function _startDrag(e, topicPath, card, kind) {
  const pt = _pointerCell(e);
  const rowPitch = _rowPitch();
  const colPitch = _colPitch();
  _drag = {
    kind, topicPath, card, rowPitch, colPitch,
    // Where inside the card the grab landed, so the card does not jump to put
    // its corner under the cursor.
    grabCol: pt.x - card.col * colPitch,
    grabRow: pt.y - card.row * rowPitch,
    startX: pt.x, startY: pt.y,
    origCol: card.col, origRow: card.row,
    origColSpan: card.colSpan, origRowSpan: card.rowSpan,
    moved: false,
    valid: true,
    pointerId: e.pointerId,
  };
  card.el.classList.add(kind === 'move' ? 'dragging' : 'resizing');
  // Deliberately no setPointerCapture. The listeners below are on `document`, so
  // they already see every move; capture only adds a way to lose them, since it
  // retargets events to the captured node and browsers drop the capture when
  // that node is re-laid-out — which is exactly what dragging a card does.
  _showGridOverlay();
  document.addEventListener('pointermove', _onPointerMove);
  document.addEventListener('pointerup', _onPointerEnd);
  document.addEventListener('pointercancel', _onPointerEnd);
}

function _onPointerMove(e) {
  if (!_drag) return;
  const pt = _pointerCell(e);
  // A few pixels of slop so a tap on a card is not recorded as a rearrangement.
  if (!_drag.moved &&
      Math.abs(pt.x - _drag.startX) < 4 && Math.abs(pt.y - _drag.startY) < 4) return;
  _drag.moved = true;

  const { card } = _drag;
  if (_drag.kind === 'move') {
    const col = Math.max(0, Math.min(_totalCols - card.colSpan,
      Math.round((pt.x - _drag.grabCol) / _drag.colPitch)));
    const row = Math.max(0, Math.round((pt.y - _drag.grabRow) / _drag.rowPitch));
    if (col !== card.col || row !== card.row) {
      card.col = col;
      card.row = row;
      _applyPlacement(card.el, col, row, card.colSpan, card.rowSpan);
    }
  } else {
    const colSpan = Math.max(1, Math.min(_totalCols - card.col,
      Math.round((pt.x - card.col * _drag.colPitch) / _drag.colPitch)));
    const rowSpan = Math.max(1, Math.min(6,
      Math.round((pt.y - card.row * _drag.rowPitch) / _drag.rowPitch)));
    if (colSpan !== card.colSpan || rowSpan !== card.rowSpan) {
      card.colSpan = colSpan;
      card.rowSpan = rowSpan;
      _applyPlacement(card.el, card.col, card.row, colSpan, rowSpan);
    }
  }

  const blocking = _blockers(_drag.topicPath, card.col, card.row, card.colSpan, card.rowSpan);
  // A drop always lands. Whatever was in the way is moved aside on release, so
  // there is no arrangement the pointer can reach that the grid will refuse.
  //
  // The previous rule only accepted a drop onto a *single* neighbour of exactly
  // matching span, which is close to unreachable in a real layout: cards get
  // resized to suit their content, so a drag usually straddles two of them or
  // lands on one of a different size, and the card silently snapped back. A drag
  // that mostly undoes itself reads as "cards cannot be moved".
  if (_drag.kind === 'move') {
    _drag.displacing = blocking;
    _drag.valid = true;
  } else {
    // Growing a card over its neighbours would cascade, so a resize still has to
    // fit in the space available.
    _drag.displacing = [];
    _drag.valid = blocking.length === 0;
  }
  card.el.classList.toggle('drag-invalid', !_drag.valid);
  card.el.classList.toggle('drag-swap', _drag.valid && blocking.length > 0);

  _autoScroll(e);
}

/**
 * Scroll the grid when a drag reaches its edge.
 *
 * Without this the reachable area was whatever happened to be on screen when the
 * drag started: the grid scrolls, so a card could not be moved into any row
 * below the fold — the pointer ran out of viewport first. Measured before the
 * fix, holding a card against the bottom edge for over a second left scrollTop
 * at 0.
 */
function _autoScroll(e) {
  clearTimeout(_edgeTimer);
  const r = _grid.getBoundingClientRect();
  let dy = 0;
  if (e.clientY > r.bottom - EDGE) dy = EDGE_SPEED;
  else if (e.clientY < r.top + EDGE) dy = -EDGE_SPEED;
  if (!dy) return;
  const before = _grid.scrollTop;
  _grid.scrollTop += dy;
  if (_grid.scrollTop === before) return;   // already at the end
  // Keep going while the pointer stays put; pointermove alone would stall.
  _edgeTimer = setTimeout(() => { if (_drag) _onPointerMove(e); }, 16);
}

function _onPointerEnd() {
  if (!_drag) return;
  clearTimeout(_edgeTimer);
  const { card, topicPath, kind, moved, valid, displacing } = _drag;

  if (!moved || !valid) {
    // A click that never became a drag must change nothing. The resize path used
    // to treat an untouched `valid` flag as success and rebuild the renderer, so
    // a stray click on the corner handle silently cleared the card's log history
    // and restarted its stream.
    _restore(card);
  } else if (displacing?.length) {
    _displace(topicPath, card, displacing);
  }

  card.el.classList.remove('dragging', 'resizing', 'drag-invalid', 'drag-swap', 'lifted');
  const changed = moved && valid;
  const resized = changed && kind === 'resize';
  _drag = null;
  _hideGridOverlay();
  document.removeEventListener('pointermove', _onPointerMove);
  document.removeEventListener('pointerup', _onPointerEnd);
  document.removeEventListener('pointercancel', _onPointerEnd);
  document.removeEventListener('touchmove', _blockTouchScroll, { passive: false });

  if (changed) _saveLayout();
  if (resized) _refreshRenderer(topicPath);
}

/**
 * Re-home the cards a drop landed on.
 *
 * Each goes to the first slot free once the dragged card is in place. For two
 * equal-sized cards that is exactly the spot the dragged card vacated, so a
 * straight swap falls out of the same rule without being a special case — and a
 * drop onto a card of a different size, or straddling two of them, resolves
 * instead of being refused.
 */
function _displace(topicPath, card, blocked) {
  const moving = new Set(blocked.map(b => b.topicPath));
  const occupied = [{ col: card.col, row: card.row, colSpan: card.colSpan, rowSpan: card.rowSpan }];
  for (const [t, o] of _cards) {
    if (t === topicPath || moving.has(t)) continue;
    occupied.push({ col: o.col, row: o.row, colSpan: o.colSpan, rowSpan: o.rowSpan });
  }
  // Largest first: a wide card left until last would be pushed past a gap that
  // the narrow ones have meanwhile filled.
  const order = [...blocked].sort((a, b) =>
    (b.colSpan * b.rowSpan) - (a.colSpan * a.rowSpan));
  for (const b of order) {
    const other = _cards.get(b.topicPath);
    if (!other) continue;
    const slot = _findFreeSlot(occupied, other.colSpan, other.rowSpan);
    other.col = slot.col;
    other.row = slot.row;
    occupied.push({ col: other.col, row: other.row, colSpan: other.colSpan, rowSpan: other.rowSpan });
    _applyPlacement(other.el, other.col, other.row, other.colSpan, other.rowSpan);
  }
}

function _restore(card) {
  card.col = _drag.origCol;
  card.row = _drag.origRow;
  card.colSpan = _drag.origColSpan;
  card.rowSpan = _drag.origRowSpan;
  _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
}

/** Cards a placement would overlap. */
function _blockers(excludeTopic, col, row, colSpan, rowSpan) {
  const out = [];
  for (const [topicPath, o] of _cards) {
    if (topicPath === excludeTopic) continue;
    if (col < o.col + o.colSpan && col + colSpan > o.col &&
        row < o.row + o.rowSpan && row + rowSpan > o.row) {
      out.push({ topicPath, colSpan: o.colSpan, rowSpan: o.rowSpan });
    }
  }
  return out;
}

// ── Grid Overlay (visual guides during drag/resize) ───────────────────────────

let _overlay = null;

/**
 * Draw the cells a drag will snap to.
 *
 * These are guides, so they have to agree with where cards actually land. They
 * did not: the row count came from a 280px constant that matched no breakpoint
 * (real rows are 109px on desktop), and the overlay's own CSS sized its rows as
 * a percentage of the *visible* box while cards sit in the scrollable content —
 * so the guides drifted further from the truth the further down you looked, and
 * stopped at the fold. Both now come from the same measurement the drag uses.
 */
function _showGridOverlay() {
  if (_overlay) return;
  const pitch = _rowPitch();
  const gap = _gaps().row;
  _overlay = document.createElement('div');
  _overlay.className = 'monitor-grid-overlay';
  const rows = Math.max(1, Math.ceil(_grid.scrollHeight / pitch) + 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < _totalCols; c++) {
      const cell = document.createElement('div');
      cell.className = 'monitor-grid-cell';
      cell.style.gridColumn = `${c + 1}`;
      cell.style.gridRow = `${r + 1}`;
      _overlay.appendChild(cell);
    }
  }
  _overlay.style.gridTemplateColumns = `repeat(${_totalCols}, 1fr)`;
  // Explicit px rows, and tall enough to cover the scrolled content rather than
  // only the part of it that happens to be on screen.
  _overlay.style.gridAutoRows = `${pitch - gap}px`;
  _overlay.style.height = `${rows * pitch}px`;
  _grid.appendChild(_overlay);
}

function _hideGridOverlay() {
  _overlay?.remove();
  _overlay = null;
}

// ── Layout Persistence ────────────────────────────────────────────────────────

function _saveLayout() {
  const layout = {};
  for (const [topic, card] of _cards) {
    layout[topic] = { col: card.col, row: card.row, colSpan: card.colSpan, rowSpan: card.rowSpan, mode: card.mode };
  }
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(layout)); } catch {}
}

function _loadLayout() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
}

export function resetLayout() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
  if (_grid) { deactivate(); activate(); }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/**
 * The part of a topic path worth showing under the card's name.
 *
 * The name is only the last segment, so two cameras both read `image` and the
 * path is what tells them apart — but it was CSS-truncated from the right, which
 * cuts off exactly the segments that differ. `/remote_control/image/depth` and
 * `/remote_control/image/depth_summary` both displayed as `/remote_control/…`.
 * Keeping the tail keeps the distinction; the full path stays in the tooltip.
 */
function _displayPath(topicPath) {
  const segs = String(topicPath || '').split('/').filter(Boolean);
  if (segs.length <= 2) return '/' + segs.join('/');
  return '…/' + segs.slice(-2).join('/');
}

function _formatClass(format) {
  if (!format) return '';
  if (format.includes('audio') || format === 'audio/pcm') return 'fmt-audio';
  if (format.includes('json') || format.startsWith('text/')) return 'fmt-json';
  if (format.includes('video') || format.includes('image') || format.includes('visual')) return 'fmt-visual';
  return '';
}

function _formatLabel(format) {
  if (!format) return 'RAW';
  if (format === 'data/json') return 'JSON';
  if (format === 'audio/pcm') return 'AUDIO';
  if (format.includes('video')) return 'VIDEO';
  if (format.includes('image')) return 'IMAGE';
  if (format.startsWith('text/')) return 'TEXT';
  return format.split('/').pop()?.toUpperCase() || 'RAW';
}
