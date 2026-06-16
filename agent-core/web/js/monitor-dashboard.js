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
import { PointCloudRenderer } from './renderers/pointcloud.js';
import { MappingRenderer }   from './renderers/mapping.js';
import { SkeletonRenderer } from './renderers/skeleton.js';
import { KvLatestRenderer } from './renderers/kv-latest.js';
import { CameraRenderer, DepthRenderer } from './renderers/camera.js';
import { HTMSGRenderer }    from './renderers/htmsg.js';

const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, ImageRenderer, AudioRenderer, PointCloudRenderer, MappingRenderer, LidarRenderer, HTMSGRenderer, SkeletonRenderer, TextRenderer, ActivityRenderer];
const STORAGE_KEY = 'monitor-dashboard-layout-v2';
const CELL_SIZE = 280;  // minimum px per grid cell
const GAP = 12;         // px gap between cells
const GRID_COLS = 5;    // fixed 5 columns
let _topicMcpMap = {};  // topic → mcpId, populated on fetch

let _grid = null;
let _cards = new Map(); // topicPath → { el, renderer, ws, format, mode, col, row, colSpan, rowSpan }
let _totalCols = GRID_COLS;

export function activate() {
  _grid = document.getElementById('monitor-dashboard-grid');
  _grid.innerHTML = '';
  _cards.clear();
  _totalCols = GRID_COLS;
  _applyGridStyle();
  _fetchAndBuild();
}

export function deactivate() {
  for (const card of _cards.values()) {
    card.ws?.close();
    card.renderer?.unmount?.();
  }
  _cards.clear();
  if (_grid) _grid.innerHTML = '';
}

function _applyGridStyle() {
  if (!_grid) return;
  _grid.style.gridTemplateColumns = `repeat(${_totalCols}, 1fr)`;
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

  const topicSet = new Set();
  _topicMcpMap = {};  // reset
  // First pass: collect all topic_out
  for (const mcp of mcps) {
    const mcpOnCanvas = canvasCards.some(c => c.mcpId === mcp.id);
    if (!mcpOnCanvas) continue;
    for (const tool of (mcp.tools || [])) {
      if (!canvasTools.has(`${mcp.id}:${tool.name}`)) continue;
      for (const t of (tool.topic_out || [])) { if (t.topic) { topicSet.add(t.topic); _topicMcpMap[t.topic] = mcp.id; } }
    }
  }
  // Second pass: add topic_in only if not already covered by topic_out (avoid duplicates)
  for (const mcp of mcps) {
    const mcpOnCanvas = canvasCards.some(c => c.mcpId === mcp.id);
    if (!mcpOnCanvas) continue;
    for (const tool of (mcp.tools || [])) {
      if (!canvasTools.has(`${mcp.id}:${tool.name}`)) continue;
      for (const t of (tool.topic_in || [])) { if (t.topic && !topicSet.has(t.topic)) { topicSet.add(t.topic); _topicMcpMap[t.topic] = mcp.id; } }
    }
  }
  // Dynamic topics from canvas connections
  for (const conn of connections) {
    if (conn.fromTopic) topicSet.add(conn.fromTopic);
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

  // Create cards with positions (avoid overlap)
  const placed = [];
  for (const topicPath of topics) {
    const detail = topicDetails[topicPath];
    const format = detail?.format || 'activity';
    const status = detail?.status || 'offline';
    const saved = savedLayout[topicPath];

    const colSpan = saved?.colSpan || 1;
    const rowSpan = saved?.rowSpan || 2;

    let col, row;
    if (saved?.col != null && saved?.row != null) {
      col = saved.col;
      row = saved.row;
    } else {
      ({ col, row } = _findFreeSlot(placed, colSpan, rowSpan));
    }

    const savedMode = saved?.mode || 'log';
    _createCard(topicPath, format, status, col, row, colSpan, rowSpan, savedMode);
    placed.push({ col, row, colSpan, rowSpan });
  }
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
      <div class="monitor-card-dot ${status}"></div>
      <div class="monitor-card-names">
        <span class="monitor-card-topic">${shortName}</span>
        <span class="monitor-card-path" title="${topicPath}">${topicPath}</span>
      </div>
      ${modeHtml}
      <span class="monitor-card-format ${fmtClass}">${_formatLabel(format)}</span>
    </div>
    <div class="monitor-card-body"></div>
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

  // Drag (header)
  const header = el.querySelector('.monitor-card-header');
  header.addEventListener('mousedown', (e) => _startDrag(e, topicPath));

  // Resize handle
  const resizeHandle = el.querySelector('.monitor-card-resize');
  resizeHandle.addEventListener('mousedown', (e) => _startResize(e, topicPath));

  _grid.appendChild(el);
}

function _applyPlacement(el, col, row, colSpan, rowSpan) {
  el.style.gridColumn = `${col + 1} / span ${colSpan}`;
  el.style.gridRow = `${row + 1} / span ${rowSpan}`;
}

function _connectWs(topicPath, format, renderer) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${location.host}/ws/bus${topicPath}`;

  function create() {
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (ev) => _handleWsMessage(ev, renderer, format);
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

function _handleWsMessage(ev, renderer, format) {
  if (ev.data instanceof ArrayBuffer) {
    if (ev.data.byteLength === 0) return;
    renderer.onData?.(ev.data, format);
  } else {
    try {
      const parsed = JSON.parse(ev.data);
      if (parsed.type === 'ping' || parsed.type === 'meta' || parsed.type === 'error') return;
    } catch {}
    const buf = new TextEncoder().encode(ev.data).buffer;
    renderer.onData?.(buf, format);
  }
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
  // Re-wire WS
  card.ws.onmessage = (ev) => _handleWsMessage(ev, card.renderer, card.format);
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

// ── Drag to reposition (snap to grid cell) ────────────────────────────────────

let _dragging = null;

function _startDrag(e, topicPath) {
  if (e.target.closest('.monitor-card-modes') || e.target.closest('.monitor-card-resize')) return;
  e.preventDefault();
  const card = _cards.get(topicPath);
  if (!card) return;

  const gridRect = _grid.getBoundingClientRect();
  const cellH = gridRect.height * 0.2 + GAP;
  _dragging = {
    topicPath, card,
    startX: e.clientX, startY: e.clientY,
    origCol: card.col, origRow: card.row,
    gridLeft: gridRect.left, gridTop: gridRect.top,
    cellW: gridRect.width / _totalCols,
    cellH,
    valid: true,
  };
  card.el.classList.add('dragging');
  _showGridOverlay();

  document.addEventListener('mousemove', _onDragMove);
  document.addEventListener('mouseup', _onDragEnd);
}

function _onDragMove(e) {
  if (!_dragging) return;
  const { card, cellW, cellH, topicPath } = _dragging;
  const dx = e.clientX - _dragging.startX;
  const dy = e.clientY - _dragging.startY;

  const colDelta = Math.round(dx / cellW);
  const rowDelta = Math.round(dy / cellH);

  const newCol = Math.max(0, Math.min(_totalCols - card.colSpan, _dragging.origCol + colDelta));
  const newRow = Math.max(0, _dragging.origRow + rowDelta);

  if (newCol !== card.col || newRow !== card.row) {
    card.col = newCol;
    card.row = newRow;
    _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
  }

  // Check collision
  const hasCollision = _checkCollision(topicPath, newCol, newRow, card.colSpan, card.rowSpan);
  _dragging.valid = !hasCollision;
  card.el.classList.toggle('drag-invalid', hasCollision);
}

function _onDragEnd() {
  if (!_dragging) return;
  const { card, topicPath, valid } = _dragging;

  if (!valid) {
    // Revert to original position
    card.col = _dragging.origCol;
    card.row = _dragging.origRow;
    _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
  }

  card.el.classList.remove('dragging', 'drag-invalid');
  _dragging = null;
  _hideGridOverlay();
  document.removeEventListener('mousemove', _onDragMove);
  document.removeEventListener('mouseup', _onDragEnd);

  if (valid) _saveLayout();
}

/** Check if placing a card at (col, row) with (colSpan, rowSpan) overlaps any other card */
function _checkCollision(excludeTopic, col, row, colSpan, rowSpan) {
  for (const [topic, other] of _cards) {
    if (topic === excludeTopic) continue;
    // Check rectangle overlap
    if (col < other.col + other.colSpan &&
        col + colSpan > other.col &&
        row < other.row + other.rowSpan &&
        row + rowSpan > other.row) {
      return true;
    }
  }
  return false;
}

// ── Resize (snap to grid cells) ───────────────────────────────────────────────

let _resizing = null;

function _startResize(e, topicPath) {
  e.preventDefault();
  e.stopPropagation();
  const card = _cards.get(topicPath);
  if (!card) return;

  const gridRect = _grid.getBoundingClientRect();
  const cellH = gridRect.height * 0.2 + GAP;
  _resizing = {
    topicPath, card,
    startX: e.clientX, startY: e.clientY,
    origCol: card.colSpan, origRow: card.rowSpan,
    cellW: gridRect.width / _totalCols,
    cellH,
  };
  card.el.classList.add('resizing');
  _showGridOverlay();

  document.addEventListener('mousemove', _onResizeMove);
  document.addEventListener('mouseup', _onResizeEnd);
}

function _onResizeMove(e) {
  if (!_resizing) return;
  const { card, cellW, cellH, topicPath } = _resizing;
  const dx = e.clientX - _resizing.startX;
  const dy = e.clientY - _resizing.startY;

  const maxCol = _totalCols - card.col;
  const newColSpan = Math.max(1, Math.min(maxCol, _resizing.origCol + Math.round(dx / cellW)));
  const newRowSpan = Math.max(1, Math.min(5, _resizing.origRow + Math.round(dy / cellH)));

  if (newColSpan !== card.colSpan || newRowSpan !== card.rowSpan) {
    card.colSpan = newColSpan;
    card.rowSpan = newRowSpan;
    _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
  }

  // Check collision
  const hasCollision = _checkCollision(topicPath, card.col, card.row, newColSpan, newRowSpan);
  _resizing.valid = !hasCollision;
  card.el.classList.toggle('drag-invalid', hasCollision);
}

function _onResizeEnd() {
  if (!_resizing) return;
  const { topicPath, card } = _resizing;
  const valid = _resizing.valid !== false;

  if (!valid) {
    // Revert to original size
    card.colSpan = _resizing.origCol;
    card.rowSpan = _resizing.origRow;
    _applyPlacement(card.el, card.col, card.row, card.colSpan, card.rowSpan);
  }

  card.el.classList.remove('resizing', 'drag-invalid');
  _resizing = null;
  _hideGridOverlay();
  document.removeEventListener('mousemove', _onResizeMove);
  document.removeEventListener('mouseup', _onResizeEnd);

  if (valid) {
    _saveLayout();
    _refreshRenderer(topicPath);
  }
}

// ── Grid Overlay (visual guides during drag/resize) ───────────────────────────

let _overlay = null;

function _showGridOverlay() {
  if (_overlay) return;
  _overlay = document.createElement('div');
  _overlay.className = 'monitor-grid-overlay';
  // Create grid cells for visual reference
  const rows = Math.ceil((_grid.scrollHeight || 600) / (CELL_SIZE + GAP)) + 2;
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
