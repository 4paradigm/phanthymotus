/**
 * canvas.js — Orchestration canvas with zoom/pan support.
 *
 * Architecture:
 *   #canvas-area  (overflow:hidden, captures wheel/pointer events)
 *     └─ #canvas-viewport  (transform: translate(tx,ty) scale(zoom))
 *          └─ .canvas-card  (positioned absolute, in world-space coords)
 *
 * Zoom: mouse wheel (centered on cursor), +/− buttons
 * Pan:  middle-button drag OR space+drag
 * Cards: pointer-capture drag within viewport (world coords)
 */

import { showTopicDetail } from './detail-panel.js';
import { showToolDetail, isToolConfigured, isInstanceConfigured, openInstanceConfigModal, hasSharedRequired } from './sidebar.js';
import { toggleMicStream, isMicActive } from './mic-stream.js';

let _canvasEl   = null;
let _viewport   = null;
let _emptyEl    = null;
let _zoomLabel  = null;
let _connSvg    = null;
let _cards      = [];   // [{ id, mcpId, toolName, driverName, x, y, el }]
let _allMcps    = [];

// Connection state
let _connections = [];  // [{id, fromCardId, fromPort, toCardId, toPort, format}]
let _execConnections = []; // [{id, fromCardId, toCardId, toToolName, toMcpId}]
let _draggingConn = null; // {fromCardId, fromPortEl, format, topic, tempPath, type?}

// Project run state
let _projectRunning = false;

export function isProjectRunning() { return _projectRunning; }

// ── Viewport transform state ──────────────────────────────────────────────────
let _zoom = 1;
let _tx   = 0;
let _ty   = 0;

const ZOOM_MIN  = 0.25;
const ZOOM_MAX  = 2.5;
const ZOOM_STEP = 0.1;

// ── Init ─────────────────────────────────────────────────────────────────────

export async function initCanvas(initialMcps) {
  _canvasEl  = document.getElementById('canvas-area');
  _viewport  = document.getElementById('canvas-viewport');
  _emptyEl   = document.getElementById('canvas-empty');
  _zoomLabel = document.getElementById('canvas-zoom-label');
  _connSvg   = document.getElementById('canvas-connectors-svg');
  if (!_canvasEl || !_viewport) return;

  if (initialMcps) _allMcps = initialMcps;

  _setupZoomPan();
  _setupDropZone();
  _setupControlButtons();
  _setupPortDrag();

  // Load persisted layout
  try {
    const layoutRes = await fetch('/api/canvas/layout');
    const layoutJson = await layoutRes.json();

    const saved = layoutJson.data?.cards || [];
    for (const c of saved) {
      _addCard(c, false);
    }
    // Restore connections — filter out any that reference cards no longer in the layout
    const cardIds = new Set(_cards.map(c => c.id));
    _connections = (layoutJson.data?.connections || []).filter(
      c => cardIds.has(c.fromCardId) && cardIds.has(c.toCardId)
    );
    _execConnections = (layoutJson.data?.execConnections || []).filter(
      c => cardIds.has(c.fromCardId) && cardIds.has(c.toCardId)
    );
    _resolveAllTopics();
    _redrawConnections();
    // Restore viewport transform if saved
    if (layoutJson.data?.transform) {
      _zoom = layoutJson.data.transform.zoom ?? 1;
      _tx   = layoutJson.data.transform.tx   ?? 0;
      _ty   = layoutJson.data.transform.ty   ?? 0;
      _applyTransform();
    }
  } catch { /* start empty */ }

  // Restore project running state from backend
  try {
    const runRes = await fetch('/api/config/project-running');
    const runData = await runRes.json();
    if (runData.running) {
      _projectRunning = true;
      _syncProjectBtn();
      document.querySelectorAll('.canvas-exec-btn').forEach(btn => btn.classList.remove('locked'));
    }
  } catch { /* ignore */ }

  _syncEmptyState();
}

export function updateCanvasMcps(mcps) {
  _allMcps = mcps || [];
  let topicsChanged = false;
  for (const card of _cards) {
    const mcp = _allMcps.find(m => m.id === card.mcpId);
    if (!mcp) continue;
    const nameEl = card.el.querySelector('.canvas-card-driver');
    if (nameEl) nameEl.textContent = mcp.server_name || mcp.name || mcp.id;

    // Update persisted topics from live tool data (when driver comes online)
    const tools = mcp.tools || [];
    const toolObj = tools.find(t => (typeof t === 'string' ? t : t.name) === card.toolName);
    const liveTopicIn  = typeof toolObj === 'object' ? toolObj.topic_in  : null;
    const liveTopicOut = typeof toolObj === 'object' ? toolObj.topic_out : null;
    if (liveTopicIn  && liveTopicIn.length  && JSON.stringify(liveTopicIn)  !== JSON.stringify(card.topicIn))  { card.topicIn  = liveTopicIn;  topicsChanged = true; }
    if (liveTopicOut && liveTopicOut.length && JSON.stringify(liveTopicOut) !== JSON.stringify(card.topicOut)) { card.topicOut = liveTopicOut; topicsChanged = true; }
  }
  if (topicsChanged) {
    // Rebuild cards that have new port counts
    for (const card of _cards) {
      const newEl = _buildCardEl({ id: card.id, mcpId: card.mcpId, toolName: card.toolName, driverName: card.driverName, x: card.x, y: card.y, topicIn: card.topicIn, topicOut: card.topicOut });
      card.el.replaceWith(newEl);
      card.el = newEl;
      _makeDraggable(newEl, card);
    }
    _resolveAllTopics();
    _redrawConnections();
    _debouncedSave();
  }
}

// ── Zoom / Pan ────────────────────────────────────────────────────────────────

function _applyTransform() {
  _viewport.style.transform = `translate(${_tx}px, ${_ty}px) scale(${_zoom})`;
  if (_zoomLabel) _zoomLabel.textContent = Math.round(_zoom * 100) + '%';
}

function _zoomAt(clientX, clientY, delta) {
  const rect    = _canvasEl.getBoundingClientRect();
  const mouseX  = clientX - rect.left;
  const mouseY  = clientY - rect.top;

  // World coords under cursor before zoom
  const worldX  = (mouseX - _tx) / _zoom;
  const worldY  = (mouseY - _ty) / _zoom;

  _zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, _zoom + delta));

  // Adjust translation so world point stays under cursor
  _tx = mouseX - worldX * _zoom;
  _ty = mouseY - worldY * _zoom;

  _applyTransform();
}

function _setupZoomPan() {
  // Wheel zoom (centered on cursor)
  _canvasEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
    _zoomAt(e.clientX, e.clientY, delta);
    _debouncedSave();
  }, { passive: false });

  // Left-click drag on canvas background = pan (like a map)
  let _panning    = false;
  let _panStartX  = 0;
  let _panStartY  = 0;
  let _panStartTx = 0;
  let _panStartTy = 0;

  _canvasEl.addEventListener('pointerdown', (e) => {
    // Only pan when clicking directly on canvas-area or canvas-viewport (not on a card)
    const isBackground = e.target === _canvasEl || e.target === _viewport || e.target === _emptyEl;
    if (!isBackground || e.button !== 0) return;

    e.preventDefault();
    _panning    = true;
    _panStartX  = e.clientX;
    _panStartY  = e.clientY;
    _panStartTx = _tx;
    _panStartTy = _ty;
    _canvasEl.setPointerCapture(e.pointerId);
    _canvasEl.style.cursor = 'grabbing';
  });

  _canvasEl.addEventListener('pointermove', (e) => {
    if (!_panning) return;
    _tx = _panStartTx + (e.clientX - _panStartX);
    _ty = _panStartTy + (e.clientY - _panStartY);
    _applyTransform();
  });

  _canvasEl.addEventListener('pointerup', () => {
    if (!_panning) return;
    _panning = false;
    _canvasEl.style.cursor = '';
    _debouncedSave();
  });

  _canvasEl.addEventListener('pointercancel', () => {
    _panning = false;
    _canvasEl.style.cursor = '';
  });
}

function _setupControlButtons() {
  const rect = _canvasEl?.getBoundingClientRect() ?? { left: 0, top: 0, width: 800, height: 600 };
  const cx = (rect.width  || 800) / 2;
  const cy = (rect.height || 600) / 2;

  document.getElementById('canvas-zoom-in')?.addEventListener('click', () => {
    const r = _canvasEl.getBoundingClientRect();
    _zoomAt(r.left + r.width / 2, r.top + r.height / 2, ZOOM_STEP);
    _debouncedSave();
  });

  document.getElementById('canvas-zoom-out')?.addEventListener('click', () => {
    const r = _canvasEl.getBoundingClientRect();
    _zoomAt(r.left + r.width / 2, r.top + r.height / 2, -ZOOM_STEP);
    _debouncedSave();
  });

  document.getElementById('canvas-zoom-reset')?.addEventListener('click', () => {
    _zoom = 1; _tx = 0; _ty = 0;
    _applyTransform();
    _debouncedSave();
  });

  document.getElementById('canvas-project-toggle')?.addEventListener('click', () => {
    _projectRunning ? _stopProject() : _startProject();
  });
  _syncProjectBtn();
}

// ── Drop zone ─────────────────────────────────────────────────────────────────

function _setupDropZone() {
  _canvasEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    _canvasEl.classList.add('drag-over');
  });

  _canvasEl.addEventListener('dragleave', (e) => {
    if (!_canvasEl.contains(e.relatedTarget)) {
      _canvasEl.classList.remove('drag-over');
    }
  });

  _canvasEl.addEventListener('drop', (e) => {
    e.preventDefault();
    _canvasEl.classList.remove('drag-over');

    if (_projectRunning) {
      _showDropReject(e, '请停止智能控制后修改');
      return;
    }

    let data;
    try {
      data = JSON.parse(e.dataTransfer.getData('application/x-cap-card'));
    } catch { return; }

    // Prevent unconfigured tools from being added
    if (data.hasConfig && !isToolConfigured(data.mcpId, data.toolName)) {
      _showDropReject(e, '请先配置后再使用');
      return;
    }

    // Prevent same tool from being added twice (unless multiInstance)
    if (!data.multiInstance) {
      const existing = _cards.find(c => c.mcpId === data.mcpId && c.toolName === data.toolName);
      if (existing) {
        _showDropReject(e, '不能两次加入同样的组件');
        return;
      }
    }

    // Convert screen coords → world coords
    const rect   = _canvasEl.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    let x = (screenX - _tx) / _zoom - 110;
    let y = (screenY - _ty) / _zoom - 24;

    // Avoid overlapping existing cards
    ({ x, y } = _findNonOverlappingPos(x, y));

    const id = 'card-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    _addCard({ id, mcpId: data.mcpId, toolName: data.toolName, driverName: data.driverName, x, y }, true);
  });
}

// ── Drop rejection feedback ──────────────────────────────────────────────────

function _showDropReject(e, reason) {
  const tip = document.createElement('div');
  tip.className = 'canvas-drop-reject';
  tip.textContent = reason;
  tip.style.left = `${e.clientX}px`;
  tip.style.top  = `${e.clientY}px`;
  document.body.appendChild(tip);
  requestAnimationFrame(() => tip.classList.add('show'));
  setTimeout(() => { tip.classList.remove('show'); setTimeout(() => tip.remove(), 200); }, 1800);
}

// ── Overlap avoidance ─────────────────────────────────────────────────────────

const CARD_W = 260, CARD_H = 140, CARD_GAP = 20;

function _findNonOverlappingPos(x, y) {
  const maxAttempts = 50;
  for (let i = 0; i < maxAttempts; i++) {
    const overlaps = _cards.some(c =>
      Math.abs(c.x - x) < CARD_W + CARD_GAP &&
      Math.abs(c.y - y) < CARD_H + CARD_GAP
    );
    if (!overlaps) return { x, y };
    // Shift right, wrap down after 4 attempts in same row
    x += CARD_W + CARD_GAP;
    if ((i + 1) % 4 === 0) {
      x -= 4 * (CARD_W + CARD_GAP);
      y += CARD_H + CARD_GAP;
    }
  }
  return { x, y };
}

// ── Card management ───────────────────────────────────────────────────────────

function _addCard(data, save = true) {
  const { id, mcpId, toolName, x, y } = data;
  let { driverName } = data;

  if (!driverName) {
    const mcp = _allMcps.find(m => m.id === mcpId);
    driverName = mcp ? (mcp.server_name || mcp.name || mcp.id) : mcpId;
  }

  const el = _buildCardEl({ id, mcpId, toolName, driverName, x, y, topicIn: data.topicIn, topicOut: data.topicOut });
  _viewport.appendChild(el);

  // Restore or initialize persisted topic data
  let topicInData  = data.topicIn  || [];
  let topicOutData = data.topicOut || [];
  if (!topicInData.length || !topicOutData.length) {
    // Try to initialize from current MCP tool data (for newly dropped cards)
    const _mcp = _allMcps.find(m => m.id === mcpId);
    const _tools = _mcp?.tools || [];
    const _toolObj = _tools.find(t => (typeof t === 'string' ? t : t.name) === toolName);
    if (typeof _toolObj === 'object') {
      if (!topicInData.length  && _toolObj.topic_in)  topicInData  = _toolObj.topic_in;
      if (!topicOutData.length && _toolObj.topic_out) topicOutData = _toolObj.topic_out;
    }
  }

  const cardData = { id, mcpId, toolName, driverName, x, y, el, topicIn: topicInData, topicOut: topicOutData };
  _cards.push(cardData);
  _makeDraggable(el, cardData);
  _syncEmptyState();

  if (save) _saveLayout();
}

function _removeCard(id) {
  if (_projectRunning) {
    _logActivity('warn', '请停止智能控制后修改');
    return;
  }
  const idx = _cards.findIndex(c => c.id === id);
  if (idx === -1) return;
  _cards[idx].el.remove();
  _cards.splice(idx, 1);
  // Trigger stop for connections where this card was the source
  const outgoing = _connections.filter(c => c.fromCardId === id);
  // Clean up topic connections
  _connections = _connections.filter(c => c.fromCardId !== id && c.toCardId !== id);
  // Trigger auto-stop on downstream cards that lost their input
  for (const conn of outgoing) {
    _autoStopOnDisconnect(conn.toCardId, conn.toPortIdx, conn.fromTopic);
  }
  // Clean up executor connections
  _execConnections = _execConnections.filter(c => c.fromCardId !== id && c.toCardId !== id);
  _resolveAllTopics();
  _redrawConnections();
  _syncEmptyState();
  // Cancel any pending debounced save, then save immediately with updated state
  clearTimeout(_saveTimer);
  _saveLayout();
}

// ── Card rendering ────────────────────────────────────────────────────────────

function _buildCardEl({ id, mcpId, toolName, driverName, x, y, topicIn: savedTopicIn, topicOut: savedTopicOut }) {
  const el = document.createElement('div');
  el.dataset.cardId = id;
  el.style.left = x + 'px';
  el.style.top  = y + 'px';

  const mcp     = _allMcps.find(m => m.id === mcpId);
  const tools   = mcp?.tools || [];
  const toolObj = tools.find(t => (typeof t === 'string' ? t : t.name) === toolName);
  const schema  = typeof toolObj === 'object' ? toolObj.inputSchema : null;
  const toolType = (typeof toolObj === 'object' ? toolObj.type : '') || '';
  const configSchema = typeof toolObj === 'object' ? toolObj.configSchema : null;
  const hasInstanceFields = configSchema && Object.values(configSchema.properties || {}).some(d => d.scope === 'instance');

  // Auto-classify perception tools as processor
  // Priority: tool-level (live) > persisted card-level > single-tool MCP fallback
  // For bundle MCPs (multiple tools), never fall back to MCP aggregate topics.
  const toolTopicIn  = typeof toolObj === 'object' ? toolObj.topic_in  : null;
  const toolTopicOut = typeof toolObj === 'object' ? toolObj.topic_out : null;
  const isBundleMcp = (mcp?.tools || []).length > 1;
  const topicIn  = toolTopicIn  || (savedTopicIn?.length  ? savedTopicIn  : (toolType || isBundleMcp ? [] : mcp?.topic_in  || []));
  const topicOut = toolTopicOut || (savedTopicOut?.length ? savedTopicOut : (toolType || isBundleMcp ? [] : mcp?.topic_out || []));
  const effectiveType = toolType || (topicIn.length && topicOut.length ? 'processor' : topicOut.length ? 'sensor' : topicIn.length ? 'actuator' : '');

  el.className = `canvas-card${effectiveType ? ' ' + effectiveType : ''}`;

  const typeBadge = effectiveType ? `<span class="cap-type-badge ${_esc(effectiveType)}">${_esc(effectiveType)}</span>` : '';

  // Build port HTML
  const inPortsHtml = topicIn.map((t, i) => {
    const fmt = t.format || '';
    const fmtShort = fmt.split('/').pop() || '?';
    const colorCls = _fmtColorClass(fmt);
    return `<div class="canvas-port in ${colorCls}" data-dir="in" data-format="${_esc(fmt)}" data-topic="${_esc(t.topic || '')}" data-idx="${i}" title="${_esc(fmt)}"><span class="canvas-port-label">${_esc(fmtShort)}</span></div>`;
  }).join('');

  const outPortsHtml = topicOut.map((t, i) => {
    const fmt = t.format || '';
    const fmtShort = fmt.split('/').pop() || '?';
    const colorCls = _fmtColorClass(fmt);
    const staticAttr = t.topic ? `data-static-topic="${_esc(t.topic)}"` : '';
    return `<div class="canvas-port out ${colorCls}" data-dir="out" data-format="${_esc(fmt)}" data-topic="${_esc(t.topic || '')}" ${staticAttr} data-idx="${i}" title="${_esc(fmt)}"><span class="canvas-port-label">${_esc(fmtShort)}</span></div>`;
  }).join('');

  if (effectiveType === 'controller') {
    // Controller cards: no fields, no execute button — only start/stop/info via header
    el.innerHTML = `
      <div class="canvas-card-body-wrap">
        <div class="canvas-card-header">
          <div class="canvas-card-info">
            <div class="canvas-card-tool" title="${_esc(toolName)}">${typeBadge} ${_esc(toolName)}</div>
            <div class="canvas-card-driver" title="${_esc(driverName)}">${_esc(driverName)}</div>
          </div>
          <button class="tool-card-info-btn canvas-card-info-btn" title="详情"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></button>
          <button class="canvas-card-close" title="从画布移除">✕</button>
        </div>
      </div>
      <div class="canvas-port-col left">${inPortsHtml}</div>
      <div class="canvas-port-col right">${outPortsHtml}</div>
      <div class="canvas-port-col bottom"><div class="canvas-port executor" data-dir="executor" data-format="executor" title="连接执行器"><span class="canvas-port-label">执行器</span></div></div>
    `;

    el.querySelector('.canvas-card-close').addEventListener('click', (e) => {
      e.stopPropagation();
      _removeCard(id);
    });

    el.querySelector('.canvas-card-info-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const liveMcp = _allMcps.find(m => m.id === mcpId);
      if (liveMcp) {
        const liveTopicIn  = _collectInTopics(id, el);
        const liveTopicOut = [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut });
      }
    });
  } else if (effectiveType === 'sensor') {
    // Check if sensor has callable actions beyond start/stop/info/config
    const sensorProps = schema?.properties || {};
    const sensorRequired = schema?.required || [];
    const _SENSOR_SYS_ACTIONS = new Set(['start', 'stop', 'info', 'config']);
    const sensorActionDef = sensorProps.action;
    const hasSensorActions = sensorActionDef?.enum?.some(a => !_SENSOR_SYS_ACTIONS.has(a));

    // Instance config button (for multiInstance sensors with instance-scope fields)
    const sensorInstanceCfgBtn = hasInstanceFields
      ? `<button class="canvas-card-instance-cfg-btn" title="实例配置"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-1.42 3.42 2 2 0 0 1-1.42-.58l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-3.42-1.42 2 2 0 0 1 .58-1.42l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 1.42-3.42 2 2 0 0 1 1.42.58l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1.08 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 3.42 1.42 2 2 0 0 1-.58 1.42l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1.08z"/></svg></button>`
      : '';

    let sensorFieldsHtml = '';
    if (hasSensorActions) {
      sensorFieldsHtml = Object.entries(sensorProps).map(([key, def]) => {
        const isReq = sensorRequired.includes(key);
        const label = key + (isReq ? ' *' : '');
        let inputHtml;
        if (def.enum) {
          const enumVals = key === 'action' ? def.enum.filter(v => !_SENSOR_SYS_ACTIONS.has(v)) : def.enum;
          if (!enumVals.length) return '';
          const opts = enumVals.map(v => `<option value="${_esc(v)}">${_esc(v)}</option>`).join('');
          inputHtml = `<select class="canvas-field-input" data-key="${_esc(key)}">${opts}</select>`;
        } else {
          const type = def.type === 'number' || def.type === 'integer' ? 'number' : 'text';
          const desc = def.description || '';
          inputHtml = `<input class="canvas-field-input" type="${type}" data-key="${_esc(key)}" placeholder="${_esc(desc.slice(0, 40))}">`;
        }
        return `
          <div class="canvas-field">
            <label class="canvas-field-label" title="${_esc(def.description || '')}">${_esc(label)}</label>
            ${inputHtml}
          </div>`;
      }).join('');
    }

    el.innerHTML = `
      <div class="canvas-card-body-wrap">
        <div class="canvas-card-header">
          <div class="canvas-card-info">
            <div class="canvas-card-tool" title="${_esc(toolName)}">${typeBadge} ${_esc(toolName)}</div>
            <div class="canvas-card-driver" title="${_esc(driverName)}">${_esc(driverName)}</div>
          </div>
          ${sensorInstanceCfgBtn}
          <button class="tool-card-info-btn canvas-card-info-btn" title="详情"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></button>
          <button class="canvas-card-close" title="从画布移除">✕</button>
        </div>
        ${sensorFieldsHtml ? `<div class="canvas-card-body">${sensorFieldsHtml}</div>` : ''}
        <div class="canvas-card-footer" style="padding:8px 10px">
          ${hasSensorActions ? `<button class="canvas-exec-btn${_projectRunning ? '' : ' locked'}">▶ 执行</button>` : ''}
          ${hasSensorActions ? '<hr class="canvas-footer-divider">' : ''}
          <button class="canvas-view-btn">📡 查看数据流</button>
        </div>
      </div>
      <div class="canvas-port-col left">${inPortsHtml}</div>
      <div class="canvas-port-col right">${outPortsHtml}</div>
    `;

    el.querySelector('.canvas-card-close').addEventListener('click', (e) => {
      e.stopPropagation();
      _removeCard(id);
    });

    el.querySelector('.canvas-card-info-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const liveMcp = _allMcps.find(m => m.id === mcpId);
      if (liveMcp) {
        const liveTopicIn  = _collectInTopics(id, el);
        const liveTopicOut = [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut });
      }
    });

    // Instance config button (multiInstance sensors with instance-scope fields)
    const sensorInstanceCfgBtnEl = el.querySelector('.canvas-card-instance-cfg-btn');
    if (sensorInstanceCfgBtnEl) {
      sensorInstanceCfgBtnEl.addEventListener('click', (e) => {
        e.stopPropagation();
        // Re-lookup configSchema at click time to avoid stale closure
        const liveMcp2 = _allMcps.find(m => m.id === mcpId);
        const liveToolObj2 = (liveMcp2?.tools || []).find(t => (typeof t === 'string' ? t : t.name) === toolName);
        const liveConfigSchema = typeof liveToolObj2 === 'object' ? liveToolObj2.configSchema : null;
        openInstanceConfigModal(mcpId, toolName, id, liveConfigSchema || configSchema);
      });
    }

    el.querySelector('.canvas-view-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const liveMcp = _allMcps.find(m => m.id === mcpId);
      const topics = topicOut.length ? topicOut : (liveMcp?.topic_out || []);
      if (topics.length) showTopicDetail(topics[0].topic, topics[0].format || '');
    });

    const sensorExecBtn = el.querySelector('.canvas-exec-btn');
    if (sensorExecBtn) {
      sensorExecBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await _executeCard(el, mcpId, toolName);
      });
    }
  } else {
    // Actuator/processor/default card
    const props   = schema?.properties || {};
    const required = schema?.required || [];

    const _SYSTEM_ACTIONS = new Set(['start', 'stop', 'info', 'config']);
    const _TOPIC_KEY_RE = /input.*topic|topic.*in|output.*topic|topic.*out/i;
    const fieldsHtml = Object.entries(props).map(([key, def]) => {
      // Hide auto-populated topic fields for processor cards
      if ((effectiveType === 'processor' || effectiveType === 'actuator') && _TOPIC_KEY_RE.test(key)) return '';
      const isReq = required.includes(key);
      const label = key + (isReq ? ' *' : '');
      let inputHtml;
      if (def.enum) {
        // Filter system actions from processor cards
        let enumVals = def.enum;
        if (key === 'action') {
          enumVals = enumVals.filter(v => !_SYSTEM_ACTIONS.has(v));
        }
        if (!enumVals.length) return '';  // hide field entirely if no options left
        const opts = enumVals.map(v => `<option value="${_esc(v)}">${_esc(v)}</option>`).join('');
        inputHtml = `<select class="canvas-field-input" data-key="${_esc(key)}">${opts}</select>`;
      } else {
        const type = def.type === 'number' || def.type === 'integer' ? 'number' : 'text';
        const desc = def.description || '';
        inputHtml = `<input class="canvas-field-input" type="${type}" data-key="${_esc(key)}" placeholder="${_esc(desc.slice(0, 40))}">`;
      }
      return `
        <div class="canvas-field">
          <label class="canvas-field-label" title="${_esc(def.description || '')}">${_esc(label)}</label>
          ${inputHtml}
        </div>`;
    }).join('');

    // Controller gets an additional bottom executor port
    const executorPortHtml = effectiveType === 'controller'
      ? `<div class="canvas-port-col bottom"><div class="canvas-port executor" data-dir="executor" data-format="executor" title="连接执行器"><span class="canvas-port-label">执行器</span></div></div>`
      : '';

    // Determine if there are any usable fields/actions left
    const hasUsableFields = fieldsHtml.replace(/\s/g, '').length > 0;

    // Processor cards get a "查看数据流" button if they have output topics
    const showViewBtn = effectiveType === 'processor' && topicOut.length > 0;

    const instanceCfgBtn = hasInstanceFields
      ? `<button class="canvas-card-instance-cfg-btn" title="实例配置"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-1.42 3.42 2 2 0 0 1-1.42-.58l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-3.42-1.42 2 2 0 0 1 .58-1.42l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 1.42-3.42 2 2 0 0 1 1.42.58l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1.08 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 3.42 1.42 2 2 0 0 1-.58 1.42l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1.08z"/></svg></button>`
      : '';

    el.innerHTML = `
      <div class="canvas-card-body-wrap">
        <div class="canvas-card-header">
          <div class="canvas-card-info">
            <div class="canvas-card-tool" title="${_esc(toolName)}">${typeBadge} ${_esc(toolName)}</div>
            <div class="canvas-card-driver" title="${_esc(driverName)}">${_esc(driverName)}</div>
          </div>
          ${instanceCfgBtn}
          <button class="tool-card-info-btn canvas-card-info-btn" title="详情"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></button>
          <button class="canvas-card-close" title="从画布移除">✕</button>
        </div>
        ${fieldsHtml ? `<div class="canvas-card-body">${fieldsHtml}</div>` : ''}
        <div class="canvas-card-footer"${!fieldsHtml ? ' style="padding:8px 10px"' : ''}>
          ${hasUsableFields ? `<button class="canvas-exec-btn${_projectRunning ? '' : ' locked'}">▶ 执行</button>` : ''}
          ${hasUsableFields && showViewBtn ? '<hr class="canvas-footer-divider">' : ''}
          ${showViewBtn ? '<button class="canvas-view-btn">📡 查看数据流</button>' : ''}
        </div>
      </div>
      <div class="canvas-port-col left">${inPortsHtml}</div>
      <div class="canvas-port-col right">${outPortsHtml}</div>
      ${executorPortHtml}
    `;

    el.querySelector('.canvas-card-close').addEventListener('click', (e) => {
      e.stopPropagation();
      _removeCard(id);
    });

    // x-action-params: 根据选中的 action 动态显隐参数字段
    const actionParams = schema?.['x-action-params'];
    if (actionParams) {
      const actionSelect = el.querySelector('.canvas-field-input[data-key="action"]');
      if (actionSelect) {
        const _applyActionParams = () => {
          const selected = actionSelect.value;
          const paramKeys = actionParams[selected]?.params || [];
          el.querySelectorAll('.canvas-field').forEach(field => {
            const key = field.querySelector('.canvas-field-input')?.dataset?.key;
            if (!key || key === 'action') return;
            field.style.display = paramKeys.includes(key) ? '' : 'none';
          });
        };
        actionSelect.addEventListener('change', _applyActionParams);
        _applyActionParams();  // 初始应用
      }
    }

    el.querySelector('.canvas-card-info-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const liveMcp = _allMcps.find(m => m.id === mcpId);
      if (liveMcp) {
        const liveTopicIn  = _collectInTopics(id, el);
        const liveTopicOut = [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut });
      }
    });

    // Instance config button (for multiInstance tools with instance-scope fields)
    const instanceCfgBtnEl = el.querySelector('.canvas-card-instance-cfg-btn');
    if (instanceCfgBtnEl) {
      instanceCfgBtnEl.addEventListener('click', (e) => {
        e.stopPropagation();
        // Re-lookup configSchema at click time to avoid stale closure
        const liveMcp2 = _allMcps.find(m => m.id === mcpId);
        const liveToolObj2 = (liveMcp2?.tools || []).find(t => (typeof t === 'string' ? t : t.name) === toolName);
        const liveConfigSchema = typeof liveToolObj2 === 'object' ? liveToolObj2.configSchema : null;
        openInstanceConfigModal(mcpId, toolName, id, liveConfigSchema || configSchema);
      });
    }

    const execBtn = el.querySelector('.canvas-exec-btn');
    if (execBtn) {
      execBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await _executeCard(el, mcpId, toolName);
      });
    }

    const viewBtn = el.querySelector('.canvas-view-btn');
    if (viewBtn) {
      viewBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const liveMcp = _allMcps.find(m => m.id === mcpId);
        const topics = topicOut.length ? topicOut : (liveMcp?.topic_out || []);
        if (topics.length) showTopicDetail(topics[0].topic, topics[0].format || '');
      });
    }

    // remote_mic 特殊渲染：麦克风录音按钮
    if (toolName === 'remote_mic') {
      const footer = el.querySelector('.canvas-card-footer');
      const micBtn = document.createElement('button');
      micBtn.className = 'canvas-mic-btn';
      micBtn.textContent = isMicActive() ? '\u23F9 停止录音' : '\uD83C\uDF99 开始录音';
      if (isMicActive()) micBtn.classList.add('recording');
      micBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const liveMcp = _allMcps.find(m => m.id === mcpId);
        if (!liveMcp) return;
        // 从 MCP url 推导驱动 WS 地址 (WS on port+1)
        const mcpUrl = new URL(liveMcp.url);
        const wsPort = parseInt(mcpUrl.port) + 1;
        const wsUrl = `ws://${mcpUrl.hostname}:${wsPort}/ws/mic`;
        try {
          await toggleMicStream(wsUrl, (active) => {
            micBtn.textContent = active ? '\u23F9 停止录音' : '\uD83C\uDF99 开始录音';
            micBtn.classList.toggle('recording', active);
          });
        } catch (err) {
          console.error('[canvas] mic toggle failed:', err);
        }
      });
      footer.prepend(micBtn);
    }
  }

  return el;
}

function _fmtColorClass(fmt) {
  if (fmt.startsWith('audio')) return 'fmt-audio';
  if (fmt.startsWith('data/json') || fmt.startsWith('text')) return 'fmt-json';
  if (fmt.startsWith('image') || fmt.startsWith('video')) return 'fmt-visual';
  return 'fmt-default';
}

// ── Config overlay helpers ─────────────────────────────────────────────────

// ── Port drag-to-connect ──────────────────────────────────────────────────────

function _setupPortDrag() {
  document.addEventListener('mousemove', (e) => {
    if (!_draggingConn) return;
    const vpRect = _viewport.getBoundingClientRect();
    const x2 = (e.clientX - vpRect.left) / _zoom;
    const y2 = (e.clientY - vpRect.top) / _zoom;
    const x1 = parseFloat(_draggingConn.tempPath.dataset.x1);
    const y1 = parseFloat(_draggingConn.tempPath.dataset.y1);

    if (_draggingConn.type === 'executor') {
      // Vertical bezier for executor connections
      const cy = Math.max(Math.abs(y2 - y1) * 0.5, 60);
      _draggingConn.tempPath.setAttribute('d', `M${x1},${y1} C${x1},${y1+cy} ${x2},${y2-cy} ${x2},${y2}`);
    } else {
      const cx = Math.abs(x2 - x1) * 0.5;
      _draggingConn.tempPath.setAttribute('d', `M${x1},${y1} C${x1+cx},${y1} ${x2-cx},${y2} ${x2},${y2}`);
    }

    // Card-level hover detection during drag
    const elUnder = document.elementFromPoint(e.clientX, e.clientY);
    const hoverCard = elUnder?.closest('.canvas-card');
    const prevHover = _draggingConn._hoveredCard;

    if (prevHover && prevHover !== hoverCard) {
      prevHover.classList.remove('conn-hover-match', 'conn-hover-mismatch');
      const oldTip = prevHover.querySelector('.conn-hover-tip');
      if (oldTip) oldTip.remove();
    }

    if (hoverCard && hoverCard.dataset.cardId !== _draggingConn.fromCardId) {
      if (_draggingConn.type === 'executor') {
        hoverCard.classList.remove('conn-hover-mismatch');
        hoverCard.classList.add('conn-hover-match');
      } else {
        const hasMatch = hoverCard.querySelector(`.canvas-port.in[data-format="${_draggingConn.format}"]`);
        const isMatch = !!hasMatch;
        hoverCard.classList.remove('conn-hover-match', 'conn-hover-mismatch');
        hoverCard.classList.add(isMatch ? 'conn-hover-match' : 'conn-hover-mismatch');

        // Show / update tooltip
        let tip = hoverCard.querySelector('.conn-hover-tip');
        if (!tip) {
          tip = document.createElement('div');
          tip.className = 'conn-hover-tip';
          hoverCard.appendChild(tip);
        }
        tip.textContent = isMatch ? '数据类型匹配' : '数据类型不匹配';
        tip.classList.toggle('match', isMatch);
        tip.classList.toggle('mismatch', !isMatch);
      }
      _draggingConn._hoveredCard = hoverCard;
    } else if (!hoverCard || hoverCard.dataset.cardId === _draggingConn.fromCardId) {
      _draggingConn._hoveredCard = null;
    }
  });

  document.addEventListener('mouseup', (e) => {
    if (!_draggingConn) return;

    // Remove highlights
    _viewport.querySelectorAll('.canvas-port.port-compatible').forEach(p => p.classList.remove('port-compatible'));
    _viewport.querySelectorAll('.canvas-card.exec-target').forEach(c => c.classList.remove('exec-target'));
    _viewport.querySelectorAll('.canvas-card.conn-hover-match, .canvas-card.conn-hover-mismatch').forEach(c => {
      c.classList.remove('conn-hover-match', 'conn-hover-mismatch');
      const tip = c.querySelector('.conn-hover-tip');
      if (tip) tip.remove();
    });
    _connSvg.classList.remove('dragging-active');

    const target = document.elementFromPoint(e.clientX, e.clientY);

    if (_draggingConn.type === 'executor') {
      // Executor connection: drop on any card (no format matching)
      const toCard = target?.closest('.canvas-card');
      if (toCard && toCard.dataset.cardId !== _draggingConn.fromCardId) {
        const toCardId = toCard.dataset.cardId;
        // Avoid duplicate executor connections
        const dup = _execConnections.some(c => c.fromCardId === _draggingConn.fromCardId && c.toCardId === toCardId);
        if (!dup) {
          const toCardData = _cards.find(c => c.id === toCardId);
          const connId = 'exec-' + Date.now().toString(36);
          _execConnections.push({
            id: connId,
            fromCardId: _draggingConn.fromCardId,
            toCardId: toCardId,
            toToolName: toCardData?.toolName || '',
            toMcpId: toCardData?.mcpId || '',
          });
          _redrawConnections();
          _logActivity('executor', `绑定执行器: ${toCardData?.toolName || toCardId}`);
          _saveLayout();
        }
      }
    } else {
      // Topic connection: drop on compatible in-port (or card-level fallback)
      let inPort = target?.closest('.canvas-port.in');
      let toCard = inPort?.closest('.canvas-card');

      // Fallback: if dropped on card area (not directly on a port), find first matching in-port
      if (!inPort) {
        toCard = target?.closest('.canvas-card');
        if (toCard && toCard.dataset.cardId !== _draggingConn.fromCardId) {
          inPort = toCard.querySelector(`.canvas-port.in[data-format="${_draggingConn.format}"]`);
        }
      }

      if (inPort && inPort.dataset.format === _draggingConn.format && toCard && toCard.dataset.cardId !== _draggingConn.fromCardId) {
        const connId = 'conn-' + Date.now().toString(36);
        _connections.push({
          id: connId,
          fromCardId: _draggingConn.fromCardId,
          fromPortIdx: _draggingConn.fromPortEl.dataset.idx,
          toCardId: toCard.dataset.cardId,
          toPortIdx: inPort.dataset.idx,
          format: _draggingConn.format,
          fromTopic: _draggingConn.topic,
        });

        _resolveAllTopics();
        _redrawConnections();
        _saveLayout();

        const toCardData = _cards.find(c => c.id === toCard.dataset.cardId);
        if (toCardData && _projectRunning) {
          // Use resolved topic from the destination's in-port
          const resolvedInPort = toCard.querySelector(`.canvas-port.in[data-idx="${inPort.dataset.idx}"]`);
          const resolvedTopic = resolvedInPort?.dataset.topic || _draggingConn.topic;
          _triggerAction(toCardData.mcpId, toCardData.toolName, 'start', { input_topic: resolvedTopic, instance_id: toCardData.id });
        }
      }
    }

    // Cleanup
    if (_draggingConn.tempPath) _draggingConn.tempPath.remove();
    _draggingConn = null;
  });

  // Delegate mousedown on out ports and executor ports
  _viewport.addEventListener('mousedown', (e) => {
    const outPort = e.target.closest('.canvas-port.out');
    const execPort = !outPort ? e.target.closest('.canvas-port.executor') : null;
    if (!outPort && !execPort) return;
    if (_projectRunning) {
      _logActivity('warn', '请停止智能控制后修改');
      return;
    }
    e.preventDefault();
    e.stopPropagation();

    const port = outPort || execPort;
    const card = port.closest('.canvas-card');
    if (!card) return;

    const portRect = port.getBoundingClientRect();
    const vpRect = _viewport.getBoundingClientRect();
    const x1 = (portRect.left + portRect.width / 2 - vpRect.left) / _zoom;
    const y1 = (portRect.top + portRect.height / 2 - vpRect.top) / _zoom;

    const isExecutor = !!execPort;
    const tempLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    tempLine.classList.add('connector-temp');
    if (isExecutor) tempLine.classList.add('executor-temp');
    tempLine.dataset.x1 = x1;
    tempLine.dataset.y1 = y1;
    tempLine.setAttribute('d', `M${x1},${y1} C${x1},${y1} ${x1},${y1} ${x1},${y1}`);
    _connSvg.appendChild(tempLine);

    _draggingConn = {
      fromCardId: card.dataset.cardId,
      fromPortEl: port,
      format: isExecutor ? 'executor' : port.dataset.format,
      topic: isExecutor ? '' : port.dataset.topic,
      tempPath: tempLine,
      type: isExecutor ? 'executor' : 'topic',
      _hoveredCard: null,
    };

    // Elevate SVG so temp line renders above cards
    _connSvg.classList.add('dragging-active');

    if (isExecutor) {
      // Highlight all other cards as valid executor targets
      _viewport.querySelectorAll('.canvas-card').forEach(c => {
        if (c.dataset.cardId !== card.dataset.cardId) c.classList.add('exec-target');
      });
    } else {
      // Highlight compatible in-ports
      _viewport.querySelectorAll('.canvas-port.in').forEach(p => {
        if (p.dataset.format === port.dataset.format && p.closest('.canvas-card') !== card) {
          p.classList.add('port-compatible');
        }
      });
    }
  });
}

function _redrawConnections() {
  if (!_connSvg) return;
  _connSvg.querySelectorAll('.connector-line, .connector-hit').forEach(l => l.remove());
  _viewport.querySelectorAll('.conn-delete-btn').forEach(b => b.remove());

  for (const conn of _connections) {
    const fromCard = _cards.find(c => c.id === conn.fromCardId);
    const toCard = _cards.find(c => c.id === conn.toCardId);
    if (!fromCard || !toCard) continue;

    const fromPort = fromCard.el.querySelector(`.canvas-port.out[data-idx="${conn.fromPortIdx}"]`);
    const toPort = toCard.el.querySelector(`.canvas-port.in[data-idx="${conn.toPortIdx}"]`);
    if (!fromPort || !toPort) continue;

    const vpRect = _viewport.getBoundingClientRect();
    const fromRect = fromPort.getBoundingClientRect();
    const toRect = toPort.getBoundingClientRect();

    const x1 = (fromRect.left + fromRect.width / 2 - vpRect.left) / _zoom;
    const y1 = (fromRect.top + fromRect.height / 2 - vpRect.top) / _zoom;
    const x2 = (toRect.left + toRect.width / 2 - vpRect.left) / _zoom;
    const y2 = (toRect.top + toRect.height / 2 - vpRect.top) / _zoom;
    const cx = Math.max(Math.abs(x2 - x1) * 0.5, 60);

    // Invisible wide hit-area path (easier to hover/click)
    const hitLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hitLine.classList.add('connector-hit');
    hitLine.setAttribute('d', `M${x1},${y1} C${x1+cx},${y1} ${x2-cx},${y2} ${x2},${y2}`);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const fmtCls = _fmtColorClass(conn.format);
    line.classList.add('connector-line', fmtCls);
    line.setAttribute('d', `M${x1},${y1} C${x1+cx},${y1} ${x2-cx},${y2} ${x2},${y2}`);
    const arrowId = fmtCls === 'fmt-audio' ? 'conn-arrow-audio'
                  : fmtCls === 'fmt-json'  ? 'conn-arrow-json'
                  : fmtCls === 'fmt-visual' ? 'conn-arrow-visual'
                  : 'conn-arrow';
    line.setAttribute('marker-end', `url(#${arrowId})`);

    // Delete button at midpoint
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const delBtn = document.createElement('button');
    delBtn.className = 'conn-delete-btn';
    delBtn.textContent = '×';
    delBtn.style.left = mx + 'px';
    delBtn.style.top  = my + 'px';
    delBtn.dataset.connId = conn.id;
    _viewport.appendChild(delBtn);

    const showBtn = () => delBtn.classList.add('visible');
    const hideBtn = () => { if (!delBtn.matches(':hover')) delBtn.classList.remove('visible'); };

    hitLine.addEventListener('mouseenter', showBtn);
    hitLine.addEventListener('mouseleave', hideBtn);
    line.addEventListener('mouseenter', showBtn);
    line.addEventListener('mouseleave', hideBtn);
    delBtn.addEventListener('mouseleave', () => delBtn.classList.remove('visible'));
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (_projectRunning) {
        _logActivity('warn', '请停止智能控制后修改');
        return;
      }
      _connections = _connections.filter(c => c.id !== conn.id);
      _resolveAllTopics();
      _autoStopOnDisconnect(conn.toCardId, conn.toPortIdx, conn.fromTopic);
      _redrawConnections();
      _saveLayout();
    });

    line.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (_projectRunning) {
        _logActivity('warn', '请停止智能控制后修改');
        return;
      }
      _connections = _connections.filter(c => c.id !== conn.id);
      _resolveAllTopics();
      _autoStopOnDisconnect(conn.toCardId, conn.toPortIdx, conn.fromTopic);
      _redrawConnections();
      _saveLayout();
    });

    _connSvg.appendChild(hitLine);
    _connSvg.appendChild(line);
  }

  // ── Draw executor connections (vertical, dashed emerald) ──
  for (const conn of _execConnections) {
    const fromCard = _cards.find(c => c.id === conn.fromCardId);
    const toCard = _cards.find(c => c.id === conn.toCardId);
    if (!fromCard || !toCard) continue;

    const execPort = fromCard.el.querySelector('.canvas-port.executor');
    if (!execPort) continue;

    const vpRect = _viewport.getBoundingClientRect();
    const fromRect = execPort.getBoundingClientRect();
    // Target: top center of the destination card
    const toCardRect = toCard.el.getBoundingClientRect();

    const x1 = (fromRect.left + fromRect.width / 2 - vpRect.left) / _zoom;
    const y1 = (fromRect.top + fromRect.height / 2 - vpRect.top) / _zoom;
    const x2 = (toCardRect.left + toCardRect.width / 2 - vpRect.left) / _zoom;
    const y2 = (toCardRect.top - vpRect.top) / _zoom;
    const cy = Math.max(Math.abs(y2 - y1) * 0.5, 60);

    const pathD = `M${x1},${y1} C${x1},${y1+cy} ${x2},${y2-cy} ${x2},${y2}`;

    const hitLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hitLine.classList.add('connector-hit');
    hitLine.setAttribute('d', pathD);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.classList.add('connector-line', 'executor-conn');
    line.setAttribute('d', pathD);
    line.setAttribute('marker-end', 'url(#exec-arrow)');

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const delBtn = document.createElement('button');
    delBtn.className = 'conn-delete-btn';
    delBtn.textContent = '×';
    delBtn.style.left = mx + 'px';
    delBtn.style.top  = my + 'px';
    delBtn.dataset.connId = conn.id;
    _viewport.appendChild(delBtn);

    const showBtn = () => delBtn.classList.add('visible');
    const hideBtn = () => { if (!delBtn.matches(':hover')) delBtn.classList.remove('visible'); };

    hitLine.addEventListener('mouseenter', showBtn);
    hitLine.addEventListener('mouseleave', hideBtn);
    line.addEventListener('mouseenter', showBtn);
    line.addEventListener('mouseleave', hideBtn);
    delBtn.addEventListener('mouseleave', () => delBtn.classList.remove('visible'));

    const removeExec = () => {
      _execConnections = _execConnections.filter(c => c.id !== conn.id);
      _logActivity('executor', `解绑执行器: ${conn.toToolName || conn.toCardId}`);
      _redrawConnections();
      _saveLayout();
    };
    delBtn.addEventListener('click', (e) => { e.stopPropagation(); removeExec(); });
    line.addEventListener('contextmenu', (e) => { e.preventDefault(); removeExec(); });

    _connSvg.appendChild(hitLine);
    _connSvg.appendChild(line);
  }
}

// ── Topic propagation (deterministic, topological-sort based) ─────────────────

/**
 * Resolve all topic assignments across the canvas graph.
 *
 * Algorithm: BFS from source nodes (cards with no inbound data connections).
 * Static topics (declared by MCP) are preserved; derived topics are computed
 * as parentTopic + '/' + toolName. Updates both DOM port attributes and
 * connection.fromTopic for persistence.
 */
function _resolveAllTopics() {
  // 1. Reset out-ports to static topics (read from _allMcps, not DOM alone)
  for (const card of _cards) {
    // Lookup the authoritative static topics from live MCP data
    const mcp = _allMcps.find(m => m.id === card.mcpId);
    const tools = mcp?.tools || [];
    const toolObj = tools.find(t => (typeof t === 'string' ? t : t.name) === card.toolName);
    const toolTopicOut = typeof toolObj === 'object' ? toolObj.topic_out : null;
    // Priority: tool-level (live) > persisted card-level > single-tool MCP fallback
    const isBundleMcp = (mcp?.tools || []).length > 1;
    const toolType = (typeof toolObj === 'object' ? toolObj.type : '') || '';
    const topicOut = toolTopicOut || (card.topicOut?.length ? card.topicOut : (toolType || isBundleMcp ? [] : mcp?.topic_out || []));

    const outPorts = [...card.el.querySelectorAll('.canvas-port.out')];
    for (let i = 0; i < outPorts.length; i++) {
      const staticTopic = topicOut[i]?.topic || outPorts[i].dataset.staticTopic || '';
      outPorts[i].dataset.topic = staticTopic;
    }
    for (const port of card.el.querySelectorAll('.canvas-port.in')) {
      port.dataset.topic = '';
    }
  }

  // 2. Build adjacency structures
  const outgoing = {};  // cardId → [connections from this card]
  const inDegree = {};  // cardId → number of inbound connections
  for (const card of _cards) {
    outgoing[card.id] = [];
    inDegree[card.id] = 0;
  }
  for (const conn of _connections) {
    if (outgoing[conn.fromCardId]) outgoing[conn.fromCardId].push(conn);
    inDegree[conn.toCardId] = (inDegree[conn.toCardId] || 0) + 1;
  }

  // 3. BFS from sources (inDegree === 0)
  const queue = _cards.filter(c => inDegree[c.id] === 0).slice();
  const visited = new Set();

  while (queue.length) {
    const card = queue.shift();
    if (visited.has(card.id)) continue;
    visited.add(card.id);

    // Derive out-port topics from in-port topic (if not already static)
    const inPorts = [...card.el.querySelectorAll('.canvas-port.in')];
    // Use first connected in-port topic as derivation source
    const inTopic = inPorts.find(p => p.dataset.topic)?.dataset.topic || '';

    for (const outPort of card.el.querySelectorAll('.canvas-port.out')) {
      if (!outPort.dataset.topic && inTopic) {
        outPort.dataset.topic = inTopic + '/' + (card.toolName || 'output');
      }
    }

    // Propagate to downstream cards
    for (const conn of outgoing[card.id]) {
      const fromPort = card.el.querySelector(`.canvas-port.out[data-idx="${conn.fromPortIdx}"]`);
      const topic = fromPort?.dataset.topic || '';

      // Sync connection's persisted fromTopic
      conn.fromTopic = topic;

      // Set destination in-port topic
      const toCard = _cards.find(c => c.id === conn.toCardId);
      if (toCard) {
        const toInPort = toCard.el.querySelector(`.canvas-port.in[data-idx="${conn.toPortIdx}"]`);
        if (toInPort) toInPort.dataset.topic = topic;

        inDegree[conn.toCardId]--;
        if (inDegree[conn.toCardId] <= 0 && !visited.has(conn.toCardId)) {
          queue.push(toCard);
        }
      }
    }
  }

  // (debug logs removed)
}

// ── Project lifecycle ─────────────────────────────────────────────────────────

function _autoStopOnDisconnect(cardId, portIdx, topic) {
  if (!_projectRunning) return;
  // Only stop if no other connection still feeds this port
  const stillConnected = _connections.some(c => c.toCardId === cardId && c.toPortIdx === portIdx);
  if (stillConnected) return;
  const card = _cards.find(c => c.id === cardId);
  if (!card) return;
  _triggerAction(card.mcpId, card.toolName, 'stop', topic ? { input_topic: topic, instance_id: card.id } : { instance_id: card.id });
}

async function _startProject() {
  // 先对 agentcore 发 stop，清理上一轮残留的 topic 订阅（必须 await 否则后续 start 会被 stop 覆盖）
  await _triggerAction('agentcore', 'decision_core', 'stop', {});

  // Check all cards on canvas that need config are configured (via sidebar per-tool config)
  const unconfigured = _cards.filter(c => {
    const mcp = _allMcps.find(m => m.id === c.mcpId);
    const tools = mcp?.tools || [];
    const toolObj = tools.find(t => (typeof t === 'string' ? t : t.name) === c.toolName);
    const configSchema = typeof toolObj === 'object' ? toolObj.configSchema : null;
    return hasSharedRequired(configSchema) && !isToolConfigured(c.mcpId, c.toolName);
  });
  if (unconfigured.length) {
    const names = unconfigured.map(c => c.toolName).join(', ');
    _logActivity('error', `无法启动：以下工具未配置: ${names}（请在侧边栏中配置）`);
    return;
  }

  // Validate topic connections: every connection must have a fromTopic
  for (const conn of _connections) {
    if (!conn.fromTopic) {
      const fromCard = _cards.find(c => c.id === conn.fromCardId);
      const toCard = _cards.find(c => c.id === conn.toCardId);
      _logActivity('error', `连线缺少 topic: ${fromCard?.toolName || '?'} → ${toCard?.toolName || '?'}，请检查连接`);
      return;
    }
  }

  _projectRunning = true;
  _syncProjectBtn();
  // Enable execute buttons
  document.querySelectorAll('.canvas-exec-btn').forEach(btn => btn.classList.remove('locked'));

  // Resolve all topics before starting (ensures correctness after page reload)
  _resolveAllTopics();

  // Start all cards on canvas, resolving input_topic(s) from connections
  for (const card of _cards) {
    // Collect ALL inbound connections to support multiple input topics
    const inConns = _connections.filter(c => c.toCardId === card.id);
    const topics = inConns.map(conn => conn.fromTopic || '').filter(Boolean);

    let args;
    if (topics.length > 1) {
      args = { input_topics: topics };
    } else if (topics.length === 1) {
      args = { input_topic: topics[0] };
    } else {
      args = {};
    }
    args.instance_id = card.id;
    _triggerAction(card.mcpId, card.toolName, 'start', args);
  }
  // 持久化运行状态
  fetch('/api/config/project-running', {
    method: 'PUT', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({running: true}),
  });
  _logActivity('project', '智能控制已开启');
}

function _stopProject() {
  _projectRunning = false;
  _syncProjectBtn();
  // Disable execute buttons
  document.querySelectorAll('.canvas-exec-btn').forEach(btn => btn.classList.add('locked'));
  // Stop all cards on canvas, resolving input_topic from connections
  for (const card of _cards) {
    const inConns = _connections.filter(c => c.toCardId === card.id);
    const topics = inConns.map(conn => conn.fromTopic || '').filter(Boolean);

    let args;
    if (topics.length > 1) {
      args = { input_topics: topics };
    } else if (topics.length === 1) {
      args = { input_topic: topics[0] };
    } else {
      args = {};
    }
    args.instance_id = card.id;
    _triggerAction(card.mcpId, card.toolName, 'stop', args);
  }
  // 持久化运行状态
  fetch('/api/config/project-running', {
    method: 'PUT', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({running: false}),
  });
  _logActivity('project', '智能控制已停止');
}

function _syncProjectBtn() {
  const btn = document.getElementById('canvas-project-toggle');
  if (!btn) return;
  btn.textContent = _projectRunning ? '停止智能控制' : '开启智能控制';
  btn.title = _projectRunning ? '停止智能控制' : '开启智能控制';
  btn.classList.toggle('running', _projectRunning);
}

async function _triggerAction(mcpId, toolName, action, extraArgs = {}) {
  try {
    await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: toolName, arguments: { action, ...extraArgs } }),
    });
  } catch (err) {
    console.error(`[canvas] ${action} call failed:`, err);
  }
}

// Collect all inbound topics for a card from connections (handles multi-connection to single port)
function _collectInTopics(cardId, el) {
  const inConns = _connections.filter(c => c.toCardId === cardId);
  if (inConns.length) {
    const topics = inConns.map(conn => {
      const inPort = el.querySelector(`.canvas-port.in[data-idx="${conn.toPortIdx}"]`);
      return { topic: conn.fromTopic || inPort?.dataset.topic || '', format: inPort?.dataset.format || conn.format || '' };
    }).filter(t => t.topic);
    if (topics.length) return topics;
  }
  // Fallback: read from DOM ports directly
  return [...el.querySelectorAll('.canvas-port.in')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
}

async function _fetchInfoAndShow(mcp, toolObj, opts) {
  const toolName = typeof toolObj === 'string' ? toolObj : toolObj.name;
  try {
    const res = await fetch(`/api/mcp/${encodeURIComponent(mcp.id)}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: toolName, arguments: { action: 'info' } }),
    });
    const json = await res.json();
    if (json.code === 200 && json.data) {
      const info = typeof json.data === 'string' ? JSON.parse(json.data) : json.data;
      // Only override with info result if it has non-empty topic paths;
      // otherwise keep the live DOM-resolved topics passed in opts.
      if (info.topic_in && info.topic_in.some(t => t.topic)) opts.topicIn = info.topic_in;
      if (info.topic_out && info.topic_out.some(t => t.topic)) opts.topicOut = info.topic_out;
      if (info.description && typeof toolObj === 'object') toolObj.description = info.description;
    }
  } catch { /* fallback to static data */ }
  showToolDetail(mcp, toolObj, opts);
}

// ── Execute ───────────────────────────────────────────────────────────────────

async function _executeCard(el, mcpId, toolName) {
  const btn = el.querySelector('.canvas-exec-btn');
  btn.disabled = true;
  btn.textContent = '执行中…';

  const args = {};
  el.querySelectorAll('.canvas-field-input').forEach(input => {
    const key = input.dataset.key;
    const val = input.value.trim();
    if (val !== '') {
      if (input.type === 'number') args[key] = Number(val);
      else if (val === 'true') args[key] = true;
      else if (val === 'false') args[key] = false;
      else args[key] = val;
    }
  });

  // Auto-inject resolved topics from connected ports (based on schema, not DOM fields)
  const inPorts = [...el.querySelectorAll('.canvas-port.in')];
  const outPorts = [...el.querySelectorAll('.canvas-port.out')];
  const _mcp = _allMcps.find(m => m.id === mcpId);
  const _toolObj = _mcp?.tools?.find(t => (typeof t === 'string' ? t : t.name) === toolName);
  const _schemaProps = (typeof _toolObj === 'object' ? _toolObj.inputSchema : null)?.properties || {};
  let inIdx = 0, outIdx = 0;
  for (const key of Object.keys(_schemaProps)) {
    if (args[key]) continue;
    if (/input.*topic|topic.*in/i.test(key) && inPorts[inIdx]) {
      const t = inPorts[inIdx++].dataset.topic;
      if (t) args[key] = t;
    } else if (/output.*topic|topic.*out/i.test(key) && outPorts[outIdx]) {
      const t = outPorts[outIdx++].dataset.topic;
      if (t) args[key] = t;
    }
  }

  _logActivity('mcp_call', `${toolName} @ ${mcpId}`);

  try {
    const res  = await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ tool: toolName, arguments: args }),
    });
    const json = await res.json();

    if (json.code === 200) {
      const resultText = typeof json.data === 'string'
        ? json.data
        : JSON.stringify(json.data, null, 2);
      _showResult(el, resultText, false);
      _logActivity('mcp_result', `${toolName} → ${resultText}`);
    } else {
      const errText = json.message || '执行失败';
      _showResult(el, errText, true);
      _logActivity('mcp_error', `${toolName} 失败: ${errText}`);
    }
  } catch (err) {
    _showResult(el, String(err), true);
    _logActivity('mcp_error', `${toolName} error: ${err}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ 执行';
  }
}

function _showResult(el, text, isError) {
  // Remove any previous inline result (results now go to the log panel only)
  const existing = el.querySelector('.canvas-result');
  if (existing) existing.remove();
}

function _logActivity(type, msg) {
  const logEl = document.getElementById('activity-log');
  if (!logEl) return;

  const now  = new Date();
  const time = now.toTimeString().slice(0, 8);
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <span class="log-time">${_esc(time)}</span>
    <span class="log-type ${_esc(type)}">${_esc(type.replace('_', ' '))}</span>
    <span class="log-msg">${_esc(msg)}</span>
  `;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Card drag (world-space pointer capture) ───────────────────────────────────

function _makeDraggable(el, cardData) {
  const header = el.querySelector('.canvas-card-header');
  if (!header) return;

  let startClientX, startClientY, startWorldX, startWorldY, isDragging = false;

  header.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.canvas-card-close')) return;
    if (e.target.closest('.canvas-card-info-btn')) return;
    if (_projectRunning) return;
    e.preventDefault();
    e.stopPropagation();

    isDragging   = true;
    startClientX = e.clientX;
    startClientY = e.clientY;
    startWorldX  = cardData.x;
    startWorldY  = cardData.y;

    header.setPointerCapture(e.pointerId);
    el.classList.add('dragging');
  });

  header.addEventListener('pointermove', (e) => {
    if (!isDragging) return;

    // Convert client delta to world delta
    const dx = (e.clientX - startClientX) / _zoom;
    const dy = (e.clientY - startClientY) / _zoom;

    cardData.x = startWorldX + dx;
    cardData.y = startWorldY + dy;

    el.style.left = cardData.x + 'px';
    el.style.top  = cardData.y + 'px';
    _redrawConnections();
  });

  header.addEventListener('pointerup', () => {
    if (!isDragging) return;
    isDragging = false;
    el.classList.remove('dragging');
    _debouncedSave();
  });
}

// ── Layout persistence ────────────────────────────────────────────────────────

let _saveTimer = null;
function _debouncedSave() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(_saveLayout, 400);
}

async function _saveLayout() {
  const cards = _cards.map(c => ({
    id:         c.id,
    mcpId:      c.mcpId,
    toolName:   c.toolName,
    driverName: c.driverName,
    x:          c.x,
    y:          c.y,
    topicIn:    c.topicIn  || [],
    topicOut:   c.topicOut || [],
  }));
  try {
    await fetch('/api/canvas/layout', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ cards, connections: _connections, execConnections: _execConnections, transform: { zoom: _zoom, tx: _tx, ty: _ty } }),
    });
  } catch { /* silent */ }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _syncEmptyState() {
  if (!_emptyEl) return;
  _emptyEl.style.display = _cards.length === 0 ? '' : 'none';
}

function _esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
