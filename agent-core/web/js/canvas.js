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

import { showToast } from './toast.js';

import { showTopicDetail } from './detail-panel.js';
import { showToolDetail, isToolConfigured, isInstanceConfigured, openInstanceConfigModal, hasSharedRequired } from './sidebar.js';
import { toggleMicStream, isMicActive } from './mic-stream.js';
import { sessionId } from './session.js';
import { getToken } from './auth.js';

let _canvasEl   = null;
let _viewport   = null;
let _emptyEl    = null;
let _zoomLabel  = null;
let _connSvg    = null;
let _cards      = [];   // [{ id, mcpId, toolName, driverName, x, y, el }]
let _allMcps    = [];

// ── Editor Lock ──────────────────────────────────────────────────────────────
// The session id comes from session.js (per-tab, see the rationale there) and is
// also sent on the /ws/motus connection — the backend releases this session's
// lock shortly after that socket drops, so closing/killing the tab frees the
// canvas without depending on an unload handler firing.
const _sessionId = sessionId();
let _isEditor = false;
let _currentEditor = null;  // session_id of current editor (null = no one)

/**
 * Ensure current session holds the edit lock, auto-claiming it if free.
 * Returns true if editing may proceed, false if the canvas is occupied by
 * another session (in which case a toast + locked UI state is shown).
 */
async function _ensureEdit() {
  if (_isEditor) return true;
  try {
    const resp = await fetch('/api/canvas/claim-edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: _sessionId }),
    });
    const data = await resp.json();
    if (resp.ok) {
      _isEditor = true;
      _currentEditor = _sessionId;
      _updateEditorUI();
      _showToast('已自动获取编辑权限');
      return true;
    }
    _currentEditor = data.editor || null;
    _updateEditorUI();
    _showToast('画布正被其他用户编辑，请稍后重试');
    return false;
  } catch {
    _showToast('无法获取编辑权限，请检查网络');
    return false;
  }
}

// Thin wrapper over the shared implementation so every existing call site
// keeps working unchanged; the canvas still hosts its toast inside _canvasEl.
function _showToast(msg) {
  showToast(msg, _canvasEl);
}

// Connection state
let _connections = [];  // [{id, fromCardId, fromPort, toCardId, toPort, format}]
let _execConnections = []; // [{id, fromCardId, toCardId, toToolName, toMcpId}]
let _draggingConn = null; // {fromCardId, fromPortEl, format, topic, tempPath, type?}

// Project run state
let _projectRunning = false;

export function isProjectRunning() { return _projectRunning; }
export function redrawCanvas() { _redrawConnections(); }
export function ensureEdit() { return _ensureEdit(); }
export function isEditor() { return _isEditor; }

/**
 * Discard the in-memory canvas and re-read it from the server.
 * Used after a solution is loaded — the backend rewrote canvas_layout directly,
 * so what's on screen is stale.
 */
export function reloadFromServer() { return _reloadLayout(); }


/**
 * Programmatically add a card to the canvas (used by mobile tap-to-add).
 * Returns true if added, false if rejected.
 */
export async function addCardFromSidebar({ mcpId, toolName, driverName, hasConfig, multiInstance }) {
  if (_projectRunning) return false;
  if (!(await _ensureEdit())) return false;
  if (hasConfig && !isToolConfigured(mcpId, toolName)) return false;
  if (!multiInstance) {
    const existing = _cards.find(c => c.mcpId === mcpId && c.toolName === toolName);
    if (existing) return false;
  }
  // Position at viewport center in world coordinates
  const rect = _canvasEl.getBoundingClientRect();
  const cx = (rect.width / 2 - _tx) / _zoom;
  const cy = (rect.height / 2 - _ty) / _zoom;
  let x = cx - 130, y = cy - 70;
  ({ x, y } = _findNonOverlappingPos(x, y));
  const id = 'card-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  _addCard({ id, mcpId, toolName, driverName, x, y }, true);
  return true;
}

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

    // Cards whose topic_out is derived resolve inside _resolveAllTopics now
    // (_revalidateDerivedTopics). The bespoke recovery that used to live here
    // only fetched for cards with an *empty* out-port that also had an outgoing
    // connection, which missed both a stale non-empty topic and a leaf card like
    // TTS.

    // Restore viewport transform if saved
    if (layoutJson.data?.transform) {
      _zoom = layoutJson.data.transform.zoom ?? 1;
      _tx   = layoutJson.data.transform.tx   ?? 0;
      _ty   = layoutJson.data.transform.ty   ?? 0;
      _applyTransform();
    }

    // On mobile, auto-fit cards to viewport instead of using saved desktop transform
    if (window.innerWidth <= 768 && _cards.length > 0) {
      _fitToViewport();
    }

    // Initialize editor lock state from layout response
    _currentEditor = layoutJson.editor || null;
    _isEditor = _currentEditor === _sessionId;
  } catch { /* start empty */ }

  // Show editor status bar
  _updateEditorUI();

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

  // Cross-tab sync: listen for project_state / editor-lock / layout events via WebSocket
  const { onMotusEvent } = await import('./motus-stream.js');
  onMotusEvent(null, (event) => {
    if (event.type === 'project_state') {
      const running = event.payload?.running;
      if (running !== _projectRunning) {
        _projectRunning = running;
        _syncProjectBtn();
        document.querySelectorAll('.canvas-exec-btn').forEach(btn => {
          btn.classList.toggle('locked', !_projectRunning);
        });
      }
    } else if (event.type === 'canvas_editor') {
      _applyEditorState(event.payload?.editor || null, event.payload?.reason || '');
    } else if (event.type === 'canvas_layout') {
      // Ignore the echo of our own autosave; readers reload to follow the editor.
      if ((event.payload?.editor || '') !== _sessionId) _scheduleReload();
    }
  });

  // Re-sync state when tab becomes visible (fallback for WS disconnect)
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      _checkEditStatus();
      fetch('/api/config/project-running').then(r => r.json()).then(d => {
        if (d.running !== _projectRunning) {
          _projectRunning = d.running;
          _syncProjectBtn();
          document.querySelectorAll('.canvas-exec-btn').forEach(btn => {
            btn.classList.toggle('locked', !_projectRunning);
          });
        }
      }).catch(() => {});
    }
  });

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

    // multiInstance tools have per-card instance topics (set by connections + start()).
    // Tool-schema-level data from pings must NOT overwrite instance-specific topics.
    const liveTopicIn  = typeof toolObj === 'object' ? toolObj.topic_in  : null;
    const liveTopicOut = typeof toolObj === 'object' ? toolObj.topic_out : null;
    if (typeof toolObj === 'object' && toolObj.multiInstance) {
      // (skip topic update — fall through to configSchema check below)
    } else {
      if (liveTopicIn  && liveTopicIn.length  && JSON.stringify(liveTopicIn)  !== JSON.stringify(card.topicIn))  {
        // Don't overwrite dynamic instance topics with static empty-topic values from MCP tool definition
        if (liveTopicIn.some(t => t.topic) || !card.topicIn?.some(t => t.topic)) { card.topicIn  = liveTopicIn;  topicsChanged = true; }
      }
      if (liveTopicOut && liveTopicOut.length && JSON.stringify(liveTopicOut) !== JSON.stringify(card.topicOut)) {
        if (liveTopicOut.some(t => t.topic) || !card.topicOut?.some(t => t.topic)) { card.topicOut = liveTopicOut; topicsChanged = true; }
      }
    }
    // Re-fetch driver-inferred topics for static (non-multiInstance) cards that still have no real topic path
    // For multiInstance tools, topics are input-dependent and must be resolved after connection
    // Only fetch once (not on every poll) — mark card to avoid repeated calls
    if (!card.topicOut?.some(t => t.topic) && liveTopicOut?.length && !toolObj?.multiInstance && !card._topicFetched) {
      card._topicFetched = true;
      _fetchTopicsFromDriver(card, '');
    }

    // Also trigger rebuild if instance-config button presence doesn't match live configSchema
    if (!topicsChanged) {
      const liveConfigSchema = typeof toolObj === 'object' ? toolObj.configSchema : null;
      const liveHasInstanceFields = liveConfigSchema &&
        Object.values(liveConfigSchema.properties || {}).some(d => d.scope === 'instance');
      const hasBtn = !!card.el.querySelector('.canvas-card-instance-cfg-btn');
      if (!!liveHasInstanceFields !== hasBtn) topicsChanged = true;
    }
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

function _fitToViewport() {
  if (!_cards.length) return;
  const rect = _canvasEl.getBoundingClientRect();
  const padding = 30;
  // Find bounding box of all cards in world coords (with port margins)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const cardW = window.innerWidth <= 768 ? 220 : 260;
  for (const c of _cards) {
    minX = Math.min(minX, c.x - 20); // port extends left
    minY = Math.min(minY, c.y);
    maxX = Math.max(maxX, c.x + cardW + 20); // port extends right
    maxY = Math.max(maxY, c.y + 160);
  }
  const contentW = maxX - minX;
  const contentH = maxY - minY;
  const availW = rect.width - padding * 2;
  const availH = rect.height - padding * 2;
  _zoom = Math.min(availW / contentW, availH / contentH, 1);
  _zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, _zoom));
  // Center content
  _tx = padding + (availW - contentW * _zoom) / 2 - minX * _zoom;
  _ty = padding + (availH - contentH * _zoom) / 2 - minY * _zoom;
  _applyTransform();
}

// ── Touch helpers ─────────────────────────────────────────────────────────────
function _getTouchDist(touches) {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.hypot(dx, dy);
}
function _getTouchCenter(touches) {
  return {
    x: (touches[0].clientX + touches[1].clientX) / 2,
    y: (touches[0].clientY + touches[1].clientY) / 2
  };
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

  // ── Pinch-to-zoom (mobile two-finger gesture) ──
  let _pinchDist = 0;
  let _pinchCenter = { x: 0, y: 0 };
  let _pinching = false;

  _canvasEl.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      _pinching = true;
      _panning = false; // cancel any single-finger pan
      _pinchDist = _getTouchDist(e.touches);
      _pinchCenter = _getTouchCenter(e.touches);
    }
  }, { passive: false });

  _canvasEl.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && _pinching) {
      e.preventDefault();
      const dist = _getTouchDist(e.touches);
      const center = _getTouchCenter(e.touches);
      const scale = dist / _pinchDist;
      const delta = (scale - 1) * 0.5;
      _zoomAt(center.x, center.y, delta);
      _pinchDist = dist;
      _pinchCenter = center;
    }
  }, { passive: false });

  _canvasEl.addEventListener('touchend', (e) => {
    if (e.touches.length < 2) {
      _pinching = false;
      _debouncedSave();
    }
  });

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

  // Auto-start toggle
  _initAutoStartToggle();
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

    if (!_isEditor) {
      _showDropReject(e, _currentEditor ? '画布已被其他用户锁定' : '请先获取编辑权');
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

  // Call info(instance_id) to get driver-inferred topics for static tools.
  // For multiInstance processors, topics depend on the connected input_topic — skip here.
  // For multiInstance sensors (ext_mic, ext_camera), the topic is deterministic from instance_id — fetch eagerly.
  const _mcp2 = _allMcps.find(m => m.id === mcpId);
  const _toolObj2 = (_mcp2?.tools || []).find(t => (typeof t === 'string' ? t : t.name) === toolName);
  const _isMultiInstanceSensor = _toolObj2?.multiInstance && _toolObj2?.type === 'sensor';
  if ((!_toolObj2?.multiInstance || _isMultiInstanceSensor) && (_toolObj2?.topic_out?.length || _toolObj2?.topic_in?.length)) {
    _fetchTopicsFromDriver(cardData, '');
  }

  _syncEmptyState();

  if (save) _saveLayout();
}

async function _removeCard(id) {
  if (_projectRunning) {
    _logActivity('warn', '请停止智能控制后修改');
    return;
  }
  if (!(await _ensureEdit())) return;
  const idx = _cards.findIndex(c => c.id === id);
  if (idx === -1) return;
  const removed = _cards[idx];
  // Stop the instance this card owns, before it stops being reachable. Nothing
  // else can: stop-project walks the saved layout, and this card is about to
  // leave it, so its ROS node, subscription and any CUDA context would live
  // until perception exits — publishing to a topic no card accounts for.
  // `stop` is idempotent, so doing this to a card that was never started is fine.
  _triggerAction(removed.mcpId, removed.toolName, 'stop', { instance_id: removed.id });
  removed.el.remove();
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

  // Priority: card saved topics (driver-inferred, real paths) > static tool definition > MCP fallback
  // Static tool.topic_out may have empty topic paths for multiInstance/dynamic tools,
  // so prefer savedTopicOut when it has real paths.
  const toolTopicIn  = typeof toolObj === 'object' ? toolObj.topic_in  : null;
  const toolTopicOut = typeof toolObj === 'object' ? toolObj.topic_out : null;
  const isBundleMcp = (mcp?.tools || []).length > 1;
  const savedOutHasReal = savedTopicOut?.some(t => t.topic);
  const staticOutHasReal = toolTopicOut?.some(t => t.topic);
  const savedInHasReal = savedTopicIn?.some(t => t.topic);
  const staticInHasReal = toolTopicIn?.some(t => t.topic);
  const topicIn  = (savedInHasReal  ? savedTopicIn  : null) || (staticInHasReal  ? toolTopicIn  : null) || (savedTopicIn?.length  ? savedTopicIn  : null) || (toolTopicIn?.length  ? toolTopicIn  : (toolType || isBundleMcp ? [] : mcp?.topic_in  || []));
  const topicOut = (savedOutHasReal ? savedTopicOut : null) || (staticOutHasReal ? toolTopicOut : null) || (savedTopicOut?.length ? savedTopicOut : null) || (toolTopicOut?.length ? toolTopicOut : (toolType || isBundleMcp ? [] : mcp?.topic_out || []));
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
        const liveCard = _cards.find(c => c.id === id);
        const liveTopicOut = (liveCard?.topicOut?.length ? liveCard.topicOut : null)
          || [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut, instanceId: id });
      }
    });
  } else if (effectiveType === 'sensor') {
    // Check if sensor has callable actions beyond start/stop/info/config
    const sensorProps = schema?.properties || {};
    const sensorRequired = schema?.required || [];
    const _SENSOR_SYS_ACTIONS = new Set(['start', 'stop', 'info', 'config']);
    const sensorActionDef = sensorProps.action;
    // Support both enum (plain list) and oneOf [{const, title}] formats
    const _actionVals = def => def?.enum || (def?.oneOf?.map(o => o.const).filter(v => v != null)) || [];
    const hasSensorActions = _actionVals(sensorActionDef).some(a => !_SENSOR_SYS_ACTIONS.has(a));

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
        const rawVals = _actionVals(def);
        if (rawVals.length || def.enum || def.oneOf) {
          // Build label map from oneOf titles
          const titleMap = {};
          (def.oneOf || []).forEach(o => { if (o.const != null) titleMap[o.const] = o.title || o.const; });
          const allVals = rawVals.length ? rawVals : (def.enum || []);
          const enumVals = key === 'action' ? allVals.filter(v => !_SENSOR_SYS_ACTIONS.has(v)) : allVals;
          if (!enumVals.length) return '';
          const opts = enumVals.map(v => `<option value="${_esc(v)}">${_esc(titleMap[v] || v)}</option>`).join('');
          inputHtml = `<select class="canvas-field-input" data-key="${_esc(key)}">${opts}</select>`;
        } else if (def.format === 'file') {
          const accept = def.accept || '*/*';
          inputHtml = `<div class="canvas-field-file"><input type="hidden" class="canvas-field-input" data-key="${_esc(key)}"><button type="button" class="canvas-file-btn" data-accept="${_esc(accept)}">Choose File</button><span class="canvas-file-name"></span></div>`;
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
        const liveCard = _cards.find(c => c.id === id);
        const liveTopicOut = (liveCard?.topicOut?.length ? liveCard.topicOut : null)
          || [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut, instanceId: id });
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
      _openTopicDetailFor(el, mcpId, topicOut);
    });

    const sensorExecBtn = el.querySelector('.canvas-exec-btn');
    if (sensorExecBtn) {
      sensorExecBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!(await _ensureEdit())) return;
        await _executeCard(el, mcpId, toolName, id);
      });
    }

    // Generic file upload buttons for sensor cards (format: 'file' in schema)
    el.querySelectorAll('.canvas-file-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        const wrapper = btn.closest('.canvas-field-file');
        const hiddenInput = wrapper.querySelector('.canvas-field-input');
        const nameSpan = wrapper.querySelector('.canvas-file-name');
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = btn.dataset.accept || '*/*';
        fileInput.onchange = async () => {
          if (!fileInput.files[0]) return;
          btn.textContent = 'Uploading...';
          const form = new FormData();
          form.append('file', fileInput.files[0]);
          form.append('path', '/tmp/uploads');
          try {
            const res = await fetch('/api/file/upload', { method: 'POST', body: form });
            const data = await res.json();
            if (data.code === 200) {
              hiddenInput.value = '/tmp/uploads/' + fileInput.files[0].name;
              nameSpan.textContent = fileInput.files[0].name;
              btn.textContent = 'Re-select';
            } else {
              btn.textContent = 'Failed';
              setTimeout(() => { btn.textContent = 'Choose File'; }, 2000);
            }
          } catch (err) {
            btn.textContent = 'Error';
            setTimeout(() => { btn.textContent = 'Choose File'; }, 2000);
          }
        };
        fileInput.click();
      });
    });
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
      } else if (def.format === 'file') {
        const accept = def.accept || '*/*';
        inputHtml = `<div class="canvas-field-file"><input type="hidden" class="canvas-field-input" data-key="${_esc(key)}"><button class="canvas-file-btn" data-accept="${_esc(accept)}">选择文件</button><span class="canvas-file-name"></span></div>`;
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
        actionSelect.addEventListener('change', async () => {
          if (!(await _ensureEdit())) {
            _applyActionParams();  // revert visual to match current state
            return;
          }
          _applyActionParams();
        });
        _applyActionParams();  // 初始应用
      }
    }

    el.querySelector('.canvas-card-info-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      const liveMcp = _allMcps.find(m => m.id === mcpId);
      if (liveMcp) {
        const liveTopicIn  = _collectInTopics(id, el);
        const liveCard = _cards.find(c => c.id === id);
        const liveTopicOut = (liveCard?.topicOut?.length ? liveCard.topicOut : null)
          || [...el.querySelectorAll('.canvas-port.out')].map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
        _fetchInfoAndShow(liveMcp, toolObj || toolName, { topicIn: liveTopicIn, topicOut: liveTopicOut, instanceId: id });
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
        if (!(await _ensureEdit())) return;
        await _executeCard(el, mcpId, toolName);
      });
    }

    const viewBtn = el.querySelector('.canvas-view-btn');
    if (viewBtn) {
      viewBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        _openTopicDetailFor(el, mcpId, topicOut);
      });
    }

    // remote_mic 特殊渲染：麦克风录音按钮
    // remote_mic: no manual button — mic auto-starts with project

    // Generic file upload buttons (format: 'file' in schema)
    el.querySelectorAll('.canvas-file-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        const wrapper = btn.closest('.canvas-field-file');
        const hiddenInput = wrapper.querySelector('.canvas-field-input');
        const nameSpan = wrapper.querySelector('.canvas-file-name');
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = btn.dataset.accept || '*/*';
        fileInput.onchange = async () => {
          if (!fileInput.files[0]) return;
          btn.textContent = 'Uploading...';
          const form = new FormData();
          form.append('file', fileInput.files[0]);
          form.append('path', '/tmp/uploads');
          try {
            const res = await fetch('/api/file/upload', { method: 'POST', body: form });
            const data = await res.json();
            if (data.code === 200) {
              hiddenInput.value = '/tmp/uploads/' + fileInput.files[0].name;
              nameSpan.textContent = fileInput.files[0].name;
              btn.textContent = 'Re-select';
            } else {
              btn.textContent = 'Failed';
              setTimeout(() => { btn.textContent = 'Choose File'; }, 2000);
            }
          } catch (err) {
            btn.textContent = 'Error';
            setTimeout(() => { btn.textContent = 'Choose File'; }, 2000);
          }
        };
        fileInput.click();
      });
    });
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
  document.addEventListener('pointermove', (e) => {
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

  document.addEventListener('pointerup', (e) => {
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
        // The destination's output topic is derived from this new input;
        // _resolveAllTopics above already scheduled that refetch, and doing it
        // here as well raced it — this path passes the dragged topic, which is
        // empty when the source's own topic has not resolved yet, and whichever
        // reply landed last won.
      }
    }

    // Cleanup
    if (_draggingConn.tempPath) _draggingConn.tempPath.remove();
    _draggingConn = null;
  });

  // Delegate pointerdown on out ports and executor ports
  _viewport.addEventListener('pointerdown', async (e) => {
    const outPort = e.target.closest('.canvas-port.out');
    const execPort = !outPort ? e.target.closest('.canvas-port.executor') : null;
    if (!outPort && !execPort) return;
    if (_projectRunning) {
      _logActivity('warn', '请停止智能控制后修改');
      return;
    }
    if (!(await _ensureEdit())) return;
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
  // Force synchronous layout flush so compositor layer is invalidated immediately
  void _connSvg.getBoundingClientRect();

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
    delBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (_projectRunning) {
        _logActivity('warn', '请停止智能控制后修改');
        return;
      }
      if (!(await _ensureEdit())) return;
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

    const removeExec = async () => {
      if (!(await _ensureEdit())) return;
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
    // Priority: card.topicOut (driver-inferred, has real paths) > static tool definition > MCP fallback
    // card.topicOut is populated by _fetchTopicsFromDriver; static tool.topic_out may have empty topic paths
    // for multiInstance tools, so only fall back to it when card has no real topics.
    const isBundleMcp = (mcp?.tools || []).length > 1;
    const toolType = (typeof toolObj === 'object' ? toolObj.type : '') || '';
    const cardHasRealTopic = card.topicOut?.some(t => t.topic);
    const staticHasRealTopic = toolTopicOut?.some(t => t.topic);
    const topicOut = (cardHasRealTopic ? card.topicOut : null)
      || (staticHasRealTopic ? toolTopicOut : null)
      || (card.topicOut?.length ? card.topicOut : (toolType || isBundleMcp ? [] : mcp?.topic_out || []));

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

  // A derived topic_out is only valid for the input it came from, so the walk
  // ends by checking that and refetching whatever no longer matches.
  _revalidateDerivedTopics();
}

// ── Project lifecycle ─────────────────────────────────────────────────────────

function _autoStopOnDisconnect(cardId, portIdx, topic) {
  // No `if (!_projectRunning) return` here. Every call site already refuses to
  // edit while the project runs, so that check made this function a no-op at all
  // of them — a card left subscribed to a topic the canvas no longer connects,
  // which is how an old TTS node kept speaking on a deleted link. Stopping a
  // stopped instance just returns {"state": "idle"}.
  // Only stop if no other connection still feeds this port
  const stillConnected = _connections.some(c => c.toCardId === cardId && c.toPortIdx === portIdx);
  if (stillConnected) return;
  const card = _cards.find(c => c.id === cardId);
  if (!card) return;
  _triggerAction(card.mcpId, card.toolName, 'stop', topic ? { input_topic: topic, instance_id: card.id } : { instance_id: card.id });
}

async function _startProject() {
  // Save canvas layout first (so backend reads latest topology)
  await _saveLayout();

  // Import motus for event subscription
  const { onMotusEvent, offMotusEvent } = await import('./motus-stream.js');

  // Subscribe to startup progress events
  let modal = null;
  let itemIndex = {};  // tool_name -> index in modal
  // Cards that accepted start but are still loading a model (perception TTS/OCR
  // fetch and warm one up). The backend settles them with a later ready/error
  // event, so the subscription has to outlive project_start_done — otherwise
  // the modal would sit at "启动中..." forever, or worse, close claiming success.
  const loading = new Set();
  let sequenceDone = false;

  function _finishIfSettled() {
    if (!sequenceDone || loading.size > 0) return;
    if (modal) modal.startCountdown(15);
    offMotusEvent(_onEvent);
  }

  function _onEvent(event) {
    const p = event.payload || {};
    if (event.type === 'project_start_begin') {
      const cards = p.cards || [];
      const items = cards.map(c => ({ card: { toolName: c.tool, mcpId: c.mcp_id } }));
      modal = _showStartupModal(items);
      cards.forEach((c, i) => { itemIndex[`${c.mcp_id}:${c.tool}`] = i; });
    } else if (event.type === 'project_start_item') {
      const key = `${p.mcp_id}:${p.tool}`;
      const idx = itemIndex[key];
      // Anything other than 'loading' is terminal for the wait: ready, error,
      // cancelled, or a status added later.
      if (p.status === 'loading') loading.add(key);
      else loading.delete(key);
      if (modal && idx !== undefined) {
        modal.updateItem(idx, p.status, p.message || '');
      }
      _finishIfSettled();
    } else if (event.type === 'project_start_done') {
      sequenceDone = true;
      if (p.has_error) {
        offMotusEvent(_onEvent);
        return;
      }
      if (modal && loading.size > 0) {
        modal.setWaiting(loading.size);
      }
      _finishIfSettled();
    }
  }

  onMotusEvent(null, _onEvent);

  // 立即启动浏览器麦克风（与 API 调用并行，解决 self-check 时序问题）
  const remoteMicCard = _cards.find(c => c.toolName === 'remote_mic');
  if (remoteMicCard && !isMicActive()) {
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProto}://${location.host}/ws/mic`;
    toggleMicStream(wsUrl, (active) => {
      const micBtn = remoteMicCard.el?.querySelector('.canvas-mic-btn');
      if (micBtn) {
        micBtn.textContent = active ? '\u23F9 停止录音' : '\uD83C\uDF99 开始录音';
        micBtn.classList.toggle('recording', active);
      }
    }).catch(err => _logActivity('warn', `麦克风启动失败: ${err.message}`));
  }

  // Call unified backend start-project
  try {
    const res = await fetch('/api/config/start-project', { method: 'POST' });
    if (res.ok) {
      _projectRunning = true;
      _syncProjectBtn();
      document.querySelectorAll('.canvas-exec-btn').forEach(btn => btn.classList.remove('locked'));
      _logActivity('project', '智能控制已开启');
    } else {
      const data = await res.json().catch(() => ({}));
      _logActivity('error', `启动失败: ${data.detail || res.status}`);
      offMotusEvent(_onEvent);
      if (modal) {
        _showStartupError(modal);
      }
    }
  } catch (e) {
    _logActivity('error', `启动失败: ${e.message}`);
    offMotusEvent(_onEvent);
    if (modal) modal.close();
  }
}

function _stopProject() {
  _projectRunning = false;
  _syncProjectBtn();
  document.querySelectorAll('.canvas-exec-btn').forEach(btn => btn.classList.add('locked'));
  // Auto-stop mic stream
  for (const card of _cards) {
    if (card.toolName === 'remote_mic' && isMicActive()) {
      toggleMicStream('', () => {}).catch(() => {});
      const micBtn = card.el?.querySelector('.canvas-mic-btn');
      if (micBtn) {
        micBtn.textContent = '\uD83C\uDF99 开始录音';
        micBtn.classList.remove('recording');
      }
    }
  }
  fetch('/api/config/stop-project', { method: 'POST' }).catch(() => {});
  _logActivity('project', '智能控制已停止');
}

function _syncProjectBtn() {
  const btn = document.getElementById('canvas-project-toggle');
  if (!btn) return;
  btn.textContent = _projectRunning ? '停止智能控制' : '开启智能控制';
  btn.title = _projectRunning ? '停止智能控制' : '开启智能控制';
  btn.classList.toggle('running', _projectRunning);
}

function _initAutoStartToggle() {
  const checkbox = document.getElementById('auto-start-checkbox');
  if (!checkbox) return;

  fetch('/api/config/auto-start')
    .then(r => r.json())
    .then(res => { checkbox.checked = res.auto_start ?? false; })
    .catch(() => {});

  checkbox.addEventListener('change', async () => {
    // 开启时警告 token 消耗
    if (checkbox.checked) {
      const confirmed = confirm(
        '开启后，设备启动时将自动开始智能控制，持续消耗 LLM Token。\n\n确认开启开机自启动？'
      );
      if (!confirmed) {
        checkbox.checked = false;
        return;
      }
    }
    try {
      await fetch('/api/config/auto-start', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_start: checkbox.checked }),
      });
    } catch (e) {
      console.error('[auto-start] save failed:', e);
      checkbox.checked = !checkbox.checked;
    }
  });
}

async function _triggerAction(mcpId, toolName, action, extraArgs = {}) {
  try {
    const res = await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: toolName, arguments: { action, ...extraArgs } }),
    });
    return await res.json();
  } catch (err) {
    console.error(`[canvas] ${action} call failed:`, err);
    return null;
  }
}

// ── Startup Modal ──────────────────────────────────────────────────────────────

function _showStartupError(modalWrapper) {
  const modalEl = modalWrapper.modal;
  const cancelBtn = modalEl.querySelector('.startup-cancel-btn');
  if (cancelBtn) {
    cancelBtn.textContent = '关闭';
    cancelBtn.onclick = () => modalWrapper.close();
  }
  // Update modal title to indicate failure
  const title = modalEl.querySelector('.modal-title');
  if (title) title.textContent = '启动失败';
}

function _showStartupModal(items) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.style.width = '420px';
  modal.innerHTML = `
    <div class="modal-header">
      <span class="modal-title">启动智能控制</span>
    </div>
    <ul class="startup-modal-list"></ul>
    <div class="startup-modal-footer">
      <button class="startup-cancel-btn">取消启动</button>
    </div>`;
  overlay.appendChild(modal);
  const list = modal.querySelector('.startup-modal-list');
  const dots = [];
  const statuses = [];
  items.forEach(({ card }) => {
    const li = document.createElement('li');
    li.className = 'startup-modal-item';
    li.innerHTML = `<span class="startup-dot"></span><span class="startup-name">${card.toolName}</span><span class="startup-status">等待启动</span>`;
    list.appendChild(li);
    dots.push(li.querySelector('.startup-dot'));
    statuses.push(li.querySelector('.startup-status'));
  });
  document.body.appendChild(overlay);

  const STATUS_TEXT = {
    starting: '启动中...', loading: '模型加载中...', ready: '已就绪',
    error: '启动失败', cancelled: '已取消',
  };
  function updateItem(i, state, msg) {
    dots[i].className = 'startup-dot ' + state;
    statuses[i].textContent = msg || STATUS_TEXT[state] || '';
  }
  function close() {
    if (_countdownTimer) clearInterval(_countdownTimer);
    overlay.remove();
  }

  let _countdownTimer = null;
  function setWaiting(count) {
    // Every card was accepted, but some are still fetching or warming a model.
    // Say so instead of auto-closing on a success the operator does not have yet.
    const title = modal.querySelector('.modal-title');
    if (title) title.textContent = `等待模型加载（${count}）`;
  }

  function startCountdown(seconds) {
    const footer = modal.querySelector('.startup-modal-footer');
    const title = modal.querySelector('.modal-title');
    if (title) title.textContent = '启动完成';
    let remaining = seconds;
    footer.innerHTML = `<button class="startup-close-btn">关闭 <span class="startup-countdown">${remaining}s</span></button>`;
    const btn = footer.querySelector('.startup-close-btn');
    const span = footer.querySelector('.startup-countdown');
    btn.addEventListener('click', close);
    _countdownTimer = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        close();
      } else {
        span.textContent = `${remaining}s`;
      }
    }, 1000);
  }

  const cancelBtn = modal.querySelector('.startup-cancel-btn');
  cancelBtn.addEventListener('click', () => {
    close();
    // Actually stop the project when user cancels during startup
    _stopProject();
  });
  return { modal, updateItem, close, startCountdown, setWaiting };
}

/**
 * Parse the result of a /api/mcp/{id}/call response.
 * The API wraps driver responses as MCP content arrays: {code:200, data:[{type:"text",text:"..."}]}
 * Returns the parsed JSON object, or null on failure.
 */
function _parseMcpCallResult(json) {
  if (!json || json.code !== 200) return null;
  const data = json.data;
  if (Array.isArray(data)) {
    const text = data[0]?.text;
    if (text) { try { return JSON.parse(text); } catch { return null; } }
    return null;
  }
  if (typeof data === 'string') { try { return JSON.parse(data); } catch { return null; } }
  return typeof data === 'object' && data !== null ? data : null;
}

/**
 * Ask the driver to infer topics for a card given an optional input topic.
 * Used for multiInstance sensors (_addCard) and processors (after wiring).
 * Updates card.topicOut and DOM out-ports if driver returns non-empty topics.
 */
async function _fetchTopicsFromDriver(card, inputTopic) {
  const want = inputTopic || '';
  if (card._topicFetchFor === want) return;   // identical request already in flight
  card._topicFetchFor = want;
  try {
    const resp = await fetch(`/api/mcp/${encodeURIComponent(card.mcpId)}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: card.toolName, arguments: { action: 'info', instance_id: card.id, input_topic: inputTopic } }),
    });
    const data = await resp.json();
    const parsed = _parseMcpCallResult(data);
    const topicOut = parsed?.topic_out;
    // Remember which input produced this answer. A derived topic is only valid
    // for the input it was derived from, and without this the cache could never
    // be told apart from a still-correct one — see _revalidateDerivedTopics.
    card.topicOutFrom = want;
    if (topicOut?.some(t => t.topic)) {
      card.topicOut = topicOut;
      const outPorts = [...card.el.querySelectorAll('.canvas-port.out')];
      topicOut.forEach((t, i) => { if (outPorts[i] && t.topic) outPorts[i].dataset.topic = t.topic; });
      // A resolved topic is some downstream card's input, so the graph below this
      // one has to be re-walked — otherwise a chain (mic → asr → tts) only ever
      // resolves its first hop.
      _resolveAllTopics();
      _redrawConnections();
      _debouncedSave();
    } else if (card.topicOut?.some(t => t.topic)) {
      // The driver cannot infer an output for this input, so whatever we are
      // holding was derived from a different one and is now wrong. Dropping it is
      // what stops a deleted connection's topic from outliving the connection.
      card.topicOut = [];
      _resolveAllTopics();
      _redrawConnections();
      _debouncedSave();
    }
  } catch (e) {
    card._topicFetchFor = null;   // let a later resolve retry after a failure
    console.warn('[canvas] info fetch failed:', e);
  }
}

/**
 * The input topic a card is currently fed, per the resolved graph.
 *
 * Read from the in-port dataset, which _resolveAllTopics' BFS has just written.
 * Deliberately *not* falling back to connection.fromTopic: that field holds
 * whatever was known when the link was drawn, so falling back to it would feed a
 * card the leftover topic of a link that has since been deleted — the very thing
 * this revalidation exists to undo. An empty string means "not known yet", which
 * is the honest answer.
 */
/**
 * Open the data-stream detail panel for a card's output topic.
 *
 * Two things this must not do, both of which it used to:
 *
 * 1. Trust the `topicOut` captured when the card was rendered. That closure
 *    goes stale as soon as a connection changes; the out-port dataset is the
 *    live value, kept current by _resolveAllTopics. Same idiom as the topic
 *    reads elsewhere in this file.
 *
 * 2. Accept an entry just because the array is non-empty. The old test was
 *    `topics.length`, and for a multiInstance tool the MCP-level `topic_out` is
 *    format-only by design — mcp_manage deliberately does not back-fill
 *    per-instance topics, since those live on the cards. So `tts` reported
 *    `[{format: 'audio/pcm-16k'}]`, which passed a length check with
 *    `topic === undefined`; showTopicDetail then built `/ws/bus` + undefined,
 *    matching no route (`/ws/bus/{topic:path}`). That is why "查看数据流" on a
 *    TTS card opened a panel that never showed a waveform, and why the server
 *    log filled with `ASGI callable returned without completing handshake`.
 */
function _openTopicDetailFor(el, mcpId, cachedTopicOut) {
  const livePorts = [...el.querySelectorAll('.canvas-port.out')]
    .map(p => ({ topic: p.dataset.topic, format: p.dataset.format }));
  const liveMcp = _allMcps.find(m => m.id === mcpId);
  const candidate = [...livePorts, ...(cachedTopicOut || []), ...(liveMcp?.topic_out || [])]
    .find(t => t && t.topic);
  if (!candidate) {
    _showToast('该卡片还没有解析出输出 topic —— 先点「开始控制」把它启动起来');
    return;
  }
  showTopicDetail(candidate.topic, candidate.format || '');
}

function _inputTopicFor(card) {
  const inConn = _connections.find(c => c.toCardId === card.id);
  if (!inConn) return '';
  const inPort = card.el.querySelector(`.canvas-port.in[data-idx="${inConn.toPortIdx}"]`);  return inPort?.dataset.topic || '';
}

/**
 * Refetch any card whose cached topic_out no longer matches its input.
 *
 * card.topicOut for a multiInstance tool is *derived*: the driver infers it from
 * the connected input topic (`/remote_control/mic` + asr → `/remote_control/mic/asr`).
 * It was cached with no record of which input produced it and given top priority
 * in _resolveAllTopics, so it survived the input changing under it. Connect TTS to
 * remote_message, delete that connection, connect it to ASR instead, and the card
 * kept publishing to `/remote_control/message/tts`: the dashboard panel watched a
 * topic nothing fed, and the audio panel stayed silent.
 *
 * Nothing recomputed it either — the disconnect paths never refetched, and the
 * page-load recovery only looks at cards with an *empty* out-port that also have
 * an outgoing connection, which a stale leaf card like TTS satisfies neither of.
 */
function _revalidateDerivedTopics() {
  for (const card of _cards) {
    const want = _inputTopicFor(card);
    const known = card.topicOutFrom;
    const hasReal = card.topicOut?.some(t => t.topic);
    // Nothing verified this card's topics in this page's lifetime. The saved
    // layout is not evidence — a stale derived topic is exactly what gets
    // persisted — so re-derive it whenever the card has an input to derive from.
    // Once per card per page load: _fetchTopicsFromDriver drops a repeat request
    // for the same input.
    // `want` is '' both when nothing feeds this card and when its source has not
    // resolved yet, and those want opposite treatment: the first should be asked
    // now (the driver answers with its default output), the second must wait or
    // it would adopt that default over the topic it is about to derive. Treating
    // them alike is what this function was written to fix but did not: TTS lost
    // its inbound connection, so want was '' with hasReal true, and the guard
    // below skipped it — the card kept '/remote_control/message/tts' while the
    // driver published on '/perception/tts', and the panel stayed empty.
    const inputless = !_connections.some(c => c.toCardId === card.id);
    if (known === undefined) {
      if (want || inputless || !hasReal) _fetchTopicsFromDriver(card, want);
      continue;
    }
    if (known !== want) _fetchTopicsFromDriver(card, want);
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
      body: JSON.stringify({ tool: toolName, arguments: { action: 'info', instance_id: opts.instanceId || '' } }),
    });
    const json = await res.json();
    const info = _parseMcpCallResult(json);
    if (info) {
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

async function _executeCard(el, mcpId, toolName, instanceId) {
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

  // Inject instance_id from card identity so multiInstance tools can resolve device_path
  if (instanceId && !args.instance_id) args.instance_id = instanceId;

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
  const existing = el.querySelector('.canvas-result');
  if (existing) existing.remove();
  const wrapper = document.createElement('div');
  wrapper.className = 'canvas-result';
  const pre = document.createElement('pre');
  pre.className = 'canvas-result-pre' + (isError ? ' error' : '');
  pre.textContent = text;
  wrapper.appendChild(pre);
  el.appendChild(wrapper);
}

function _flashStartError(msg) {
  const btn = document.getElementById('canvas-project-toggle');
  if (btn) {
    btn.classList.add('error-flash');
    setTimeout(() => btn.classList.remove('error-flash'), 2000);
  }
  // Show a toast near the button
  const ctrl = document.getElementById('canvas-top-control');
  if (!ctrl) return;
  let toast = ctrl.querySelector('.start-error-toast');
  if (toast) toast.remove();
  toast = document.createElement('div');
  toast.className = 'start-error-toast';
  toast.textContent = msg;
  ctrl.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
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

  header.addEventListener('pointerdown', async (e) => {
    if (e.target.closest('.canvas-card-close')) return;
    if (e.target.closest('.canvas-card-info-btn')) return;
    if (e.target.closest('.canvas-card-instance-cfg-btn')) return;
    if (_projectRunning) return;
    if (!(await _ensureEdit())) return;
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
  if (!_isEditor) return;  // only the current editor may persist; system-triggered saves must not auto-claim
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
    const resp = await fetch('/api/canvas/layout', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ cards, connections: _connections, execConnections: _execConnections, transform: { zoom: _zoom, tx: _tx, ty: _ty }, session_id: _sessionId }),
    });
    if (resp.status === 403) {
      // Lost edit permission — reload layout from server
      _isEditor = false;
      _updateEditorUI();
      await _reloadLayout();
    }
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

// ── Editor Lock UI ───────────────────────────────────────────────────────────

const _SVG_PEN = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>';
const _SVG_LOCK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>';

function _createEditorBar() {
  const bar = document.createElement('div');
  bar.id = 'canvas-editor-bar';
  bar.className = 'canvas-editor-bar';
  _canvasEl.appendChild(bar);
  return bar;
}

function _updateEditorUI() {
  let bar = document.getElementById('canvas-editor-bar');
  if (!bar) bar = _createEditorBar();
  bar.classList.toggle('canvas-editor-bar--locked', !_isEditor && !!_currentEditor);

  if (_isEditor) {
    bar.innerHTML = `${_SVG_PEN}<span class="editor-label editor-label--active">编辑中</span><button class="editor-btn" id="canvas-release-btn">释放</button>`;
    bar.querySelector('#canvas-release-btn').onclick = _releaseEdit;
    _setCanvasReadonly(false);
  } else if (_currentEditor) {
    // Read-only, not broken: reading the canvas and its data streams stays open to
    // everyone, only writes need the lock.
    bar.innerHTML = `${_SVG_LOCK}<span class="editor-label editor-label--locked">画布由其他人编辑中（只读）</span>`;
    _setCanvasReadonly(true);
  } else {
    bar.innerHTML = `${_SVG_PEN}<button class="editor-btn editor-btn--claim" id="canvas-claim-btn">编辑</button>`;
    bar.querySelector('#canvas-claim-btn').onclick = _ensureEdit;
    _setCanvasReadonly(true);
  }
}

function _setCanvasReadonly(readonly) {
  // Don't use pointer-events: none — it blocks all interaction including toast triggers.
  // Instead, each action handler calls _ensureEdit() individually.
  document.querySelectorAll('.sidebar-tool-item').forEach(el => {
    el.draggable = !readonly;
  });
}

async function _releaseEdit() {
  try {
    await fetch('/api/canvas/release-edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: _sessionId }),
    });
  } catch { /* silent */ }
  _isEditor = false;
  _currentEditor = null;
  _updateEditorUI();
}

/**
 * Adopt a lock state pushed by the server (canvas_editor event) or read from a
 * poll. Readers reload once the canvas is freed so they end up on the editor's
 * final layout.
 */
function _applyEditorState(editor, reason) {
  const wasEditor = _isEditor;
  _currentEditor = editor;
  _isEditor = editor === _sessionId;
  _updateEditorUI();

  if (!editor) {
    if (!wasEditor) {
      // Canvas just went free — pick up whatever the last editor left behind.
      _scheduleReload(reason === 'release');
      if (reason === 'release') _showToast('画布编辑权已释放，可点击「编辑」接管');
    }
  } else if (wasEditor && !_isEditor) {
    _showToast('编辑权已被释放，画布转为只读');
  }
}

// Reload debounce: an editor dragging cards autosaves repeatedly, and readers
// should not refetch on every one of those.
let _reloadTimer = null;
let _lastReloadToast = 0;

function _scheduleReload(silent = false) {
  clearTimeout(_reloadTimer);
  _reloadTimer = setTimeout(async () => {
    await _reloadLayout();
    // Throttled, and skipped when the caller already explained what happened —
    // otherwise the reload toast would immediately replace that message.
    if (!silent && Date.now() - _lastReloadToast > 5000) {
      _lastReloadToast = Date.now();
      _showToast('画布已更新');
    }
  }, 800);
}

async function _checkEditStatus() {
  try {
    const resp = await fetch(`/api/canvas/edit-status?session_id=${encodeURIComponent(_sessionId)}`);
    const data = await resp.json();
    const editor = data.editor || null;
    if (editor === _currentEditor) return;   // no change — don't re-render or reload
    _applyEditorState(editor, '');
  } catch { /* silent */ }
}

async function _reloadLayout() {
  try {
    const layoutRes = await fetch('/api/canvas/layout');
    const layoutJson = await layoutRes.json();
    // Clear current cards
    for (const c of _cards) c.el.remove();
    _cards = [];
    _connections = [];
    _execConnections = [];
    // Reload
    const saved = layoutJson.data?.cards || [];
    for (const c of saved) _addCard(c, false);
    const cardIds = new Set(_cards.map(c => c.id));
    _connections = (layoutJson.data?.connections || []).filter(c => cardIds.has(c.fromCardId) && cardIds.has(c.toCardId));
    _execConnections = (layoutJson.data?.execConnections || []).filter(c => cardIds.has(c.fromCardId) && cardIds.has(c.toCardId));
    _resolveAllTopics();
    _redrawConnections();
    _syncEmptyState();
    // Update editor info
    _currentEditor = layoutJson.editor || null;
    _isEditor = _currentEditor === _sessionId;
    _updateEditorUI();
  } catch { /* silent */ }
}

// Release on page close — a fast path only: correctness comes from the backend
// dropping the lock when this tab's /ws/motus connection closes, since these
// events never fire on a killed process or a reclaimed mobile tab.
//
// The beacon has to carry an application/json Blob (a plain string is sent as
// text/plain, which FastAPI refuses to JSON-decode) and its own ?token= (it
// bypasses the fetch patch in auth.js that injects the Authorization header).
function _releaseBeacon() {
  if (!_isEditor) return;
  const token = getToken();
  const url = '/api/canvas/release-edit' + (token ? `?token=${encodeURIComponent(token)}` : '');
  const blob = new Blob([JSON.stringify({ session_id: _sessionId })],
                        { type: 'application/json' });
  navigator.sendBeacon(url, blob);
}
window.addEventListener('pagehide', _releaseBeacon);
window.addEventListener('beforeunload', _releaseBeacon);
// Restored from the back/forward cache: the beacon already released our lock, so
// re-read the real state instead of trusting the stale in-memory flag.
window.addEventListener('pageshow', (e) => { if (e.persisted) _checkEditStatus(); });

// Periodically check edit status (piggyback on existing polling interval)
setInterval(_checkEditStatus, 10000);
