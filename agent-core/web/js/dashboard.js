/**
 * dashboard.js — Main dashboard view.
 * Manages skill sidebar, renderer selection, and activity log.
 */

import { connectMotus, onMotusEvent, offMotusEvent } from './motus-stream.js';
import { ActivityRenderer } from './renderers/activity.js';
import { TextRenderer }     from './renderers/text.js';
import { VideoRenderer }    from './renderers/video.js';
import { ImageRenderer }    from './renderers/image.js';
import { AudioRenderer }    from './renderers/audio.js';
import { LidarRenderer }    from './renderers/lidar.js';

const RENDERERS = [VideoRenderer, ImageRenderer, AudioRenderer, LidarRenderer, TextRenderer, ActivityRenderer];

const PREVIEW_CAPACITY = 100;  // sliding window frame count per topic

/**
 * SkillBuffer — background WebSocket + ring-buffer for one topic.
 * Keeps the last PREVIEW_CAPACITY frames so any renderer can replay them on mount.
 */
class SkillBuffer {
  constructor(wsUrl, hint) {
    this._wsUrl   = wsUrl;
    this._hint    = hint;
    this._frames  = [];          // ring buffer (ArrayBuffer[])
    this._pos     = 0;           // write head (mod PREVIEW_CAPACITY)
    this._full    = false;
    this._ws      = null;
    this._consumer = null;       // active renderer's onData callback
    this._connect();
  }

  _connect() {
    const ws = new WebSocket(this._wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen  = () => console.log('[preview ws] open', this._wsUrl);
    ws.onclose = (e) => {
      console.log('[preview ws] closed', e.code, this._wsUrl);
      if (this._ws === ws) {
        this._ws = null;
        // Reconnect after 3 s if not destroyed
        if (!this._destroyed) setTimeout(() => this._connect(), 3000);
      }
    };
    ws.onerror = () => {};
    ws.onmessage = (ev) => {
      let buf;
      if (ev.data instanceof ArrayBuffer) {
        // Binary frame — skip if odd byte length (not valid PCM s16)
        if (ev.data.byteLength % 2 !== 0) return;
        buf = ev.data;
      } else {
        // Text frame (JSON ASR result, ping, meta) — skip pure keepalive pings
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === 'ping') return;
        } catch { /* not JSON, pass through */ }
        buf = new TextEncoder().encode(ev.data).buffer;
      }
      // Store in ring buffer
      this._frames[this._pos] = buf;
      this._pos = (this._pos + 1) % PREVIEW_CAPACITY;
      if (!this._full && this._pos === 0) this._full = true;
      // Forward to active consumer
      this._consumer?.(buf, this._hint);
    };
    this._ws = ws;
  }

  /** Replay buffered frames into a renderer, then register it as consumer.
   *  onDataFn(buf, hint, isReplay) — isReplay=true for historical frames. */
  attach(onDataFn) {
    this._consumer = onDataFn;
    const len   = this._full ? PREVIEW_CAPACITY : this._pos;
    const start = this._full ? this._pos : 0;
    for (let i = 0; i < len; i++) {
      const frame = this._frames[(start + i) % PREVIEW_CAPACITY];
      if (frame) onDataFn(frame, this._hint, true);  // isReplay=true
    }
  }

  /** Detach consumer without closing the WebSocket. */
  detach() { this._consumer = null; }

  destroy() {
    this._destroyed = true;
    this._consumer  = null;
    this._ws?.close();
    this._ws = null;
  }
}

// skill.id → { out: SkillBuffer|null, in: SkillBuffer|null }
const _buffers = {};

// skill.id → last mcp_result event (for label restoration on remount)
const _lastSkillEvents = {};

function _ensureBuffer(skillId, wsUrl, hint, side /* 'out'|'in' */) {
  if (!_buffers[skillId]) _buffers[skillId] = { out: null, in: null };
  if (!_buffers[skillId][side] && wsUrl) {
    _buffers[skillId][side] = new SkillBuffer(wsUrl, hint);
  }
  return _buffers[skillId]?.[side] || null;
}

let _skills = [];          // MCP list from backend
let _activeSkillId = null;
let _activeRenderer = null;
let _inRenderer = null;    // renderer for split-view in_topic panel
let _onGotoConfig = null;
let _activityListener = null;
let _globalRunning = false; // global start/stop state

export function initDashboard(onGotoConfig) {
  _onGotoConfig = onGotoConfig;

  document.getElementById('btn-goto-config').onclick = () => _onGotoConfig && _onGotoConfig();
  setupGlobalCtrl();

  // Load skills and connect WS
  loadSkills().then(() => {
    connectMotus(updateWsStatus);
    setupActivityLog();
  });
}

// ── Skills ────────────────────────────────────────────────────────────────────

async function loadSkills() {
  try {
    const res  = await fetch('/api/mcp');
    const json = await res.json();
    _skills = json.data || [];
  } catch {
    _skills = [];
  }

  // Ping each MCP for online status (fire-and-forget)
  _skills.forEach(async (skill) => {
    try {
      const r = await fetch(`/api/mcp/${skill.id}/ping`, { method: 'POST' });
      const j = await r.json();
      console.log('[ping]', skill.id, JSON.stringify(j.data));
      skill.online      = j.data?.online;
      skill.render_hint = j.data?.render_hint || skill.render_hint;
      skill.tools       = j.data?.tools       || skill.tools       || [];
      skill.resources   = j.data?.resources   || skill.resources   || [];
      skill.server_name = j.data?.server_name || skill.server_name || '';
      skill.topic_out   = j.data?.topic_out   || skill.topic_out   || [];
      skill.topic_in    = j.data?.topic_in    || skill.topic_in    || [];
      skill.ws_path     = j.data?.ws_path     || skill.ws_path     || '';
      // url already in skill from GET /api/mcp
      renderSkillList();
      // Spin up background preview buffers for this skill
      _startPreviewBuffers(skill);
      // Auto-start if global state is running
      if (_globalRunning && skill.online) _callToolForSkill(skill, 'start');
    } catch {
      skill.online = false;
      renderSkillList();
    }
  });

  renderSkillList();
}

function renderSkillList() {
  const ul = document.getElementById('skill-list');
  if (!_skills.length) {
    ul.innerHTML = '<li style="padding:10px 10px;color:var(--text-dim);font-size:12px">暂无硬件数据总线</li>';
  } else {
    ul.innerHTML = _skills.map(s => {
      const status = s.online === true ? 'online' : s.online === false ? 'offline' : 'pending';
      const active = s.id === _activeSkillId ? ' active' : '';
      return `
        <li class="skill-item${active}" data-id="${s.id}">
          <span class="skill-dot ${status}"></span>
          <span class="skill-name">${s.name}</span>
          <span class="skill-hint">${s.render_hint || ''}</span>
        </li>
      `;
    }).join('');
    ul.querySelectorAll('.skill-item').forEach(li => {
      li.onclick = () => selectSkill(li.dataset.id);
    });
  }

  // Always refresh device overview when not in renderer mode
  if (!_activeSkillId) renderDeviceOverview();
}

function selectSkill(id) {
  _activeSkillId = id;
  renderSkillList();

  const skill = _skills.find(s => s.id === id);
  if (skill) mountRenderer(skill);
}

// ── Global Start / Stop ───────────────────────────────────────────────────────

async function _callToolForSkill(skill, actionName) {
  const tools = skill.tools || [];
  for (const tool of tools) {
    const args = { action: actionName };
    if (actionName === 'start' && skill.topic_in?.[0]?.topic) {
      const t = skill.topic_in[0].topic;
      args.input_topic = t;
      args.topic       = t;
    }
    console.log(`[ctrl] ${tool} ${skill.id} action=${actionName} args=${JSON.stringify(args)}`);
    try {
      const resp = await fetch(`/api/mcp/${encodeURIComponent(skill.id)}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool, arguments: args }),
      });
      const j = await resp.json();
      console.log(`[ctrl] ${tool} ${skill.id} →`, JSON.stringify(j).slice(0, 200));
    } catch (e) {
      console.error(`[ctrl] ${tool} ${skill.id} failed:`, e);
    }
  }
}

async function globalStart() {
  _globalRunning = true;
  localStorage.setItem('motus_global_running', '1');
  updateGlobalCtrlUI();
  await Promise.all(_skills.filter(s => s.online).map(s => _callToolForSkill(s, 'start')));
  // Re-select the active skill to re-attach the renderer after start.
  // Without this, Stop → Start All leaves the renderer detached (no waveform).
  const targetId = _activeSkillId || _skills.find(s => s.online)?.id;
  if (targetId) selectSkill(targetId);
}

async function globalStop() {
  _globalRunning = false;
  localStorage.removeItem('motus_global_running');
  updateGlobalCtrlUI();
  _activeRenderer?.stopPlayback?.();
  _inRenderer?.stopPlayback?.();
  await Promise.all(_skills.filter(s => s.online).map(s => _callToolForSkill(s, 'stop')));
}

function updateGlobalCtrlUI() {
  const startBtn = document.getElementById('btn-global-start');
  const stopBtn  = document.getElementById('btn-global-stop');
  if (startBtn) startBtn.disabled = _globalRunning;
  if (stopBtn)  stopBtn.disabled  = !_globalRunning;
}

function setupGlobalCtrl() {
  _globalRunning = localStorage.getItem('motus_global_running') === '1';
  const startBtn = document.getElementById('btn-global-start');
  const stopBtn  = document.getElementById('btn-global-stop');
  if (startBtn) startBtn.onclick = globalStart;
  if (stopBtn)  stopBtn.onclick  = globalStop;
  updateGlobalCtrlUI();
}

// ── Background preview buffers ────────────────────────────────────────────────

function _wsUrlFor(path) {
  if (!path) return null;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${path}`;
}

function _startPreviewBuffers(skill) {
  if (!skill.online) return;
  const topicOut = skill.topic_out?.[0];
  const topicIn  = skill.topic_in?.[0];

  // out buffer: skill.ws_path (the primary stream)
  if (skill.ws_path) {
    const hint = skill.render_hint || 'activity';
    _ensureBuffer(skill.id, _wsUrlFor(skill.ws_path), hint, 'out');
  }
  // in buffer: /ws/bus + topic_in path
  if (topicIn?.topic) {
    const inHint = topicIn.format || 'activity';
    _ensureBuffer(skill.id, _wsUrlFor('/ws/bus' + topicIn.topic), inHint, 'in');
  }
}

// ── Device Overview (placeholder) ────────────────────────────────────────────

function renderDeviceOverview() {
  const el = document.getElementById('device-overview');
  if (!el) return;

  if (!_skills.length) {
    el.innerHTML = `
      <div class="overview-empty">
        <div class="placeholder-icon">◈</div>
        <p>暂无硬件数据总线</p>
        <p style="font-size:12px;color:var(--text-dim);margin-top:4px">前往 ⚙ 配置 添加 MCP 硬件数据总线</p>
      </div>`;
    return;
  }

  el.innerHTML = `
    <div class="overview-title">硬件数据总线状态</div>
    <div class="device-grid">
      ${_skills.map(s => {
        const status = s.online === true ? 'online' : s.online === false ? 'offline' : 'pending';
        const statusText = { online: '在线', offline: '离线', pending: '检测中…' }[status];
        const hint = s.render_hint || 'activity';
        const tools = s.tools || [];
        const toolNames = tools.map(t => (typeof t === 'string' ? t : t.name) || '').filter(Boolean);
        const toolLabel = toolNames.length
          ? toolNames.slice(0, 3).join(', ') + (toolNames.length > 3 ? ` +${toolNames.length - 3}` : '')
          : '';
        return `
          <div class="device-card" data-id="${s.id}">
            <div class="device-card-header">
              <span class="skill-dot ${status}" style="flex-shrink:0"></span>
              <span class="device-card-name">${s.name}</span>
            </div>
            ${s.server_name ? `<div class="device-card-server">${s.server_name}</div>` : ''}
            <div class="device-card-meta">
              <span class="device-card-status ${status}">${statusText}</span>
              <span class="device-card-hint">${hint}</span>
            </div>
            ${toolLabel ? `<div class="device-card-tools">${toolLabel}</div>` : ''}
          </div>`;
      }).join('')}
    </div>`;

  el.querySelectorAll('.device-card').forEach(card => {
    card.onclick = () => selectSkill(card.dataset.id);
  });
}

// ── Renderer ──────────────────────────────────────────────────────────────────

// Track which skill's buffers are currently attached to renderers
let _attachedSkillId = null;

function mountRenderer(skill) {
  // Detach previous renderers from their buffers (buffers keep running)
  if (_attachedSkillId && _buffers[_attachedSkillId]) {
    _buffers[_attachedSkillId].out?.detach();
    _buffers[_attachedSkillId].in?.detach();
  }
  _attachedSkillId = null;

  // Unmount previous renderer instances
  if (_activeRenderer) {
    if (_activityListener) offMotusEvent(_activityListener);
    _activeRenderer.unmount();
    _activeRenderer = null;
  }
  if (_inRenderer) {
    _inRenderer.unmount?.();
    _inRenderer = null;
  }

  const hint     = skill.render_hint || 'activity';
  const Renderer = RENDERERS.find(r => r.canRender(hint)) || ActivityRenderer;

  const topicIn  = skill.topic_in?.[0]  || null;
  const topicOut = skill.topic_out?.[0] || null;
  const hasIn    = !!(topicIn?.topic || topicIn?.format);
  const hasOut   = !!(topicOut?.topic || topicOut?.format);
  const isSplit  = hasIn && hasOut;

  const content = document.getElementById('renderer-content');
  content.innerHTML = '';
  content.classList.remove('hidden');
  document.getElementById('renderer-placeholder').classList.add('hidden');

  // ── Device control toolbar ──
  const toolbar = document.createElement('div');
  toolbar.className = 'renderer-toolbar';

  // Build topic badges
  const topicBadges = [
    hasIn  ? `<span class="topic-badge topic-in"  title="输入 topic"><span class="topic-dir">IN</span>${topicIn.topic  || topicIn.format  || ''}</span>` : '',
    hasOut ? `<span class="topic-badge topic-out" title="输出 topic"><span class="topic-dir">OUT</span>${topicOut.topic || topicOut.format || ''}</span>` : '',
  ].filter(Boolean).join('');

  toolbar.innerHTML = `
    <div class="renderer-toolbar-info">
      <span class="renderer-device-name">${skill.server_name || skill.name}</span>
      <span class="renderer-device-hint">${hint}</span>
      ${topicBadges}
    </div>
  `;
  content.appendChild(toolbar);

  console.log('[device] skill=', JSON.stringify({id: skill.id, url: skill.url, ws_path: skill.ws_path, topic_out: skill.topic_out}));

  // ── Renderer area (split or single) ──
  const rendererWrap = document.createElement('div');
  rendererWrap.className = isSplit ? 'renderer-body split' : 'renderer-body';

  let outPanel;
  if (isSplit) {
    // Left panel: topic_in — subscribe and render upstream data
    const inPanel = document.createElement('div');
    inPanel.className = 'renderer-panel';
    inPanel.innerHTML = `
      <div class="renderer-panel-header topic-in-header">
        <span class="topic-dir-label">in_topic</span>
        <span class="topic-path">${topicIn.topic || '—'}</span>
        <span class="topic-fmt">${topicIn.format || ''}</span>
      </div>`;
    const inBody = document.createElement('div');
    inBody.className = 'renderer-panel-body';
    inPanel.appendChild(inBody);
    rendererWrap.appendChild(inPanel);

    // Mount a renderer for in_topic based on its format — no playback (monitoring only)
    const inHint = topicIn.format || 'activity';
    const InRenderer = RENDERERS.find(r => r.canRender(inHint)) || ActivityRenderer;
    _inRenderer = Object.assign(Object.create(Object.getPrototypeOf(InRenderer)), InRenderer);
    _inRenderer.mount(inBody, skill.id + '_in');

    // Attach to background buffer (replay last N frames then receive live data)
    const inBuf = _buffers[skill.id]?.in;
    if (inBuf) {
      inBuf.attach((buf, fmt, isReplay) => isReplay ? _inRenderer?.onDataSilent?.(buf) : _inRenderer?.onData?.(buf, fmt));
    } else if (topicIn.topic) {
      // Buffer not ready yet (device came online after initial ping) — create on demand
      const inWsUrl = _wsUrlFor('/ws/bus' + topicIn.topic);
      const buf = _ensureBuffer(skill.id, inWsUrl, inHint, 'in');
      buf?.attach((b, fmt, isReplay) => isReplay ? _inRenderer?.onDataSilent?.(b) : _inRenderer?.onData?.(b, fmt));
    }

    // Right panel: topic_out (actual renderer)
    outPanel = document.createElement('div');
    outPanel.className = 'renderer-panel';
    outPanel.innerHTML = `
      <div class="renderer-panel-header topic-out-header">
        <span class="topic-dir-label">out_topic</span>
        <span class="topic-path">${topicOut.topic || '—'}</span>
        <span class="topic-fmt">${topicOut.format || ''}</span>
      </div>`;
    const outBody = document.createElement('div');
    outBody.className = 'renderer-panel-body';
    outPanel.appendChild(outBody);
    rendererWrap.appendChild(outPanel);
    _activeRenderer = Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
    _activeRenderer.mount(outBody, skill.id);
  } else {
    // Single panel
    const dir    = hasIn ? 'in_topic' : 'out_topic';
    const topic  = hasIn ? (topicIn?.topic || '—') : (topicOut?.topic || '—');
    const fmt    = hasIn ? (topicIn?.format || '') : (topicOut?.format || '');
    const hdrCls = hasIn ? 'topic-in-header' : 'topic-out-header';
    const singleHeader = document.createElement('div');
    singleHeader.className = `renderer-panel-header ${hdrCls}`;
    singleHeader.innerHTML = `
      <span class="topic-dir-label">${dir}</span>
      <span class="topic-path">${topic}</span>
      <span class="topic-fmt">${fmt}</span>`;
    rendererWrap.appendChild(singleHeader);

    const bodyEl = document.createElement('div');
    bodyEl.className = 'renderer-panel-body';
    rendererWrap.appendChild(bodyEl);
    _activeRenderer = Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
    _activeRenderer.mount(bodyEl, skill.id);
  }

  content.appendChild(rendererWrap);

  // Route motus events to active renderer; cache last event per skill for label restoration
  _activityListener = (event) => {
    _lastSkillEvents[skill.id] = event;
    _activeRenderer?.onEvent(event);
  };
  onMotusEvent(skill.id, _activityListener);
  // Restore last cached event (e.g. ASR text label) immediately on remount
  if (_lastSkillEvents[skill.id]) _activeRenderer?.onEvent(_lastSkillEvents[skill.id]);

  // Attach to background preview buffer (replay + live stream)
  // For devices with only topic_in (e.g. speaker), fall back to inBuf.
  const outBuf = _buffers[skill.id]?.out;
  const inBuf  = _buffers[skill.id]?.in;
  const activeBuf = outBuf || (!skill.ws_path && hasIn ? inBuf : null);
  console.log(`[attach] skill=${skill.id} outBuf=${!!outBuf} inBuf=${!!inBuf} activeBuf=${!!activeBuf} ws_path=${skill.ws_path}`);
  if (activeBuf) {
    let replayCount = 0, liveCount = 0;
    activeBuf.attach((buf, fmt, isReplay) => {
      if (isReplay) { replayCount++; _activeRenderer?.onDataSilent?.(buf); }
      else {
        if (liveCount === 0) console.log(`[audio:live] first live frame after ${replayCount} replay frames, skill=${skill.id}`);
        liveCount++;
        _activeRenderer?.onData?.(buf, fmt);
      }
    });
    console.log(`[attach] attached to activeBuf, replay frames will be silent`);
  } else if (skill.ws_path) {
    // Buffer not ready yet — create on demand
    const buf = _ensureBuffer(skill.id, _wsUrlFor(skill.ws_path), hint, 'out');
    console.log(`[attach] created on-demand outBuf for ws_path=${skill.ws_path}`);
    buf?.attach((b, fmt, isReplay) => isReplay ? _activeRenderer?.onDataSilent?.(b) : _activeRenderer?.onData?.(b, fmt));
  } else if (hasIn && topicIn?.topic) {
    // No out buffer and no ws_path — wire directly to in_topic stream
    const inWsUrl = _wsUrlFor('/ws/bus' + topicIn.topic);
    const buf = _ensureBuffer(skill.id, inWsUrl, hint, 'in');
    console.log(`[attach] wired to in_topic=${topicIn.topic}`);
    buf?.attach((b, fmt, isReplay) => isReplay ? _activeRenderer?.onDataSilent?.(b) : _activeRenderer?.onData?.(b, fmt));
  } else {
    console.log('[preview] no ws_path for skill', skill.id);
  }
  _attachedSkillId = skill.id;
}

// ── Activity log ──────────────────────────────────────────────────────────────

function setupActivityLog() {
  onMotusEvent(null, appendActivityEntry);
}

function appendActivityEntry(event) {
  if (event.type === 'ping') return;  // skip keepalive

  const log = document.getElementById('activity-log');
  const atBottom = log.scrollHeight - log.scrollTop <= log.clientHeight + 40;

  const row = document.createElement('div');
  row.className = 'log-entry';

  const t = new Date(event.ts * 1000).toLocaleTimeString();
  const typeLabel = event.mcp_id ? `${event.type}` : event.type;
  const mcpTag = event.mcp_id ? `<span style="color:var(--accent);margin-right:4px">[${event.mcp_id}]</span>` : '';
  const msg = summarize(event);

  row.innerHTML = `
    <span class="log-time">${t}</span>
    <span class="log-type ${event.type}">${typeLabel}</span>
    <span class="log-msg">${mcpTag}${msg}</span>
  `;
  log.appendChild(row);

  // Keep a max of 500 entries
  while (log.children.length > 500) log.removeChild(log.firstChild);

  if (atBottom) log.scrollTop = log.scrollHeight;
}

function summarize(event) {
  const p = event.payload || {};
  switch (event.type) {
    case 'mcp_call':      return `${p.tool || ''}(${truncate(JSON.stringify(p.args || {}), 80)})`;
    case 'mcp_result':    return `← ${truncate(JSON.stringify(p.result), 100)}`;
    case 'agent_thought': return p.text || '';
    case 'render':        return `renderer=${p.renderer}`;
    case 'status':        return 'connected' in p
                            ? (p.connected ? '● 已连接' : '○ 断开')
                            : (p.online ? '⬤ online' : '○ offline') + (p.mcp_id ? ` [${p.mcp_id}]` : '');
    default:              return truncate(JSON.stringify(p), 80);
  }
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + '…' : str;
}

// ── WebSocket status ──────────────────────────────────────────────────────────

function updateWsStatus(state) {
  const dot   = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  dot.className = 'dot-status ' + state;
  label.textContent = { connected: '已连接', connecting: '重连中…', error: '连接失败' }[state] || state;
}