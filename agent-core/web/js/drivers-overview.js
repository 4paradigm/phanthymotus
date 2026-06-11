/**
 * drivers-overview.js — Left sidebar showing deployed driver cards.
 *
 * Reads from /api/mcp (same source as flow-view).
 * Each card shows: driver name, online status, "所有能力" button, HW node tree.
 */

let _container = null;
let _countEl   = null;
let _emptyEl   = null;
let _onRefresh = null;  // callback(mcpId)

export function initDriversOverview({ onRefresh } = {}) {
  _container = document.getElementById('drivers-overview-list');
  _countEl   = document.getElementById('drivers-overview-count');
  _emptyEl   = document.getElementById('drivers-overview-empty');
  _onRefresh = onRefresh || null;

  // Bind capability modal close button
  const closeBtn = document.getElementById('capability-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      document.getElementById('capability-overlay')?.classList.add('hidden');
    });
  }
}

/**
 * Refresh the sidebar from a pre-fetched MCP list (to avoid double fetch).
 * Call this with the same data returned by flow-view's refresh().
 */
export function renderDriversOverview(mcps, topicStatuses = {}) {
  if (!_container) return;

  const drivers = (mcps || []).filter(m => m.category === 'driver');

  _countEl.textContent = drivers.length;

  if (!drivers.length) {
    _emptyEl?.classList.remove('hidden');
    _container.querySelectorAll('.dov-card').forEach(el => el.remove());
    return;
  }
  _emptyEl?.classList.add('hidden');

  // Build a map of existing cards for diffing
  const existing = {};
  _container.querySelectorAll('.dov-card').forEach(el => {
    existing[el.dataset.mcpId] = el;
  });

  const seen = new Set();
  for (const mcp of drivers) {
    seen.add(mcp.id);
    const el = existing[mcp.id];
    if (el) {
      _updateCard(el, mcp, topicStatuses);
    } else {
      const card = _buildCard(mcp, topicStatuses);
      _container.appendChild(card);
    }
  }

  // Remove cards for MCPs that are no longer present
  for (const [id, el] of Object.entries(existing)) {
    if (!seen.has(id)) el.remove();
  }
}

// ── Classification ───────────────────────────────────────────────────────────

function _classifyDriver(mcp) {
  const hasOut = (mcp.topic_out || []).filter(t => t.topic).length > 0;
  const hasIn  = (mcp.topic_in  || []).filter(t => t.topic).length > 0;
  if (hasOut && hasIn)  return 'processor';
  if (hasOut)           return 'sensor';
  if (hasIn)            return 'actuator';
  return 'none';
}

const _CLASS_LABELS = {
  sensor:    '传感器',
  actuator:  '执行器',
  processor: '处理器',
  none:      'MCP',
};

// ── Card build ──────────────────────────────────────────────────────────────

function _buildCard(mcp, topicStatuses) {
  const card = document.createElement('div');
  card.className = 'dov-card';
  card.dataset.mcpId = mcp.id;
  _updateCard(card, mcp, topicStatuses);
  return card;
}

function _updateCard(card, mcp, topicStatuses) {
  const online = mcp.online;
  const statusCls = online === true ? 'online' : online === false ? 'offline' : 'pending';
  const name = mcp.server_name || mcp.name || mcp.id;

  const tools     = (mcp.tools    || []).map(t => typeof t === 'string' ? t : t.name);
  const topicsOut = (mcp.topic_out || []).filter(t => t.topic);
  const topicsIn  = (mcp.topic_in  || []).filter(t => t.topic);
  const classification = _classifyDriver(mcp);
  const classLabel     = _CLASS_LABELS[classification] || 'MCP';

  // Hardware node tree HTML
  const hasTopics = topicsOut.length || topicsIn.length;
  const hwTreeHtml = hasTopics ? `
    <div class="dov-hw-tree">
      <div class="dov-hw-node">
        <span class="dov-hw-badge ${_esc(classification)}">${_esc(classLabel)}</span>
        <div class="dov-hw-topics">
          ${topicsOut.map(t => `
            <div class="dov-hw-topic topic-out" title="${_esc(t.topic)}">
              <span class="dov-topic-dot ${_esc(topicStatuses[t.topic] ?? 'unknown')}"></span>
              <span class="dov-hw-topic-arrow">↑</span>
              <span class="dov-hw-topic-path">${_esc(_shortTopic(t.topic))}</span>
            </div>`).join('')}
          ${topicsIn.map(t => `
            <div class="dov-hw-topic topic-in" title="${_esc(t.topic)}">
              <span class="dov-topic-dot ${_esc(topicStatuses[t.topic] ?? 'unknown')}"></span>
              <span class="dov-hw-topic-arrow">↓</span>
              <span class="dov-hw-topic-path">${_esc(_shortTopic(t.topic))}</span>
            </div>`).join('')}
        </div>
      </div>
    </div>` : '';

  card.innerHTML = `
    <div class="dov-card-header">
      <span class="dov-status-dot ${statusCls}"></span>
      <span class="dov-card-name">${_esc(name)}</span>
      <button class="dov-refresh-btn" title="重新 ping" data-mcp-id="${_esc(mcp.id)}">↺</button>
    </div>
    ${tools.length ? `
    <div class="dov-section dov-tools-row">
      <div class="dov-section-label">工具 · ${tools.length}</div>
      <button class="dov-capabilities-btn" data-mcp-id="${_esc(mcp.id)}">所有能力</button>
    </div>` : ''}
    ${hwTreeHtml}
  `;

  // Bind refresh button
  const refreshBtn = card.querySelector('.dov-refresh-btn');
  if (refreshBtn && _onRefresh) {
    refreshBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      refreshBtn.classList.add('spinning');
      _onRefresh(mcp.id).finally?.(() => refreshBtn.classList.remove('spinning'));
    });
  }

  // Bind "所有能力" button
  const capBtn = card.querySelector('.dov-capabilities-btn');
  if (capBtn) {
    capBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _openCapabilityModal(mcp);
    });
  }
}

// ── Capability Modal ─────────────────────────────────────────────────────────

async function _openCapabilityModal(mcp) {
  const overlay = document.getElementById('capability-overlay');
  const title   = document.getElementById('capability-modal-title');
  const body    = document.getElementById('capability-modal-body');
  if (!overlay || !title || !body) return;

  const driverName = mcp.server_name || mcp.name || mcp.id;
  const classification = _classifyDriver(mcp);
  const classLabel     = _CLASS_LABELS[classification] || '';

  title.innerHTML = `${_esc(driverName)}&ensp;<span class="dov-hw-badge ${_esc(classification)}" style="vertical-align:middle">${_esc(classLabel)}</span>`;
  body.innerHTML = '<div class="cap-loading">加载能力清单…</div>';
  overlay.classList.remove('hidden');

  // Fetch full tool schemas
  let tools = mcp.tools || [];
  try {
    const res  = await fetch(`/api/mcp/${mcp.id}/tools`);
    const json = await res.json();
    if (json.code === 200 && Array.isArray(json.data)) tools = json.data;
  } catch { /* use cached tools */ }

  const topicsOut = (mcp.topic_out || []).filter(t => t.topic);
  const topicsIn  = (mcp.topic_in  || []).filter(t => t.topic);

  if (!tools.length) {
    body.innerHTML = '<div class="cap-loading">暂无工具信息</div>';
    return;
  }

  body.innerHTML = tools.map(tool => _buildToolCard(tool, topicsOut, topicsIn)).join('');
}

function _buildToolCard(tool, topicsOut, topicsIn) {
  const name   = typeof tool === 'string' ? tool : (tool.name || '');
  const desc   = typeof tool === 'object' ? (tool.description || '') : '';
  const schema = typeof tool === 'object' ? (tool.inputSchema || null) : null;

  const schemaHtml = schema && Object.keys(schema).length > 0
    ? `<div class="cap-schema">
        <div class="cap-schema-label">Input Schema</div>
        <pre class="cap-schema-body">${_esc(JSON.stringify(schema, null, 2))}</pre>
       </div>`
    : '';

  const topicsHtml = [
    ...topicsOut.map(t => `<span class="cap-topic cap-topic-out" title="${_esc(t.topic)}">↑ ${_esc(_shortTopic(t.topic))}</span>`),
    ...topicsIn.map(t =>  `<span class="cap-topic cap-topic-in"  title="${_esc(t.topic)}">↓ ${_esc(_shortTopic(t.topic))}</span>`),
  ].join('');

  return `
    <div class="cap-tool-card">
      <div class="cap-tool-header">
        <span class="cap-tool-name">${_esc(name)}</span>
        ${topicsHtml ? `<div class="cap-topics">${topicsHtml}</div>` : ''}
      </div>
      ${desc ? `<div class="cap-tool-desc">${_esc(desc)}</div>` : ''}
      ${schemaHtml}
    </div>`;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function _esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _shortTopic(path) {
  const parts = path.split('/').filter(Boolean);
  if (parts.length <= 2) return path;
  return '/' + parts.slice(-2).join('/');
}
