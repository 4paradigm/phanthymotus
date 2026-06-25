/**
 * sidebar.js — Left sidebar: driver/perception sections with tool cards.
 * Cards show tool name, description, type badge, and detail button.
 * Tools with configSchema show config status and config button.
 */

import { isProjectRunning } from './canvas.js';

let _scroll = null;
let _empty  = null;
let _backdrop = null;
let _modal    = null;

// Cached per-tool config states: { "mcp_id:tool_name": { ... } }
let _toolConfigs = {};

export function initSidebar() {
  _scroll  = document.getElementById('sidebar-scroll');
  _empty   = document.getElementById('sidebar-empty');
  _backdrop = document.getElementById('tool-detail-backdrop');
  _modal    = document.getElementById('tool-detail-modal');

  document.getElementById('tool-detail-close').addEventListener('click', _hideDetail);
  _backdrop.addEventListener('click', (e) => { if (e.target === _backdrop) _hideDetail(); });

  // Search filter
  const searchInput = document.getElementById('sidebar-search');
  if (searchInput) {
    searchInput.addEventListener('input', () => _onSearchInput(searchInput.value));
  }

  // Load saved tool configs
  _loadToolConfigs();
}

async function _loadToolConfigs() {
  try {
    const resp = await fetch('/api/canvas/tool-configs');
    const data = await resp.json();
    _toolConfigs = data.data || {};
  } catch (e) { /* ignore */ }
}

/** Check if a tool (by mcp_id:tool_name key) has shared config saved. */
export function isToolConfigured(mcpId, toolName) {
  return !!_toolConfigs[`${mcpId}:${toolName}`];
}

/** Check if a specific instance has instance config saved. */
export function isInstanceConfigured(mcpId, toolName, instanceId) {
  return !!_toolConfigs[`${mcpId}:${toolName}:${instanceId}`];
}

/** Get all tool configs (shared + instance). */
export function getToolConfigs() { return _toolConfigs; }

/**
 * Refresh sidebar from pre-fetched MCP list.
 */
export function renderSidebar(mcps, topicStatuses = {}) {
  if (!_scroll) return;

  const allMcps = mcps || [];
  const controllers = allMcps.filter(m => m.category === 'controller');
  const drivers = allMcps.filter(m => m.category === 'driver');
  const perceptions = allMcps.filter(m => m.category === 'perception');

  _scroll.innerHTML = '';

  if (!controllers.length && !drivers.length && !perceptions.length) {
    _scroll.appendChild(_empty);
    return;
  }

  // Controller sections (top)
  if (controllers.length) {
    const section = _buildControllerSection(controllers);
    _scroll.appendChild(section);
  }

  // Perception section
  if (perceptions.length) {
    const section = _buildPerceptionSection(perceptions);
    _scroll.appendChild(section);
  }

  // Driver sections (one per driver)
  for (const mcp of drivers) {
    const section = _buildSection(mcp);
    _scroll.appendChild(section);
  }
}

// ── Section builders ──────────────────────────────────────────────────────────

const _TYPE_ORDER = ['sensor', 'actuator', 'processor', 'controller', ''];

function _buildSection(mcp) {
  const name = mcp.server_name || mcp.name || mcp.id;
  const online = mcp.online;
  const statusCls = online === true ? 'online' : online === false ? 'offline' : 'pending';

  const section = document.createElement('div');
  section.className = 'sidebar-section';

  const header = document.createElement('div');
  header.className = 'sidebar-section-header';
  header.innerHTML = `
    <span class="sidebar-section-dot ${_esc(statusCls)}"></span>
    <span class="sidebar-section-name">${_esc(name)}</span>
    <span class="sidebar-section-count">${(mcp.tools || []).length}</span>
  `;
  section.appendChild(header);

  const tools = (mcp.tools || []).map(t => typeof t === 'string' ? { name: t } : t);
  const useChip = tools.length > 6;

  const grid = document.createElement('div');
  grid.className = `sidebar-tool-list${useChip ? ' chip-mode' : ''}`;

  _renderGroupedTools(grid, mcp, tools, useChip);

  section.appendChild(grid);
  return section;
}

function _buildPerceptionSection(perceptions) {
  const section = document.createElement('div');
  section.className = 'sidebar-section sidebar-section-perception';

  const header = document.createElement('div');
  header.className = 'sidebar-section-header';
  const allTools = [];
  for (const mcp of perceptions) {
    const tools = (mcp.tools || []).map(t => typeof t === 'string' ? { name: t } : t);
    for (const tool of tools) {
      if (!tool.type && (tool.topic_in || []).length && (tool.topic_out || []).length) {
        tool.type = 'processor';
      }
      tool._mcp = mcp; // attach mcp ref for rendering
      allTools.push(tool);
    }
  }
  header.innerHTML = `
    <span class="sidebar-section-icon">◈</span>
    <span class="sidebar-section-name">感知</span>
    <span class="sidebar-section-count">${allTools.length}</span>
  `;
  section.appendChild(header);

  const useChip = allTools.length > 6;
  const grid = document.createElement('div');
  grid.className = `sidebar-tool-list${useChip ? ' chip-mode' : ''}`;

  // Group and render
  const groups = {};
  for (const tool of allTools) {
    const t = tool.type || '';
    (groups[t] = groups[t] || []).push(tool);
  }
  for (const type of _TYPE_ORDER) {
    if (!groups[type]?.length) continue;
    grid.appendChild(_buildSubgroupLabel(type, groups[type].length));
    for (const tool of groups[type]) {
      const mcp = tool._mcp;
      grid.appendChild(useChip ? _buildChip(mcp, tool) : _buildToolCard(mcp, tool));
    }
  }

  section.appendChild(grid);
  return section;
}

function _buildControllerSection(controllers) {
  const section = document.createElement('div');
  section.className = 'sidebar-section sidebar-section-controller';

  const header = document.createElement('div');
  header.className = 'sidebar-section-header';
  const totalTools = controllers.reduce((s, m) => s + (m.tools || []).length, 0);
  header.innerHTML = `
    <span class="sidebar-section-icon controller">◆</span>
    <span class="sidebar-section-name">AgentCore</span>
    <span class="sidebar-section-count">${totalTools}</span>
  `;
  section.appendChild(header);

  const useChip = totalTools > 6;
  const grid = document.createElement('div');
  grid.className = `sidebar-tool-list${useChip ? ' chip-mode' : ''}`;

  const allTools = [];
  for (const mcp of controllers) {
    const tools = (mcp.tools || []).map(t => typeof t === 'string' ? { name: t } : t);
    for (const tool of tools) {
      if (!tool.type) tool.type = 'controller';
      tool._mcp = mcp;
      allTools.push(tool);
    }
  }
  _renderGroupedTools(grid, null, allTools, useChip);

  section.appendChild(grid);
  return section;
}

// ── Grouped rendering helpers ─────────────────────────────────────────────────

function _renderGroupedTools(grid, defaultMcp, tools, useChip) {
  const groups = {};
  for (const tool of tools) {
    const t = tool.type || '';
    (groups[t] = groups[t] || []).push(tool);
  }
  for (const type of _TYPE_ORDER) {
    if (!groups[type]?.length) continue;
    grid.appendChild(_buildSubgroupLabel(type, groups[type].length));
    for (const tool of groups[type]) {
      const mcp = tool._mcp || defaultMcp;
      grid.appendChild(useChip ? _buildChip(mcp, tool) : _buildToolCard(mcp, tool));
    }
  }
}

function _buildSubgroupLabel(type, count) {
  const label = document.createElement('div');
  label.className = 'sidebar-subgroup-label';
  const displayName = type ? type.toUpperCase() : 'OTHER';
  label.textContent = `${displayName} · ${count}`;
  return label;
}

function _buildChip(mcp, tool) {
  const chip = document.createElement('div');
  const toolType = tool.type || '';
  chip.className = `sidebar-chip${toolType ? ' type-' + toolType : ''}`;
  chip.draggable = true;
  chip.dataset.mcpId = mcp.id;
  chip.dataset.toolName = tool.name;
  chip.dataset.desc = (tool.description || '').toLowerCase();
  chip.title = tool.description || tool.name;

  // Config button for tools with shared fields
  const configSchema = typeof tool === 'object' ? tool.configSchema : null;
  const hasSharedFields = configSchema && Object.values(configSchema.properties || {}).some(def => def.scope !== 'instance');
  const configKey = `${mcp.id}:${tool.name}`;
  const configured = hasSharedFields ? !!_toolConfigs[configKey] : true;

  const configBtnHtml = hasSharedFields
    ? `<button class="chip-config-btn" title="配置"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-1.42 3.42 2 2 0 0 1-1.42-.58l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-3.42-1.42 2 2 0 0 1 .58-1.42l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 1.42-3.42 2 2 0 0 1 1.42.58l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1.08 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 3.42 1.42 2 2 0 0 1-.58 1.42l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1.08z"/></svg></button>`
    : '';
  chip.innerHTML = `<span class="chip-name">${_esc(tool.name)}</span>${configBtnHtml}`;

  // Config button click
  if (hasSharedFields) {
    chip.querySelector('.chip-config-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      _openToolConfigModal(mcp.id, tool.name, configSchema);
    });
  }

  // Click to show detail
  chip.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    _showDetail(mcp, tool);
  });

  chip.addEventListener('dragstart', (e) => {
    chip.classList.add('dragging-source');
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('application/x-cap-card', JSON.stringify({
      mcpId: mcp.id, toolName: tool.name, driverName: mcp.server_name || mcp.name || mcp.id,
      hasConfig: !!hasSharedFields, configured,
      multiInstance: !!(tool.multiInstance),
      hasInstanceConfig: _hasInstanceFields(configSchema),
    }));
  });
  chip.addEventListener('dragend', () => chip.classList.remove('dragging-source'));

  return chip;
}

// ── Search filter ─────────────────────────────────────────────────────────────

function _onSearchInput(query) {
  if (!_scroll) return;
  const q = query.trim().toLowerCase();

  // Filter cards and chips
  const items = _scroll.querySelectorAll('.sidebar-tool-card, .sidebar-chip');
  for (const el of items) {
    const name = (el.dataset.toolName || '').toLowerCase();
    const desc = (el.dataset.desc || '').toLowerCase();
    const match = !q || name.includes(q) || desc.includes(q);
    el.classList.toggle('hidden', !match);
  }

  // Hide subgroup labels if all items in that group are hidden
  const labels = _scroll.querySelectorAll('.sidebar-subgroup-label');
  for (const label of labels) {
    let next = label.nextElementSibling;
    let hasVisible = false;
    while (next && !next.classList.contains('sidebar-subgroup-label')) {
      if ((next.classList.contains('sidebar-tool-card') || next.classList.contains('sidebar-chip')) && !next.classList.contains('hidden')) {
        hasVisible = true;
        break;
      }
      next = next.nextElementSibling;
    }
    label.classList.toggle('hidden', !hasVisible);
  }

  // Hide sections if all their content is hidden
  const sections = _scroll.querySelectorAll('.sidebar-section');
  for (const sec of sections) {
    const visibleItems = sec.querySelectorAll('.sidebar-tool-card:not(.hidden), .sidebar-chip:not(.hidden)');
    sec.classList.toggle('hidden', visibleItems.length === 0);
  }
}

// ── Tool card ─────────────────────────────────────────────────────────────────

function _buildToolCard(mcp, tool) {
  const card = document.createElement('div');
  const toolType = tool.type || '';
  card.className = `sidebar-tool-card${toolType ? ' type-' + toolType : ''}`;
  card.draggable = true;
  card.dataset.mcpId = mcp.id;
  card.dataset.toolName = tool.name;
  card.dataset.desc = (tool.description || '').toLowerCase();

  const configSchema = typeof tool === 'object' ? tool.configSchema : null;
  const configKey = `${mcp.id}:${tool.name}`;
  // Only tools with shared (non-instance-scope) fields need sidebar-level configuration
  const hasSharedFields = configSchema && Object.values(configSchema.properties || {}).some(def => def.scope !== 'instance');
  const configured = hasSharedFields ? !!_toolConfigs[configKey] : true;

  // Badge
  const badgeHtml = toolType
    ? `<span class="cap-type-badge ${_esc(toolType)}">${_esc(toolType)}</span>`
    : '';

  // Config status indicator — only for tools that have shared fields to configure
  const configHtml = hasSharedFields
    ? `<span class="tool-card-config-status ${configured ? 'configured' : 'unconfigured'}" title="${configured ? 'Configured' : 'Not configured'}">${configured ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>' : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'}</span>`
    : '';

  // Description (truncated)
  const desc = tool.description || '';
  const descHtml = desc ? `<div class="tool-card-desc" title="${_esc(desc)}">${_esc(desc)}</div>` : '';

  card.innerHTML = `
    <div class="tool-card-header">
      <div class="tool-card-title-row">
        ${badgeHtml}
        <span class="tool-card-name" title="${_esc(tool.name)}">${_esc(tool.name)}</span>
      </div>
      <div class="tool-card-actions">
        ${hasSharedFields ? '<button class="tool-card-config-btn" title="配置"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-1.42 3.42 2 2 0 0 1-1.42-.58l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-3.42-1.42 2 2 0 0 1 .58-1.42l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 1.42-3.42 2 2 0 0 1 1.42.58l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1.08 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 3.42 1.42 2 2 0 0 1-.58 1.42l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1.08z"/></svg></button>' : ''}
        <button class="tool-card-info-btn" title="详情"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></button>
      </div>
    </div>
    ${descHtml}
    ${configHtml}
  `;

  // Detail button
  card.querySelector('.tool-card-info-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    _showDetail(mcp, tool);
  });

  // Config button (only rendered for tools with shared fields)
  if (hasSharedFields) {
    card.querySelector('.tool-card-config-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      _openToolConfigModal(mcp.id, tool.name, configSchema);
    });
  }

  // Drag (for canvas drop)
  card.addEventListener('dragstart', (e) => {
    card.classList.add('dragging-source');
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('application/x-cap-card', JSON.stringify({
      mcpId: mcp.id, toolName: tool.name, driverName: mcp.server_name || mcp.name || mcp.id,
      hasConfig: !!hasSharedFields, configured,
      multiInstance: !!(tool.multiInstance),
      hasInstanceConfig: _hasInstanceFields(configSchema),
    }));
  });
  card.addEventListener('dragend', () => card.classList.remove('dragging-source'));

  return card;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Check if a configSchema has any instance-scope fields. */
function _hasInstanceFields(configSchema) {
  if (!configSchema || !configSchema.properties) return false;
  return Object.values(configSchema.properties).some(def => def.scope === 'instance');
}

/** Check if a configSchema has any shared-scope required fields. */
export function hasSharedRequired(configSchema) {
  if (!configSchema) return false;
  const props = configSchema.properties || {};
  const required = configSchema.required || [];
  return required.some(k => props[k] && props[k].scope !== 'instance');
}

// ── Tool config modal (shared fields only) ────────────────────────────────────

function _openToolConfigModal(mcpId, toolName, configSchema) {
  if (isProjectRunning()) {
    alert('Stop agent before modifying');
    return;
  }
  const overlay = document.getElementById('tool-config-overlay');
  const titleEl = document.getElementById('tool-config-title');
  const bodyEl  = document.getElementById('tool-config-body');
  const saveBtn = document.getElementById('tool-config-save');

  titleEl.textContent = `Configure ${toolName}`;
  bodyEl.innerHTML = '';

  const props = configSchema.properties || {};
  const required = configSchema.required || [];
  const configKey = `${mcpId}:${toolName}`;
  const savedValues = _toolConfigs[configKey] || {};

  // If all fields are instance-scope, show them here too (as shared defaults)
  const hasSharedFields = Object.values(props).some(d => d.scope !== 'instance');

  for (const [key, def] of Object.entries(props)) {
    // Skip instance-scope fields only if there are also shared fields
    if (hasSharedFields && def.scope === 'instance') continue;
    const label = document.createElement('label');
    label.className = 'tool-config-label';
    label.textContent = `${def.title || def.description || key}${required.includes(key) ? ' *' : ''}`;

    let input;
    if (def.oneOf && Array.isArray(def.oneOf)) {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = `-- Select --`;
      input.appendChild(placeholder);
      for (const item of def.oneOf) {
        const option = document.createElement('option');
        option.value = item.const ?? '';
        option.textContent = item.title || String(item.const);
        if (String(savedValues[key]) === String(item.const)) option.selected = true;
        input.appendChild(option);
      }
      if (savedValues[key] != null) input.value = savedValues[key];
    } else if (def.enum && Array.isArray(def.enum)) {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const effectiveValue = savedValues[key] ?? def.default ?? '';
      if (!effectiveValue) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = `-- Select --`;
        input.appendChild(placeholder);
      }
      for (const opt of def.enum) {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        if (effectiveValue === opt) option.selected = true;
        input.appendChild(option);
      }
      input.value = effectiveValue;
    } else if (def.type === 'boolean') {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const optTrue = document.createElement('option');
      optTrue.value = 'true'; optTrue.textContent = 'Yes';
      const optFalse = document.createElement('option');
      optFalse.value = 'false'; optFalse.textContent = 'No';
      input.appendChild(optTrue);
      input.appendChild(optFalse);
      input.value = (savedValues[key] != null ? String(savedValues[key]) : String(def.default ?? 'false'));
    } else {
      input = document.createElement('input');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      input.type = def.format === 'password' ? 'password' : 'text';
      input.placeholder = def.default || '';
      input.value = savedValues[key] || '';
    }

    bodyEl.appendChild(label);
    bodyEl.appendChild(input);
  }

  // Handlers
  const close = () => { overlay.classList.add('hidden'); };
  const save = async () => {
    const values = {};
    bodyEl.querySelectorAll('[data-key]').forEach(input => {
      const v = input.value.trim();
      if (!v) return;
      const fieldDef = props[input.dataset.key];
      if (fieldDef?.type === 'integer') values[input.dataset.key] = parseInt(v, 10);
      else if (fieldDef?.type === 'number') values[input.dataset.key] = parseFloat(v);
      else if (fieldDef?.type === 'boolean') values[input.dataset.key] = v === 'true';
      else values[input.dataset.key] = v;
    });

    // Save to per-tool API
    try {
      await fetch(`/api/canvas/tool-config/${encodeURIComponent(mcpId)}/${encodeURIComponent(toolName)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
    } catch (err) { console.error('[config] save failed:', err); }

    // Update local cache
    _toolConfigs[configKey] = values;

    // Update sidebar card status indicator
    const card = _scroll.querySelector(`.sidebar-tool-card[data-mcp-id="${mcpId}"][data-tool-name="${toolName}"]`);
    if (card) {
      const statusEl = card.querySelector('.tool-card-config-status');
      if (statusEl) {
        statusEl.className = 'tool-card-config-status configured';
        statusEl.textContent = '✓';
        statusEl.title = 'Configured';
      }
    }

    close();
  };

  // Replace old listeners
  const newSaveBtn = saveBtn.cloneNode(true);
  saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
  newSaveBtn.addEventListener('click', save);

  // Close / cancel buttons
  const closeBtn = document.getElementById('tool-config-close');
  if (closeBtn) {
    const newCloseBtn = closeBtn.cloneNode(true);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    newCloseBtn.addEventListener('click', close);
  }
  const cancelBtn = document.getElementById('tool-config-cancel');
  if (cancelBtn) {
    const newCancelBtn = cancelBtn.cloneNode(true);
    cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);
    newCancelBtn.addEventListener('click', close);
  }

  overlay.classList.remove('hidden');
}

// ── Detail modal ──────────────────────────────────────────────────────────────

export function showToolDetail(mcp, tool, opts = {}) { _showDetail(mcp, tool, opts); }

function _showDetail(mcp, tool, opts = {}) {
  const title = document.getElementById('tool-detail-title');
  const body = document.getElementById('tool-detail-body');

  // Support plain string tool names
  const toolName = typeof tool === 'string' ? tool : tool.name;
  title.textContent = toolName;

  // Use overridden topics if provided (canvas instantiation), else fall back to tool definition
  const topicInData  = opts.topicIn  || (typeof tool === 'object' ? tool.topic_in  : null) || [];
  const topicOutData = opts.topicOut || (typeof tool === 'object' ? tool.topic_out : null) || [];
  const schema       = typeof tool === 'object' && tool.inputSchema ? JSON.stringify(tool.inputSchema, null, 2) : null;
  const description  = typeof tool === 'object' ? tool.description : null;

  const topicIn  = topicInData.map(t => `<li><code title="${_esc(t.topic || '?')}">${_esc(t.topic || '?')}</code> <span class="detail-fmt">${_esc(t.format || '')}</span></li>`).join('');
  const topicOut = topicOutData.map(t => `<li><code title="${_esc(t.topic || '?')}">${_esc(t.topic || '?')}</code> <span class="detail-fmt">${_esc(t.format || '')}</span></li>`).join('');

  body.innerHTML = `
    ${description ? `<div class="detail-section"><div class="detail-label">描述</div><div class="detail-text">${_esc(description)}</div></div>` : ''}
    <div class="detail-section">
      <div class="detail-label">驱动</div>
      <div class="detail-text">${_esc(mcp.server_name || mcp.name || mcp.id)}</div>
    </div>
    ${topicIn ? `<div class="detail-section"><div class="detail-label">输入 Topics</div><ul class="detail-topics">${topicIn}</ul></div>` : ''}
    ${topicOut ? `<div class="detail-section"><div class="detail-label">输出 Topics</div><ul class="detail-topics">${topicOut}</ul></div>` : ''}
    ${schema ? `<div class="detail-section"><div class="detail-label">Input Schema</div><pre class="detail-schema">${_esc(schema)}</pre></div>` : ''}
  `;

  _backdrop.classList.remove('hidden');
}

function _hideDetail() {
  _backdrop.classList.add('hidden');
}

// ── Instance config modal (instance-scope fields only) ────────────────────────

/**
 * Open a config modal for a specific canvas card instance.
 * Only shows fields with scope === "instance".
 */
export function openInstanceConfigModal(mcpId, toolName, instanceId, configSchema) {
  if (isProjectRunning()) {
    alert('Stop agent before modifying');
    return;
  }
  const overlay = document.getElementById('tool-config-overlay');
  const titleEl = document.getElementById('tool-config-title');
  const bodyEl  = document.getElementById('tool-config-body');
  const saveBtn = document.getElementById('tool-config-save');

  titleEl.textContent = `Instance Config: ${toolName}`;
  bodyEl.innerHTML = '';

  const props = (configSchema && configSchema.properties) || {};
  const required = (configSchema && configSchema.required) || [];
  const configKey = `${mcpId}:${toolName}:${instanceId}`;
  const savedValues = _toolConfigs[configKey] || {};

  let hasFields = false;
  for (const [key, def] of Object.entries(props)) {
    // Only show instance-scope fields
    if (def.scope !== 'instance') continue;
    hasFields = true;

    const label = document.createElement('label');
    label.className = 'tool-config-label';
    label.textContent = `${def.title || def.description || key}${required.includes(key) ? ' *' : ''}`;

    let input;
    if (def.oneOf && Array.isArray(def.oneOf)) {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = `-- Select --`;
      input.appendChild(placeholder);
      for (const item of def.oneOf) {
        const option = document.createElement('option');
        option.value = item.const ?? '';
        option.textContent = item.title || String(item.const);
        if (String(savedValues[key]) === String(item.const)) option.selected = true;
        input.appendChild(option);
      }
      if (savedValues[key] != null) input.value = savedValues[key];
    } else if (def.enum && Array.isArray(def.enum)) {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const effectiveValue = savedValues[key] ?? def.default ?? '';
      if (!effectiveValue) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = `-- Select --`;
        input.appendChild(placeholder);
      }
      for (const opt of def.enum) {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        if (effectiveValue === opt) option.selected = true;
        input.appendChild(option);
      }
      input.value = effectiveValue;
    } else if (def.type === 'boolean') {
      input = document.createElement('select');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      const optTrue = document.createElement('option');
      optTrue.value = 'true'; optTrue.textContent = 'Yes';
      const optFalse = document.createElement('option');
      optFalse.value = 'false'; optFalse.textContent = 'No';
      input.appendChild(optTrue);
      input.appendChild(optFalse);
      input.value = (savedValues[key] != null ? String(savedValues[key]) : String(def.default ?? 'false'));
    } else {
      input = document.createElement('input');
      input.className = 'tool-config-input';
      input.dataset.key = key;
      input.type = def.format === 'password' ? 'password' : 'text';
      input.placeholder = def.default || '';
      input.value = savedValues[key] || '';
    }

    bodyEl.appendChild(label);
    bodyEl.appendChild(input);
  }

  if (!hasFields) {
    bodyEl.innerHTML = '<p style="color:var(--text-secondary)">No instance config fields</p>';
  }

  const close = () => { overlay.classList.add('hidden'); };
  const save = async () => {
    const values = {};
    bodyEl.querySelectorAll('[data-key]').forEach(input => {
      const v = input.value.trim();
      if (!v) return;
      const fieldDef = props[input.dataset.key];
      if (fieldDef?.type === 'integer') values[input.dataset.key] = parseInt(v, 10);
      else if (fieldDef?.type === 'number') values[input.dataset.key] = parseFloat(v);
      else if (fieldDef?.type === 'boolean') values[input.dataset.key] = v === 'true';
      else values[input.dataset.key] = v;
    });

    try {
      await fetch(`/api/canvas/tool-config/${encodeURIComponent(mcpId)}/${encodeURIComponent(toolName)}/${encodeURIComponent(instanceId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
    } catch (err) { console.error('[config] instance save failed:', err); }

    _toolConfigs[configKey] = values;
    close();
  };

  const newSaveBtn = saveBtn.cloneNode(true);
  saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
  newSaveBtn.addEventListener('click', save);

  const closeBtn = document.getElementById('tool-config-close');
  if (closeBtn) {
    const newCloseBtn = closeBtn.cloneNode(true);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    newCloseBtn.addEventListener('click', close);
  }
  const cancelBtn = document.getElementById('tool-config-cancel');
  if (cancelBtn) {
    const newCancelBtn = cancelBtn.cloneNode(true);
    cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);
    newCancelBtn.addEventListener('click', close);
  }

  overlay.classList.remove('hidden');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
