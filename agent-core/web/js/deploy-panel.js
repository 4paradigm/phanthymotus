/**
 * deploy-panel.js — 部署服务 modal
 *
 * 单一 modal，展示所有驱动类别。每张卡内联版本下拉选择器。
 * 底部确认按钮触发部署，支持同时操作多个驱动（stop 仍为卡片内即时操作）。
 */

let _overlay  = null;
let _polling  = null;

let _catalog  = { core: [], driver: [], perception: [], inspection: [] };
let _statuses = {};   // driver_id → { running, status }
let _logPolls = {};   // driver_id → intervalId

// { driverId → { registry_image, tag } }
let _pending = {};

export function initDeployPanel() {
  _overlay = document.getElementById('deploy-overlay');

  document.getElementById('btn-deploy').addEventListener('click', _open);
  document.getElementById('deploy-close').addEventListener('click', _close);
  document.getElementById('deploy-modal-confirm').addEventListener('click', _confirmAll);
  document.getElementById('hw-search').addEventListener('input', () => _renderHardwareSection(_catalog.driver));
}

function _open() {
  _pending = {};
  _overlay.classList.remove('hidden');
  _load();
  _polling = setInterval(_loadStatuses, 5000);
}

function _close() {
  _overlay.classList.add('hidden');
  clearInterval(_polling);
  _polling = null;
}

// ── Data loading ──────────────────────────────────────────────────────────

async function _load() {
  // Auto-sync from registry on every open (transparent to user)
  try {
    await fetch('/api/drivers/sync', { method: 'POST' });
  } catch { /* ignore */ }
  await Promise.all([_loadCatalog(true), _loadStatuses()]);
  _render();
}

async function _loadCatalog(refresh = false) {
  try {
    const url  = refresh ? '/api/registry/catalog?refresh=true' : '/api/registry/catalog';
    const res  = await fetch(url);
    const json = await res.json();
    if (json.data) _catalog = json.data;
  } catch {
    // keep existing catalog
  }
}

async function _loadStatuses() {
  try {
    const res  = await fetch('/api/drivers');
    const json = await res.json();
    _statuses = {};
    for (const d of (json.data || [])) {
    _statuses[d.id] = {
        running:       d.running,
        status:        d.status,
        logs:          d.logs || '',
        running_image: d.running_image || '',
        image:         d.image || '',
        last_deploy:   d.last_deploy || null,
      };
    }
  } catch {
    // keep existing statuses
  }
  _updateStatusDots();
}

// ── Empty state illustrations ─────────────────────────────────────────────

const _EMPTY_ICONS = {
  core:       `<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="10" width="28" height="18" rx="3.5" stroke="currentColor" stroke-width="1.4"/><path d="M4 15h28" stroke="currentColor" stroke-width="1.2"/><circle cx="8.5" cy="12.5" r="1" fill="currentColor"/><circle cx="12" cy="12.5" r="1" fill="currentColor"/><rect x="9" y="19" width="8" height="4" rx="1.2" stroke="currentColor" stroke-width="1.1"/><rect x="19" y="19" width="8" height="4" rx="1.2" stroke="currentColor" stroke-width="1.1"/></svg>`,
  driver:     `<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="11" width="28" height="18" rx="3.5" stroke="currentColor" stroke-width="1.4"/><path d="M9 8v3M14 8v3M22 8v3M27 8v3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><rect x="9" y="17" width="7" height="5" rx="1.2" stroke="currentColor" stroke-width="1.1"/><rect x="20" y="17" width="7" height="5" rx="1.2" stroke="currentColor" stroke-width="1.1"/><circle cx="18" cy="26" r="1.2" fill="currentColor" opacity=".6"/></svg>`,
  perception: `<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="18" cy="18" r="9" stroke="currentColor" stroke-width="1.4"/><circle cx="18" cy="18" r="3.5" stroke="currentColor" stroke-width="1.2"/><path d="M18 4v4M18 28v4M4 18h4M28 18h4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`,
};
const _EMPTY_TITLE = {
  core:       '暂无核心服务驱动',
  driver:     '暂无硬件驱动',
  perception: '暂无感知服务驱动',
};
function _emptyStateHTML(category) {
  return `<div class="drivers-empty-state"><div class="drivers-empty-icon">${_EMPTY_ICONS[category] || _EMPTY_ICONS.driver}</div><div class="drivers-empty-title">${_EMPTY_TITLE[category] || '暂无可用驱动'}</div><div class="drivers-empty-hint">点击右上角「同步镜像」从镜像仓库拉取驱动</div></div>`;
}

// ── Rendering ─────────────────────────────────────────────────────────────

function _render() {
  _renderSection('drivers-core',       _catalog.core,       'core');
  _renderHardwareSection(_catalog.driver);
  _renderSection('drivers-perception', _catalog.perception, 'perception');
  _syncFooter();
}

function _renderSection(containerId, items, category) {
  const container = document.getElementById(containerId);
  if (!items || items.length === 0) {
    container.innerHTML = _emptyStateHTML(category);
    return;
  }
  container.innerHTML = items.map(item => _cardHTML(item, category)).join('');
  // Bind version selectors & stop buttons & upgrade buttons
  container.querySelectorAll('.vsel').forEach(vsel => _bindVsel(vsel));
  container.querySelectorAll('[data-action="stop"]').forEach(btn => {
    btn.addEventListener('click', () => _stopDriver(btn.dataset.driverId, btn));
  });
  container.querySelectorAll('[data-action="upgrade"]').forEach(btn => {
    btn.addEventListener('click', () => _showUpgradeConfirm(btn.dataset));
  });
}

// ── Hardware: provider accordion groups ────────────────────────────────────

function _renderHardwareSection(items) {
  const container = document.getElementById('drivers-hardware-inner');
  const q = (document.getElementById('hw-search')?.value || '').trim().toLowerCase();

  if (!items || items.length === 0) {
    container.innerHTML = _emptyStateHTML('driver');
    return;
  }

  // Group by provider
  const groups = {};
  for (const item of items) {
    const prov = item.provider || 'Unknown';
    (groups[prov] ??= []).push(item);
  }

  // Filter by search query; within each group, also filter items
  let entries = Object.entries(groups);
  let hasQuery = q.length > 0;
  if (hasQuery) {
    entries = entries
      .map(([prov, provItems]) => {
        const provMatch = prov.toLowerCase().includes(q);
        const filtered = provMatch
          ? provItems
          : provItems.filter(it => (it.model || '').toLowerCase().includes(q));
        return filtered.length ? [prov, filtered] : null;
      })
      .filter(Boolean);
  }

  if (entries.length === 0) {
    container.innerHTML = _emptyStateHTML('driver');
    return;
  }

  // Render; first group open by default (or all open when searching)
  container.innerHTML = entries
    .map(([prov, provItems], idx) => _providerGroupHTML(prov, provItems, hasQuery || idx === 0))
    .join('');

  // Bind accordion toggle
  container.querySelectorAll('.hw-provider-header').forEach(header => {
    header.addEventListener('click', () => {
      header.closest('.hw-provider-group').classList.toggle('open');
    });
  });

  // Bind vsel + stop buttons
  container.querySelectorAll('.vsel').forEach(vsel => _bindVsel(vsel));
  container.querySelectorAll('[data-action="stop"]').forEach(btn => {
    btn.addEventListener('click', () => _stopDriver(btn.dataset.driverId, btn));
  });
  container.querySelectorAll('[data-action="upgrade"]').forEach(btn => {
    btn.addEventListener('click', () => _showUpgradeConfirm(btn.dataset));
  });
}

function _providerGroupHTML(provider, items, open) {
  const cards = items.map(item => _cardHTML(item, 'driver')).join('');
  const chevron = `<svg class="hw-provider-chevron" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  return `
  <div class="hw-provider-group${open ? ' open' : ''}">
    <div class="hw-provider-header">
      <span class="hw-provider-left">
        <span>${provider}</span>
        <span class="hw-provider-count">${items.length}</span>
      </span>
      ${chevron}
    </div>
    <div class="hw-provider-body">${cards}</div>
  </div>`;
}


function _cardLabel(item, category) {
  if (category === 'driver') return item.model || item.image;
  return item.name || item.image;
}

function _driverIdForItem(item, category) {
  if (category === 'driver') return `${item.provider}-${item.model}`;
  return item.image;
}

function _vselHTML(driverId, imageBase, runningImage, tags, running, latestTag, deployedTag) {
  const opts = tags.map((t, idx) => {
    const fullImg   = t.imageRef || (imageBase + ':' + t.tag);
    const isCurrent = runningImage && runningImage.endsWith(':' + t.tag);
    // 仅在运行中时，已部署版本不可重新选择
    const isDisabled = isCurrent && running;
    const isLatest  = idx === 0;
    const dateStr   = t.created ? t.created.replace(/\s+\d{2}:\d{2}$/, '') : '';
    const timeStr   = t.created ? t.created.match(/\d{2}:\d{2}$/)?.[0] ?? '' : '';
    const dateLabel = dateStr + (timeStr ? ' ' + timeStr : '');
    return `<div class="vsel-option${isDisabled ? ' disabled' : ''}" data-value="${t.tag}" data-full-image="${fullImg}">
      <span class="vsel-option-tag">${t.tag}</span>
      ${dateLabel ? `<span class="vsel-option-date">${dateLabel}</span>` : ''}
      ${isCurrent ? `<span class="vsel-option-current">已部署</span>` : ''}
      ${isLatest && !isCurrent ? `<span class="vsel-option-latest">最新</span>` : ''}
    </div>`;
  }).join('');
  // 若有已部署版本，按钮初始显示该版本名
  const btnLabel = deployedTag ? `${running ? '▶' : '⏹'} ${deployedTag}` : '— 选择版本 —';
  const hasValue = !!deployedTag;
  return `<div class="vsel${hasValue ? ' has-value' : ''}" data-driver-id="${driverId}" data-image-base="${imageBase}">
    <button type="button" class="vsel-btn">
      <span class="vsel-label">${btnLabel}</span>
      <svg class="vsel-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 4l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
    <div class="vsel-dropdown">${opts}</div>
  </div>`;
}

function _cardHTML(item, category) {
  const label    = _cardLabel(item, category);
  const driverId = _driverIdForItem(item, category);
  const s        = _statuses[driverId] || {};
  const running  = s.running || false;
  const errored  = s.status === 'error';
  const tags     = item.tags || [];
  const imageBase = item.full_repo || item.image;
  const runningImage = s.running_image || '';

  // core: no status dot, no stop button — version selector replaces deployed tag
  if (category === 'core') {
    const currentTag = runningImage.includes(':') ? runningImage.split(':').pop() : runningImage;
    const versionCtrl = tags.length === 0
      ? `<span style="font-size:11px;color:var(--text-dim)">暂无版本</span>`
      : _vselHTML(driverId, imageBase, runningImage, tags, false, tags.length > 0 ? tags[0].tag : null, currentTag);
    return `
    <div class="driver-card" id="card-${driverId}">
      <div class="driver-card-info">
        <div class="driver-card-name">${label}</div>
        <div class="driver-card-image">${imageBase}</div>
        <div class="driver-card-vsel">${versionCtrl}</div>
      </div>
    </div>
    <div class="deploy-log hidden" id="log-${driverId}"></div>`;
  }

  const statusClass = running ? 'running' : errored ? 'error' : 'stopped';

  // Determine currently deployed tag for non-core drivers
  let deployedTag = '';
  if (runningImage) {
    deployedTag = runningImage.includes(':') ? runningImage.split(':').pop() : runningImage;
  } else if (s.last_deploy?.image) {
    deployedTag = s.last_deploy.image.includes(':') ? s.last_deploy.image.split(':').pop() : s.last_deploy.image;
  }

  // 检测是否有新版本（tags[0] 为最新）
  const latestTag = tags.length > 0 ? tags[0].tag : null;
  const currentTag = runningImage?.includes(':') ? runningImage.split(':').pop() : null;
  const hasNewVersion = latestTag && currentTag && latestTag !== currentTag;

  const versionCtrl = tags.length === 0
    ? `<span style="font-size:11px;color:var(--text-dim)">暂无版本</span>`
    : _vselHTML(driverId, imageBase, runningImage, tags, running, latestTag, deployedTag);

  const stopBtn = running
    ? `<button class="btn-deploy-action running" data-action="stop" data-driver-id="${driverId}">停止</button>`
    : '';

  return `
    <div class="driver-card" id="card-${driverId}">
      <div class="driver-status-dot ${statusClass}" id="dot-${driverId}"></div>
      <div class="driver-card-info">
        <div class="driver-card-name">${label}${running ? `<span class="driver-running-badge">运行中</span>` : ''}${hasNewVersion ? `<span class="btn-upgrade" data-action="upgrade" data-driver-id="${driverId}" data-current-tag="${currentTag}" data-latest-tag="${latestTag}" data-latest-image="${imageBase}:${latestTag}" data-label="${label}">有新版本</span>` : ''}</div>
        <div class="driver-card-image">${imageBase}</div>
        <div class="driver-card-vsel">${versionCtrl}</div>
      </div>
      ${stopBtn ? `<div class="driver-card-ctrl">${stopBtn}</div>` : ''}
    </div>
    <div class="deploy-log hidden" id="log-${driverId}"></div>`;
}

function _updateStatusDots() {
  for (const [id, s] of Object.entries(_statuses)) {
    const dot = document.getElementById(`dot-${id}`);
    if (dot) {
      dot.className = 'driver-status-dot ' + (s.running ? 'running' : s.status === 'error' ? 'error' : 'stopped');
    }
  }
}

// ── Version select ─────────────────────────────────────────────────────────

function _bindVsel(vsel) {
  const btn      = vsel.querySelector('.vsel-btn');
  const dropdown = vsel.querySelector('.vsel-dropdown');

  btn.addEventListener('click', e => {
    e.stopPropagation();
    const isOpen = vsel.classList.contains('open');
    // Close all others
    document.querySelectorAll('.vsel.open').forEach(v => v.classList.remove('open'));
    if (!isOpen) vsel.classList.add('open');
  });

  dropdown.querySelectorAll('.vsel-option').forEach(opt => {
    opt.addEventListener('click', () => {
      if (opt.classList.contains('disabled')) return;

      const tag       = opt.dataset.value;
      const fullImage = opt.dataset.fullImage;
      const driverId  = vsel.dataset.driverId;

      // Update label & selected state
      dropdown.querySelectorAll('.vsel-option').forEach(o => o.classList.remove('selected'));
      opt.classList.add('selected');
      vsel.querySelector('.vsel-label').textContent = tag;
      vsel.classList.add('has-value');
      vsel.classList.remove('open');

      if (tag) {
        _pending[driverId] = { image: fullImage };
      } else {
        delete _pending[driverId];
      }
      _syncFooter();
    });
  });
}

// Close dropdowns when clicking outside
document.addEventListener('click', () => {
  document.querySelectorAll('.vsel.open').forEach(v => v.classList.remove('open'));
});

function _onVersionChange(sel) {
  const driverId  = sel.dataset.driverId;
  const tag       = sel.value;
  const opt       = sel.selectedIndex >= 0 ? sel.options[sel.selectedIndex] : null;
  // Prefer the full imageRef stored on the option; fall back to building from imageBase
  const fullImage = (opt && opt.dataset.fullImage) || (sel.dataset.imageBase + ':' + tag);

  if (tag) {
    _pending[driverId] = { image: fullImage };
  } else {
    delete _pending[driverId];
  }
  _syncFooter();
}

function _syncFooter() {
  const footer = document.getElementById('deploy-modal-footer');
  const hint   = document.getElementById('deploy-footer-hint');
  const count  = Object.keys(_pending).length;
  if (count > 0) {
    footer.style.display = '';
    hint.textContent = `已选 ${count} 个驱动`;
  } else {
    footer.style.display = 'none';
  }
}

// ── Deploy confirm modal (shared) ─────────────────────────────────────────

/**
 * Show the unified deploy-confirm modal.
 * @param {Array<{label: string, currentTag: string, newTag: string}>} items
 * @param {Function} onConfirm - called when user clicks confirm
 */
export function showDeployConfirmModal(items, onConfirm) {
  const overlay = document.getElementById('deploy-confirm-overlay');
  const body    = document.getElementById('deploy-confirm-body');

  body.innerHTML = items.map(it => `
    <div class="deploy-confirm-item">
      <div class="deploy-confirm-item-name">${it.label}</div>
      <div class="deploy-confirm-item-versions">
        <span class="deploy-confirm-tag current">${it.currentTag || '—'}</span>
        <span class="deploy-confirm-arrow">→</span>
        <span class="deploy-confirm-tag latest">${it.newTag}</span>
      </div>
    </div>`).join('');

  overlay.classList.remove('hidden');

  const btnOk     = document.getElementById('deploy-confirm-ok');
  const btnCancel = document.getElementById('deploy-confirm-cancel');

  const cleanup = () => {
    btnOk.removeEventListener('click', doConfirm);
    btnCancel.removeEventListener('click', doCancel);
  };
  const doConfirm = () => { overlay.classList.add('hidden'); cleanup(); onConfirm(); };
  const doCancel  = () => { overlay.classList.add('hidden'); cleanup(); };

  btnOk.addEventListener('click', doConfirm);
  btnCancel.addEventListener('click', doCancel);
}

// ── Upgrade confirm (single driver → deploy immediately) ──────────────────

function _showUpgradeConfirm({ driverId, currentTag, latestTag, latestImage, label }) {
  showDeployConfirmModal(
    [{ label, currentTag, newTag: latestTag }],
    () => {
      // Deploy directly
      _executeDeploys([[driverId, { image: latestImage }]]);
    }
  );
}

// ── Stop ──────────────────────────────────────────────────────────────────

async function _stopDriver(driverId, btn) {
  btn.disabled    = true;
  btn.textContent = '停止中…';
  try {
    await fetch(`/api/drivers/${driverId}/stop`, { method: 'POST' });
  } catch (e) {
    console.error('[deploy] stop', e);
  }
  setTimeout(_loadStatuses, 1500);
}

// ── Confirm all pending deploys ───────────────────────────────────────────

async function _confirmAll() {
  const entries = Object.entries(_pending);
  if (!entries.length) return;

  // Build items for the confirm modal
  const items = entries.map(([id, { image }]) => {
    const s = _statuses[id] || {};
    const currentTag = s.running_image?.includes(':') ? s.running_image.split(':').pop() : '—';
    const newTag = image.split(':').pop();
    let label = id;
    for (const cat of [_catalog.core, _catalog.driver, _catalog.perception]) {
      for (const item of (cat || [])) {
        const cid = _driverIdForItem(item, cat === _catalog.driver ? 'driver' : 'core');
        if (cid === id) { label = _cardLabel(item, cat === _catalog.driver ? 'driver' : 'core'); break; }
      }
    }
    return { label, currentTag, newTag };
  });

  showDeployConfirmModal(items, () => {
    _executeDeploys(entries);
  });
}

async function _executeDeploys(entries) {
  document.getElementById('deploy-modal-confirm').disabled = true;

  for (const [driverId, { image }] of entries) {
    const isCoreDriver = _catalog.core.some(item => _driverIdForItem(item, 'core') === driverId);

    if (isCoreDriver) {
      _showDeployLog(driverId, '正在启动升级…');
      try {
        const res = await fetch('/api/system/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const json = await res.json();
        if (json.code !== 200) {
          _appendLog(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
        } else {
          _appendLog(driverId, '升级任务已启动，拉取镜像中…');
          _startCoreUpdatePolling(driverId);
        }
      } catch (e) {
        _appendLog(driverId, `✗ 网络错误: ${e.message}`, 'error');
      }
    } else {
      _showDeployLog(driverId, '正在请求部署…');
      try {
        const res  = await fetch(`/api/drivers/${driverId}/deploy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const json = await res.json();
        if (json.code !== 200) {
          _appendLog(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
        } else {
          _appendLog(driverId, '容器启动中…');
          _startLogPolling(driverId);
        }
      } catch (e) {
        _appendLog(driverId, `✗ 网络错误: ${e.message}`, 'error');
      }
    }
  }

  _pending = {};
  _syncFooter();
  document.getElementById('deploy-modal-confirm').disabled = false;
}

// ── Deploy log (inline) ───────────────────────────────────────────────────

function _showDeployLog(driverId, msg) {
  const el = document.getElementById(`log-${driverId}`);
  if (!el) return;
  el.innerHTML = `<div class="deploy-log-line">${msg}</div>`;
  el.classList.remove('hidden');
}

function _appendLog(driverId, msg, type = '') {
  const el = document.getElementById(`log-${driverId}`);
  if (!el) return;
  const line = document.createElement('div');
  line.className = 'deploy-log-line' + (type ? ` ${type}` : '');
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function _startLogPolling(driverId) {
  if (_logPolls[driverId]) clearInterval(_logPolls[driverId]);

  let attempts = 0;
  _logPolls[driverId] = setInterval(async () => {
    attempts++;
    try {
      const res  = await fetch(`/api/drivers/${driverId}/status`);
      const json = await res.json();
      const data = json.data || {};
      const status = data.status || '';
      const logs   = data.logs   || '';

      const el = document.getElementById(`log-${driverId}`);
      if (el && logs) {
        const lines = logs.trim().split('\n').slice(-5);
        el.querySelectorAll('.log-output').forEach(e => e.remove());
        const pre = document.createElement('pre');
        pre.className = 'log-output';
        pre.textContent = lines.join('\n');
        el.appendChild(pre);
      }

      if (status === 'running') {
        _stopLogPolling(driverId);
        _appendLog(driverId, '✓ 运行中', 'success');
        setTimeout(() => {
          const logEl = document.getElementById(`log-${driverId}`);
          if (logEl) logEl.classList.add('hidden');
        }, 5000);
        _loadStatuses();
      } else if (status === 'error' || attempts > 30) {
        _stopLogPolling(driverId);
        _appendLog(driverId, `✗ ${status === 'error' ? (data.error || '启动失败') : '部署超时'}`, 'error');
      }
    } catch {
      // ignore
    }
  }, 2000);
}

function _stopLogPolling(driverId) {
  if (_logPolls[driverId]) {
    clearInterval(_logPolls[driverId]);
    delete _logPolls[driverId];
  }
}

// ── Core update polling (uses /api/system/update-status) ─────────────────

function _startCoreUpdatePolling(driverId) {
  if (_logPolls[driverId]) clearInterval(_logPolls[driverId]);

  let attempts = 0;
  _logPolls[driverId] = setInterval(async () => {
    attempts++;
    try {
      const res  = await fetch('/api/system/update-status');
      const json = await res.json();
      const data = json.data || {};

      if (data.error) {
        _stopLogPolling(driverId);
        _appendLog(driverId, `✗ 升级失败：${data.error}`, 'error');
      } else if (data.step) {
        // 更新进度显示
        const el = document.getElementById(`log-${driverId}`);
        if (el) {
          el.querySelectorAll('.log-output').forEach(e => e.remove());
          const pre = document.createElement('div');
          pre.className = 'log-output';
          pre.textContent = data.step;
          el.appendChild(pre);
        }
      }

      if (attempts > 90) {
        _stopLogPolling(driverId);
        _appendLog(driverId, '✗ 升级超时', 'error');
      }
    } catch {
      // 服务重启中，连接断开是正常的 — 等待重连
    }
  }, 2000);
}
