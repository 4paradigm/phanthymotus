/**
 * deploy-panel.js — 部署服务 modal
 *
 * 双 Tab 设计：
 *   Tab 1 「我的服务」— 管理已安装的服务（升级/停止/启动/卸载）
 *   Tab 2 「驱动市场」— 浏览和安装新驱动（flat grid + filter chips）
 */

let _overlay  = null;
let _polling  = null;

let _catalog  = { core: [], perception: [], actucore: [], driver: [] };

// Fallback descriptions, used when resource-center didn't supply one
const _CAT_DESC = {
  perception: '语音感知套件 — ASR 语音识别 + TTS 语音合成 + VAD 静音检测 + 唤醒词检测',
  actucore:   '执行模型层 — VLA 策略 / 导航 / 抓取 / locomotion / 全身控制',
};
let _statuses = {};   // driver_id → { running, status, running_image, image, last_deploy }

// 本机架构 facet，由 /api/registry/catalog 返回；resource-center 据此过滤掉跑不了的镜像。
// _filter.applied 为 false 说明服务端不认识这两个参数（老版本），此时目录未经过滤。
let _facets = null;   // { acc_arch, cpu_arch }
let _filter = { applied: false, hidden_tags: 0, hidden_images: 0 };

let _logPolls = {};   // driver_id → intervalId
let _currentChannel = 'ga'; // mirrors config.core.update_channel; kept in sync by _loadChannel/_onChannelChange

// { driverId → { image } }
let _pending = {};

let _activeTab = 'my-services';
let _activeFilter = null; // null = all providers

export function initDeployPanel() {
  _overlay = document.getElementById('deploy-overlay');

  document.getElementById('btn-deploy').addEventListener('click', _open);
  document.getElementById('deploy-close').addEventListener('click', _close);
  document.getElementById('deploy-modal-confirm').addEventListener('click', _confirmAll);

  // Tab switching
  _overlay.querySelectorAll('.deploy-tab').forEach(tab => {
    tab.addEventListener('click', () => _switchTab(tab.dataset.tab));
  });

  // Marketplace search
  document.getElementById('marketplace-search').addEventListener('input', _renderMarketplace);

  // Channel selector
  const channelSelect = document.getElementById('deploy-channel-select');
  channelSelect.addEventListener('change', _onChannelChange);
  _loadChannel();
}

// ── Channel management ────────────────────────────────────────────────────

async function _loadChannel() {
  try {
    const res = await fetch('/api/config/update-channel');
    const json = await res.json();
    const channel = json.data?.channel || 'ga';
    document.getElementById('deploy-channel-select').value = channel;
    _currentChannel = channel;
  } catch { /* keep default */ }
}

async function _onChannelChange(e) {
  const channel = e.target.value;
  const warnings = {
    preview: '预览版可能不稳定，仅建议用于测试环境。确定切换？',
    release: '正式版已通过基础测试，但未经长期稳定性验证。确定切换？',
  };
  if (warnings[channel] && !confirm(warnings[channel])) {
    await _loadChannel();
    return;
  }
  try {
    await fetch('/api/config/update-channel', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel }),
    });
    _currentChannel = channel;
    await _loadCatalog(true);
    _render();
  } catch { /* ignore */ }
}

// Versions visible per channel. Release also shows ga (a stable fallback);
// preview is deliberately NOT inclusive of release/ga — mixing in the far
// more sparsely-published stable tags just buries the preview builds you're
// there to see. Anything not in this map's active list is hidden, not merely
// re-labelled — resource-center's own channel param already narrows what it
// returns, this is the client's independent guarantee that the version list
// never shows a build outside the selected channel.
const _CHANNEL_TAGS = {
  ga:      ['ga'],
  release: ['release', 'ga'],
  preview: ['preview'],
};

function _channelTags(item) {
  const allowed = _CHANNEL_TAGS[_currentChannel] || _CHANNEL_TAGS.ga;
  return (item.tags || []).filter(t => allowed.includes(t.channel));
}

// ── Tab switching ─────────────────────────────────────────────────────────

function _switchTab(tabId) {
  _activeTab = tabId;
  _overlay.querySelectorAll('.deploy-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabId);
  });
  _overlay.querySelectorAll('.deploy-tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${tabId}`);
  });
  _render();
}

// ── Open / Close ──────────────────────────────────────────────────────────

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
    if (json.facets) _facets = json.facets;
    // Only applied=true when resource-center actually understands the arch params — an
    // older one returns everything, and we must not claim the list was filtered.
    if (json.filter) _filter = json.filter;
  } catch { /* keep existing */ }
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
        name:          d.name || '',
        category:      d.category || 'driver',
      };
    }
  } catch { /* keep existing */ }
  // Update dots if visible
  _updateStatusDots();
}

// ── Rendering ─────────────────────────────────────────────────────────────

function _render() {
  if (_activeTab === 'my-services') {
    _renderMyServices();
  } else {
    _renderMarketplace();
  }
  _syncFooter();
}

// ══════════════════════════════════════════════════════════════════════════
//  TAB 1: 我的服务
// ══════════════════════════════════════════════════════════════════════════

function _renderMyServices() {
  const container = document.getElementById('pane-my-services');

  // Collect all items from catalog that have a status (i.e., deployed)
  const allItems = [
    ...(_catalog.core || []).map(it => ({ ...it, _cat: 'core' })),
    ...(_catalog.perception || []).map(it => ({ ...it, _cat: 'perception' })),
    ...(_catalog.actucore || []).map(it => ({ ...it, _cat: 'actucore' })),
    ...(_catalog.driver || []).map(it => ({ ...it, _cat: 'driver' })),
  ];

  // Only show items that have actually been deployed (not just synced from catalog)
  const deployed = allItems.filter(item => {
    const id = _driverIdForItem(item, item._cat);
    const s = _statuses[id];
    if (!s) return false;
    return s.running || s.last_deploy || item._cat === 'core';
  });

  // Orphan guard: the catalog is arch-filtered by resource-center, so a service that
  // was installed with a wrong-arch image (e.g. a jp5 perception on a JetPack 6 robot,
  // installed before that filtering existed) has no catalog entry any more. Without
  // this it would vanish from 我的服务 entirely — no way to stop, uninstall, or read
  // its logs. The manifest and GET /api/drivers still know about it, so rebuild a
  // minimal item from the status. tags: [] makes _svcRowHTML drop the version switcher
  // while keeping 启动 / 停止 / 卸载 / 日志.
  const seenIds = new Set(deployed.map(it => _driverIdForItem(it, it._cat)));
  for (const [id, s] of Object.entries(_statuses)) {
    if (seenIds.has(id) || !(s.running || s.last_deploy)) continue;
    const friendly = s.name || id;
    deployed.push({
      _cat:      s.category || 'driver',
      _id:       id,
      _orphan:   true,
      // _svcRowHTML 取 model（driver）或 name（其余），两个都给，否则会退化成裸镜像地址
      name:      friendly,
      model:     friendly,
      image:     id,
      full_repo: (s.image || s.running_image || '').split(':')[0],
      tags:      [],
    });
  }

  if (deployed.length === 0) {
    container.innerHTML = `<div class="svc-empty">
      <div class="svc-empty-title">暂无已安装服务</div>
      <div class="svc-empty-hint">前往「驱动市场」安装驱动</div>
    </div>`;
    return;
  }

  // Split into groups: updatable, running, stopped
  const updatable = [];
  const running   = [];
  const stopped   = [];

  for (const item of deployed) {
    const id = _driverIdForItem(item, item._cat);
    const s  = _statuses[id] || {};
    const tags = _channelTags(item);
    const latestTag = tags.length > 0 ? tags[0].tag : null;
    const currentTag = s.running_image?.includes(':') ? s.running_image.split(':').pop() : null;
    const hasUpdate = latestTag && currentTag && latestTag !== currentTag;

    const entry = { item, id, s, latestTag, currentTag, hasUpdate };

    if ((s.running || item._cat === 'core') && hasUpdate) {
      updatable.push(entry);
    } else if (s.running || item._cat === 'core') {
      running.push(entry);
    } else {
      stopped.push(entry);
    }
  }

  let html = '';

  if (updatable.length) {
    html += _svcGroupHTML('可更新', updatable.length, 'updatable');
    html += updatable.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }
  if (running.length) {
    html += _svcGroupHTML('运行中', running.length, 'running');
    html += running.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }
  if (stopped.length) {
    html += _svcGroupHTML('已停止', stopped.length, 'stopped');
    html += stopped.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }

  container.innerHTML = html;

  // Bind actions
  container.querySelectorAll('[data-action="upgrade"]').forEach(btn => {
    btn.addEventListener('click', () => _showUpgradeConfirm(btn.dataset));
  });
  container.querySelectorAll('[data-action="stop"]').forEach(btn => {
    btn.addEventListener('click', () => _stopDriver(btn.dataset.driverId, btn));
  });
  container.querySelectorAll('[data-action="start"]').forEach(btn => {
    btn.addEventListener('click', () => _startDriver(btn.dataset.driverId, btn.dataset.image, btn));
  });
  container.querySelectorAll('[data-action="remove"]').forEach(btn => {
    btn.addEventListener('click', () => _removeDriver(btn.dataset.driverId, btn));
  });
  container.querySelectorAll('[data-action="log"]').forEach(btn => {
    btn.addEventListener('click', () => _toggleLog(btn.dataset.driverId));
  });
  // Version switcher dropdowns
  container.querySelectorAll('[data-action="switch-version"]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const wrap = btn.closest('.svc-ver-wrap');
      const dd = wrap.querySelector('.svc-ver-dropdown');

      if (window.innerWidth <= 768) {
        // Mobile: show bottom action sheet
        _showVersionSheet(dd.innerHTML, wrap.dataset.driverId || '');
        return;
      }

      // Desktop: position fixed dropdown
      const wasHidden = dd.classList.contains('hidden');
      document.querySelectorAll('.svc-ver-dropdown').forEach(d => d.classList.add('hidden'));
      if (wasHidden) {
        const rect = btn.getBoundingClientRect();
        dd.classList.remove('hidden');
        const ddHeight = dd.offsetHeight;
        const spaceBelow = window.innerHeight - rect.bottom;
        if (spaceBelow < ddHeight + 10) {
          dd.style.top = (rect.top - ddHeight - 4) + 'px';
        } else {
          dd.style.top = (rect.bottom + 4) + 'px';
        }
        dd.style.right = (window.innerWidth - rect.right) + 'px';
        dd.style.left = '';
      }
    });
  });
  container.querySelectorAll('.svc-ver-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      if (opt.classList.contains('current')) return;
      const { driverId, fullImage, tag, label, channel } = opt.dataset;
      opt.closest('.svc-ver-dropdown').classList.add('hidden');
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag, channel: channel || '' }],
        () => _executeDeploys([[driverId, { image: fullImage }]])
      );
    });
  });
}

function _svcGroupHTML(title, count, cls) {
  return `<div class="svc-group ${cls}">
    <div class="svc-group-header">
      <span class="svc-group-title">${title}</span>
      <span class="svc-group-count">${count}</span>
    </div>`;
}

function _svcRowHTML({ item, id, s, latestTag, currentTag, hasUpdate }) {
  const label = item._cat === 'driver' ? (item.model || item.image) : (item.name || item.image);
  const isRunning = s.running || item._cat === 'core';
  const statusDot = isRunning ? 'running' : s.status === 'error' ? 'error' : 'stopped';
  const imageBase = item.full_repo || item.image;
  const tags = _channelTags(item);

  let actions = '';
  // Version switcher (dropdown)
  if (tags.length > 1) {
    const versionOpts = tags.map(t => {
      const fullImg = t.imageRef || (imageBase + ':' + t.tag);
      const isCurrent = currentTag && t.tag === currentTag;
      const ch = _channelLabel(t.channel);
      return `<div class="svc-ver-opt${isCurrent ? ' current' : ''}" data-driver-id="${id}" data-full-image="${fullImg}" data-tag="${t.tag}" data-label="${label}" data-channel="${t.channel || ''}">
        <span class="svc-ver-tag">${t.tag}</span>
        ${ch ? `<span class="svc-ver-channel">${ch}</span>` : ''}
        ${isCurrent ? '<span class="svc-ver-badge">当前</span>' : ''}
      </div>`;
    }).join('');
    actions += `<div class="svc-ver-wrap">
      <button class="svc-btn svc-btn-ver" data-action="switch-version">切换版本 ▾</button>
      <div class="svc-ver-dropdown hidden">${versionOpts}</div>
    </div>`;
  }
  if (hasUpdate) {
    const latestImage = tags[0]?.imageRef || (imageBase + ':' + latestTag);
    actions += `<button class="svc-btn svc-btn-upgrade" data-action="upgrade" data-driver-id="${id}" data-current-tag="${currentTag}" data-latest-tag="${latestTag}" data-latest-image="${latestImage}" data-label="${label}">升级</button>`;
  }
  if (item._cat === 'core') {
    // Core cannot stop itself — no stop button
  } else if (isRunning) {
    actions += `<button class="svc-btn svc-btn-stop" data-action="stop" data-driver-id="${id}">停止</button>`;
  } else {
    // Stopped: show start + remove
    const lastImage = s.running_image || s.last_deploy?.image || s.image || '';
    if (lastImage) {
      actions += `<button class="svc-btn svc-btn-start" data-action="start" data-driver-id="${id}" data-image="${lastImage}">启动</button>`;
    }
    actions += `<button class="svc-btn svc-btn-remove" data-action="remove" data-driver-id="${id}">卸载</button>`;
  }

  // Log button (always available)
  actions += `<button class="svc-btn svc-btn-log" data-action="log" data-driver-id="${id}">日志</button>`;

  // 孤儿服务没有 catalog 条目，停止时 running_image 也是空的 —— 退回到 manifest 里记录的
  // 镜像，否则用户看到的只有一个「—」，没法知道自己装的到底是哪个版本。
  const versionText = currentTag
    || (s.running_image?.split(':').pop())
    || (item._orphan ? (s.image || s.last_deploy?.image || '').split(':').pop() : '')
    || '—';

  return `
    <div class="svc-row" id="card-${id}">
      <div class="svc-row-dot ${statusDot}" id="dot-${id}"></div>
      <div class="svc-row-info">
        <span class="svc-row-name">${label}</span>
        <div class="svc-row-version-line">
          <span class="svc-row-version">${versionText}</span>
          ${item._orphan ? '<span class="svc-ver-channel" title="该镜像与本机架构不匹配，或已从资源中心下架">架构不匹配</span>' : ''}
          ${hasUpdate ? `<span class="svc-row-arrow">→</span><span class="svc-row-new-version">${latestTag}</span>` : ''}
        </div>
      </div>
      <div class="svc-row-actions">${actions}</div>
    </div>
    <div class="deploy-log hidden" id="log-${id}"></div>`;
}

// ══════════════════════════════════════════════════════════════════════════
//  TAB 2: 驱动市场
// ══════════════════════════════════════════════════════════════════════════

function _renderMarketplace() {
  const q = (document.getElementById('marketplace-search')?.value || '').trim().toLowerCase();

  // Merge perception + actucore + driver for marketplace (core is managed in My Services only)
  const allItems = [
    ...(_catalog.perception || []).map(it => ({ ...it, _cat: 'perception' })),
    ...(_catalog.actucore || []).map(it => ({ ...it, _cat: 'actucore' })),
    ...(_catalog.driver || []).map(it => ({ ...it, _cat: 'driver' })),
  ];

  // Build provider list for filter chips
  const providers = [...new Set(allItems.map(it => it.provider || it.name || 'Other').filter(Boolean))];

  // Render filter chips
  const filtersEl = document.getElementById('marketplace-filters');
  filtersEl.innerHTML = `
    <button class="mp-chip${_activeFilter === null ? ' active' : ''}" data-filter="">全部</button>
    ${providers.map(p => `<button class="mp-chip${_activeFilter === p ? ' active' : ''}" data-filter="${p}">${p}</button>`).join('')}
  `;
  filtersEl.querySelectorAll('.mp-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      _activeFilter = chip.dataset.filter || null;
      _renderMarketplace();
    });
  });

  // Filter items
  let filtered = allItems;
  if (_activeFilter) {
    filtered = filtered.filter(it => (it.provider || it.name || 'Other') === _activeFilter);
  }
  if (q) {
    filtered = filtered.filter(it => {
      const model = (it.model || '').toLowerCase();
      const provider = (it.provider || '').toLowerCase();
      const name = (it.name || '').toLowerCase();
      return model.includes(q) || provider.includes(q) || name.includes(q);
    });
  }

  // Render grid
  const gridEl = document.getElementById('marketplace-grid');
  const hintEl = document.getElementById('marketplace-host-hint');
  if (hintEl) hintEl.innerHTML = _hostHintHTML();
  if (filtered.length === 0) {
    gridEl.innerHTML = `<div class="svc-empty">
      <div class="svc-empty-title">未找到驱动</div>
      <div class="svc-empty-hint">尝试其他搜索词或切换更新通道</div>
    </div>`;
    return;
  }

  gridEl.innerHTML = filtered.map(item => _mpCardHTML(item)).join('');

  // Bind install buttons
  gridEl.querySelectorAll('.mp-install-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleInstallDropdown(btn);
    });
  });

  // Bind card click → detail
  gridEl.querySelectorAll('.mp-card').forEach(card => {
    card.addEventListener('click', (e) => {
      // Don't open detail if clicking install button or version dropdown
      if (e.target.closest('.mp-install-btn') || e.target.closest('.mp-versions') || e.target.closest('.mp-installed-badge')) return;
      _showDriverDetail(card);
    });
  });

  // Bind version options
  gridEl.querySelectorAll('.mp-version-opt').forEach(opt => {
    opt.addEventListener('click', (e) => {
      e.stopPropagation();
      const driverId = opt.dataset.driverId;
      const image = opt.dataset.fullImage;
      const label = opt.dataset.label;
      const tag = opt.dataset.tag;
      const channel = opt.dataset.channel || '';
      opt.closest('.mp-card').querySelector('.mp-versions').classList.add('hidden');
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag, channel }],
        () => _executeDeploys([[driverId, { image }]])
      );
    });
  });
}

function _mpCardHTML(item) {
  const cat = item._cat;
  const label = cat === 'driver' ? (item.model || item.image) : (item.name || item.image);
  const provider = item.provider || '';
  const driverId = _driverIdForItem(item, cat);
  const s = _statuses[driverId];
  const isInstalled = s && (s.running || s.last_deploy);
  const tags = _channelTags(item);
  const imageBase = item.full_repo || item.image;
  const desc = item.description || _CAT_DESC[cat] || '';
  const fullName = item.name || label;

  const versionOpts = tags.map(t => {
    const fullImg = t.imageRef || (imageBase + ':' + t.tag);
    const ch = _channelLabel(t.channel);
    return `<div class="mp-version-opt" data-driver-id="${driverId}" data-full-image="${fullImg}" data-tag="${t.tag}" data-label="${label}" data-channel="${t.channel || ''}">
      <span class="mp-version-tag">${t.tag}</span>
      ${ch ? `<span class="svc-ver-channel">${ch}</span>` : ''}
      ${_archBadgeHTML(t)}
      ${t.created ? `<span class="mp-version-date">${t.created.replace(/\s+\d{2}:\d{2}$/, '')}</span>` : ''}
    </div>`;
  }).join('');

  const installBtn = isInstalled
    ? `<span class="mp-action-btn mp-installed-badge">已安装</span>`
    : tags.length > 0
      ? `<button class="mp-action-btn mp-install-btn">安装 ▾</button>`
      : `<span class="mp-action-btn mp-no-version">暂无版本</span>`;

  return `
    <div class="mp-card" data-driver-id="${driverId}" data-name="${_escAttr(fullName)}" data-desc="${_escAttr(desc)}" data-provider="${_escAttr(provider)}">
      <div class="mp-card-header">
        <span class="mp-card-name">${label}</span>
        ${provider ? `<span class="mp-card-provider">${provider}</span>` : ''}
      </div>
      ${desc ? `<div class="mp-card-desc">${_escHTML(desc)}</div>` : ''}
      <div class="mp-card-action">${installBtn}</div>
      <div class="mp-versions hidden">${versionOpts}</div>
      <div class="mp-card-log hidden" id="mp-log-${driverId}"></div>
    </div>`;
}

function _escAttr(s) { return s.replace(/"/g, '&quot;').replace(/</g, '&lt;'); }
function _escHTML(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function _toggleInstallDropdown(btn) {
  const card = btn.closest('.mp-card');
  const dropdown = card.querySelector('.mp-versions');

  if (window.innerWidth <= 768) {
    // Mobile: use bottom action sheet
    const label = card.querySelector('.mp-card-name')?.textContent || '';
    _showInstallSheet(dropdown.innerHTML, label);
    return;
  }

  // Desktop: position fixed dropdown
  const wasHidden = dropdown.classList.contains('hidden');
  document.querySelectorAll('.mp-versions').forEach(d => d.classList.add('hidden'));
  if (wasHidden) {
    const rect = btn.getBoundingClientRect();
    dropdown.classList.remove('hidden');
    dropdown.style.position = 'fixed';
    dropdown.style.left = '';
    dropdown.style.transform = '';
    const ddHeight = dropdown.offsetHeight;
    const spaceBelow = window.innerHeight - rect.bottom;
    if (spaceBelow < ddHeight + 10) {
      dropdown.style.top = (rect.top - ddHeight - 4) + 'px';
    } else {
      dropdown.style.top = (rect.bottom + 4) + 'px';
    }
    dropdown.style.right = (window.innerWidth - rect.right) + 'px';
  }
}

function _showInstallSheet(optionsHTML, title) {
  document.getElementById('ver-sheet-overlay')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'ver-sheet-overlay';
  overlay.className = 'ver-sheet-overlay';
  overlay.innerHTML = `
    <div class="ver-sheet">
      <div class="ver-sheet-header">
        <span class="ver-sheet-title">${title || '选择版本'}</span>
        <button class="ver-sheet-close">✕</button>
      </div>
      <div class="ver-sheet-body">${optionsHTML}</div>
    </div>
  `;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('active'));

  const close = () => {
    overlay.classList.remove('active');
    setTimeout(() => overlay.remove(), 200);
  };

  overlay.querySelector('.ver-sheet-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  overlay.querySelectorAll('.mp-version-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      const { driverId, fullImage, tag, label, channel } = opt.dataset;
      close();
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag, channel: channel || '' }],
        () => _executeDeploys([[driverId, { image: fullImage }]])
      );
    });
  });
}

// Close marketplace dropdowns when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.mp-card')) {
    document.querySelectorAll('.mp-versions').forEach(d => d.classList.add('hidden'));
  }
  if (!e.target.closest('.svc-ver-wrap')) {
    document.querySelectorAll('.svc-ver-dropdown').forEach(d => d.classList.add('hidden'));
  }
});

// ── Driver detail (in-modal view) ────────────────────────────────────────

function _showDriverDetail(card) {
  const name = card.dataset.name || '';
  const desc = card.dataset.desc || '';
  const provider = card.dataset.provider || '';
  const driverId = card.dataset.driverId || '';
  const s = _statuses[driverId];
  const isInstalled = s && (s.running || s.last_deploy);

  // Get tags from the card's versions dropdown
  const versions = [...card.querySelectorAll('.mp-version-opt')].map(opt => ({
    tag: opt.dataset.tag,
    fullImage: opt.dataset.fullImage,
    label: opt.dataset.label,
    channel: opt.dataset.channel || '',
  }));

  const pane = document.getElementById('pane-marketplace');
  // Save current content for back navigation
  const savedHTML = pane.innerHTML;

  const versionsHTML = versions.map(v => {
    const ch = _channelLabel(v.channel);
    return `
    <div class="detail-ver-row" data-full-image="${v.fullImage}" data-tag="${v.tag}" data-label="${v.label}" data-driver-id="${driverId}" data-channel="${v.channel}">
      <span class="detail-ver-tag">${v.tag}</span>
      ${ch ? `<span class="svc-ver-channel">${ch}</span>` : ''}
      <button class="svc-btn svc-btn-upgrade">部署此版本</button>
    </div>`;
  }).join('');

  pane.innerHTML = `
    <div class="mp-detail">
      <button class="mp-detail-back">← 返回</button>
      <div class="mp-detail-header">
        <div class="mp-detail-title">${name}</div>
        ${provider ? `<div class="mp-detail-provider">${provider}</div>` : ''}
        ${isInstalled ? '<span class="mp-installed-badge" style="margin-top:8px;display:inline-block">已安装</span>' : ''}
      </div>
      ${desc ? `<div class="mp-detail-desc">${desc}</div>` : '<div class="mp-detail-desc" style="color:var(--text-dim)">暂无描述</div>'}
      <div class="deploy-log hidden" id="mp-log-${driverId}"></div>
      <div class="mp-detail-versions">
        <div class="mp-detail-section-title">可用版本</div>
        ${versionsHTML || '<div style="color:var(--text-dim);font-size:12px">暂无版本</div>'}
      </div>
    </div>
  `;

  // Bind back
  pane.querySelector('.mp-detail-back').addEventListener('click', () => {
    pane.innerHTML = savedHTML;
    _renderMarketplace();
  });

  // Bind deploy buttons
  pane.querySelectorAll('.detail-ver-row .svc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const row = btn.closest('.detail-ver-row');
      showDeployConfirmModal(
        [{ label: row.dataset.label, currentTag: '—', newTag: row.dataset.tag, channel: row.dataset.channel || '' }],
        () => _executeDeploys([[row.dataset.driverId, { image: row.dataset.fullImage }]])
      );
    });
  });

  // Always fetch and show deploy logs on detail page
  const logEl = document.getElementById(`mp-log-${driverId}`);
  if (logEl) {
    fetch(`/api/drivers/${driverId}/status`).then(r => r.json()).then(json => {
      const logs = (json.data || {}).logs || '';
      if (logs.trim()) {
        const lines = logs.trim().split('\n');
        logEl.innerHTML = `<div class="deploy-log-title">上次部署日志</div><pre class="log-output">${lines.join('\n')}</pre>`;
        logEl.scrollTop = logEl.scrollHeight;
        logEl.classList.remove('hidden');
      }
    }).catch(() => {});
  }
}

// ══════════════════════════════════════════════════════════════════════════
//  SHARED UTILITIES
// ══════════════════════════════════════════════════════════════════════════

// ── Mobile: bottom action sheet for version selection ─────────────────────

function _showVersionSheet(optionsHTML) {
  // Remove existing sheet if any
  document.getElementById('ver-sheet-overlay')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'ver-sheet-overlay';
  overlay.className = 'ver-sheet-overlay';
  overlay.innerHTML = `
    <div class="ver-sheet">
      <div class="ver-sheet-header">
        <span class="ver-sheet-title">选择版本</span>
        <button class="ver-sheet-close">✕</button>
      </div>
      <div class="ver-sheet-body">${optionsHTML}</div>
    </div>
  `;
  document.body.appendChild(overlay);

  // Animate in
  requestAnimationFrame(() => overlay.classList.add('active'));

  const close = () => {
    overlay.classList.remove('active');
    setTimeout(() => overlay.remove(), 200);
  };

  overlay.querySelector('.ver-sheet-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  // Bind version option clicks
  overlay.querySelectorAll('.svc-ver-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      if (opt.classList.contains('current')) return;
      const { driverId, fullImage, tag, label, channel } = opt.dataset;
      close();
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag, channel: channel || '' }],
        () => _executeDeploys([[driverId, { image: fullImage }]])
      );
    });
  });
}

function _driverIdForItem(item, category) {
  // Orphan items (rebuilt from _statuses, not from the catalog) carry their id directly
  if (item._id) return item._id;
  if (category === 'driver') return `${item.provider}-${item.model}`;
  return item.image;
}

function _channelLabel(channel) {
  if (channel === 'ga') return 'Stable';
  if (channel === 'release') return 'Release';
  if (channel === 'preview') return 'Preview';
  return '';
}

// ── 架构展示 ──────────────────────────────────────────────────────────────

const _ACC_LABEL = {
  'jetson-jp4': 'JetPack 4',
  'jetson-jp5': 'JetPack 5',
  'jetson-jp6': 'JetPack 6',
  'agnostic':   '通用',
  'none':       '无 NVIDIA 加速器',
};

function _accLabel(v) {
  return _ACC_LABEL[v] || v || '未知';
}

/**
 * 单个版本的架构徽标。只在版本确实绑定了某种加速器时显示 —— agnostic / 未知不显示，
 * 免得给每个 driver 都挂一个没信息量的「通用」。
 */
function _archBadgeHTML(t) {
  const acc = t.acc_arch;
  if (!acc || acc === 'agnostic') return '';
  return `<span class="mp-version-arch">${_accLabel(acc)}</span>`;
}

/**
 * 本机架构提示条。
 *
 * 存在的理由：市场里的列表是按本机架构过滤过的，跟官网 motus.phanthy.com 看到的**不一样**。
 * 用户如果不知道这件事，只会以为某个组件消失了。所以把本机架构和被隐藏的数量都摊开讲清楚。
 */
function _hostHintHTML() {
  if (!_facets) return '';

  const arch = `${_accLabel(_facets.acc_arch)} · ${_facets.cpu_arch || '未知'}`;

  if (!_filter.applied) {
    // 老 resource-center 忽略了架构参数 —— 不能谎称已过滤，但要提醒用户自己看清版本后缀
    return `<div class="mp-host-hint mp-host-hint-warn">
      <span class="mp-host-hint-label">本机架构</span>
      <span class="mp-host-hint-arch">${arch}</span>
      <span class="mp-host-hint-note">资源中心版本较旧，未按架构过滤 —— 安装前请自行核对版本后缀</span>
    </div>`;
  }

  const hidden = _filter.hidden_tags > 0
    ? `已为你隐藏 ${_filter.hidden_tags} 个跑不了的版本${_filter.hidden_images > 0 ? `（含 ${_filter.hidden_images} 个组件）` : ''}，因此这里比官网少`
    : '当前所有版本都能在本机运行';

  return `<div class="mp-host-hint">
    <span class="mp-host-hint-label">本机架构</span>
    <span class="mp-host-hint-arch">${arch}</span>
    <span class="mp-host-hint-note">${hidden}</span>
  </div>`;
}

function _updateStatusDots() {
  for (const [id, s] of Object.entries(_statuses)) {
    const dot = document.getElementById(`dot-${id}`);
    if (dot) {
      const isRunning = s.running || s.category === 'core';
      dot.className = 'svc-row-dot ' + (isRunning ? 'running' : s.status === 'error' ? 'error' : 'stopped');
    }
  }
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

  body.innerHTML = items.map(it => {
    const ch = it.channel ? _channelLabel(it.channel) : '';
    return `
    <div class="deploy-confirm-item">
      <div class="deploy-confirm-item-name">${it.label}</div>
      <div class="deploy-confirm-item-versions">
        <span class="deploy-confirm-tag current">${it.currentTag || '—'}</span>
        <span class="deploy-confirm-arrow">→</span>
        <span class="deploy-confirm-tag latest">${it.newTag}</span>
        ${ch ? `<span class="svc-ver-channel">${ch}</span>` : ''}
      </div>
    </div>`;
  }).join('');

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

// ── Upgrade confirm ───────────────────────────────────────────────────────

function _showUpgradeConfirm({ driverId, currentTag, latestTag, latestImage, label }) {
  showDeployConfirmModal(
    [{ label, currentTag, newTag: latestTag }],
    () => _executeDeploys([[driverId, { image: latestImage }]])
  );
}

// ── Stop / Start / Remove ─────────────────────────────────────────────────

async function _stopDriver(driverId, btn) {
  btn.disabled    = true;
  btn.textContent = '停止中…';
  try {
    await fetch(`/api/drivers/${driverId}/stop`, { method: 'POST' });
  } catch (e) {
    console.error('[deploy] stop', e);
  }
  setTimeout(async () => { await _loadStatuses(); _render(); }, 1500);
}

async function _startDriver(driverId, image, btn) {
  if (!image) return;
  btn.disabled    = true;
  btn.textContent = '启动中…';
  _showDeployLog(driverId, '正在启动…');
  try {
    const res = await fetch(`/api/drivers/${driverId}/deploy`, {
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

async function _removeDriver(driverId, btn) {
  if (!confirm('确定卸载此服务？将移除容器并释放空间。')) return;
  btn.disabled = true;
  btn.textContent = '卸载中…';
  try {
    await fetch(`/api/drivers/${driverId}/remove`, { method: 'POST' });
  } catch (e) {
    console.error('[deploy] remove', e);
  }
  setTimeout(async () => { await _loadStatuses(); _render(); }, 1500);
}

// ── Confirm all pending deploys ───────────────────────────────────────────

async function _confirmAll() {
  const entries = Object.entries(_pending);
  if (!entries.length) return;

  const items = entries.map(([id, { image }]) => {
    const s = _statuses[id] || {};
    const currentTag = s.running_image?.includes(':') ? s.running_image.split(':').pop() : '—';
    const newTag = image.split(':').pop();
    return { label: s.name || id, currentTag, newTag };
  });

  showDeployConfirmModal(items, () => _executeDeploys(entries));
}

async function _executeDeploys(entries) {
  const confirmBtn = document.getElementById('deploy-modal-confirm');
  if (confirmBtn) confirmBtn.disabled = true;

  for (const [driverId, { image }] of entries) {
    const isCoreDriver = (_catalog.core || []).some(item => _driverIdForItem(item, 'core') === driverId);

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
      _showDeployLogAny(driverId, '正在请求部署…');
      try {
        const res = await fetch(`/api/drivers/${driverId}/deploy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const json = await res.json();
        if (json.code !== 200) {
          _appendLogAny(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
        } else {
          _appendLogAny(driverId, '镜像拉取中…');
          _startLogPolling(driverId);
        }
      } catch (e) {
        _appendLogAny(driverId, `✗ 网络错误: ${e.message}`, 'error');
      }
    }
  }

  _pending = {};
  _syncFooter();
  if (confirmBtn) confirmBtn.disabled = false;
}

// ── Deploy log (inline) ───────────────────────────────────────────────────

function _ensureLogElement(driverId) {
  if (document.getElementById(`log-${driverId}`)) return;
  // Create a temporary card + log area for new installs
  const container = document.getElementById('pane-my-services');
  if (!container) return;
  const html = `
    <div class="svc-row" id="card-${driverId}">
      <div class="svc-row-dot deploying" id="dot-${driverId}"></div>
      <div class="svc-row-info">
        <span class="svc-row-name">${driverId}</span>
        <div class="svc-row-version-line"><span class="svc-row-version">部署中…</span></div>
      </div>
    </div>
    <div class="deploy-log" id="log-${driverId}"></div>`;
  container.insertAdjacentHTML('afterbegin', html);
}

async function _toggleLog(driverId) {
  const el = document.getElementById(`log-${driverId}`);
  if (!el) return;

  if (!el.classList.contains('hidden')) {
    el.classList.add('hidden');
    return;
  }

  el.innerHTML = `<div class="deploy-log-line" style="color:var(--text-dim)">加载中…</div>`;
  el.classList.remove('hidden');

  try {
    const res = await fetch(`/api/drivers/${driverId}/status`);
    const json = await res.json();
    const data = json.data || {};
    const logs = data.logs || '';

    if (logs.trim()) {
      const lines = logs.trim().split('\n').slice(-20);
      el.innerHTML = `<pre class="log-output">${lines.join('\n')}</pre>`;
    } else {
      el.innerHTML = `<div class="deploy-log-line" style="color:var(--text-dim)">暂无日志</div>`;
    }
  } catch {
    el.innerHTML = `<div class="deploy-log-line error">获取日志失败</div>`;
  }
}

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

// Variants that check both marketplace (mp-log-) and my-services (log-) elements
function _getLogEl(driverId) {
  return document.getElementById(`mp-log-${driverId}`) || document.getElementById(`log-${driverId}`);
}

function _isCardLog(el) {
  return el && el.classList.contains('mp-card-log');
}

function _showDeployLogAny(driverId, msg) {
  const el = _getLogEl(driverId);
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden');
}

function _appendLogAny(driverId, msg, type = '') {
  const el = _getLogEl(driverId);
  if (!el) return;
  if (_isCardLog(el)) {
    // Single-line overwrite on card
    el.textContent = msg;
    if (type === 'error') { el.style.color = 'var(--red)'; }
    else if (type === 'success') { el.style.color = 'var(--green)'; }
    else { el.style.color = ''; }
    el.classList.remove('hidden');
  } else {
    // Multi-line append on detail/my-services
    const line = document.createElement('div');
    line.className = 'deploy-log-line' + (type ? ` ${type}` : '');
    line.textContent = msg;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }
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

      const el = _getLogEl(driverId);
      if (el && logs) {
        if (_isCardLog(el)) {
          // Single-line: show last meaningful line
          const lastLine = logs.trim().split('\n').filter(Boolean).pop() || '';
          el.textContent = lastLine;
          el.style.color = '';
        } else {
          // Multi-line: show full log
          const lines = logs.trim().split('\n').slice(-30);
          el.querySelectorAll('.log-output').forEach(e => e.remove());
          const pre = document.createElement('pre');
          pre.className = 'log-output';
          pre.textContent = lines.join('\n');
          el.appendChild(pre);
          el.scrollTop = el.scrollHeight;
        }
      }

      if (status === 'running') {
        _stopLogPolling(driverId);
        _appendLogAny(driverId, '✓ 运行中', 'success');
        setTimeout(() => {
          const logEl = _getLogEl(driverId);
          if (logEl) logEl.classList.add('hidden');
        }, 5000);
        await _loadStatuses();
        _render();
      } else if (status === 'error' || attempts > 30) {
        _stopLogPolling(driverId);
        _appendLogAny(driverId, `✗ ${status === 'error' ? (data.error || '启动失败') : '部署超时'}`, 'error');
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

// ── Core update polling ───────────────────────────────────────────────────

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
      // 服务重启中，连接断开是正常的
    }
  }, 2000);
}
