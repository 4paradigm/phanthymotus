/**
 * solutions.js — 解决方案：查看当前方案 / 方案市场载入 / 打包保存。
 *
 * 入口在「我的」里（js/account.js），它点的是隐藏的 #btn-solutions。
 *
 * 包体格式与"为什么卡片用 deviceRef 而不是 mcpId"见 src/api/solutions.py 顶部注释。
 * 这里只负责三件事：
 *   1. 展示当前方案，并把打包时脱敏掉的字段引导用户补填
 *   2. 市场载入：先 preflight（缺驱动 → 一键安装；覆盖清单 → 逐项确认），再 apply
 *   3. 打包保存：画布强制勾选，技能只允许选"已激活且已上架"的
 */

import { openInstanceConfigModal, openToolConfigModal } from './sidebar.js';
import { reloadFromServer } from './canvas.js';
import { isRcLoggedIn, rcHeaders, rcFetch, showAccount, refreshAccount } from './account.js';
import { sessionId } from './session.js';

// 与 resource-center/lib/solution.ts 的 INDUSTRIES 保持一致
const INDUSTRIES = [
  ['general',       '泛行业'],
  ['inspection',    '巡检'],
  ['tour-guide',    '导览讲解'],
  ['logistics',     '物流配送'],
  ['education',     '教育'],
  ['healthcare',    '医疗康养'],
  ['security',      '安防'],
  ['manufacturing', '工业制造'],
  ['research',      '科研'],
  ['service',       '商业服务'],
];

const BLOCK_LABELS = {
  'canvas':          '画布',
  'skills':          '技能',
  'prompt.identity': '身份定义',
  'prompt.system':   '系统提示',
  'prompt.memory':   '长期记忆',
  'tasks':           '任务',
};

let _overlay, _closeBtn, _tabs, _panels;
let _marketCache = [];
let _packable = null;
let _loadTarget = null;   // 待载入的方案 {slug, name, ...}
let _alignVersions = false;  // 载入时是否把相关容器对齐到方案记录的 tag

// ── Init ─────────────────────────────────────────────────────────────────────

export function initSolutions() {
  _overlay  = document.getElementById('solution-overlay');
  if (!_overlay) return;
  _closeBtn = document.getElementById('solution-close');
  _tabs     = _overlay.querySelectorAll('.skill-tab');
  _panels   = _overlay.querySelectorAll('.skill-panel');

  document.getElementById('btn-solutions').addEventListener('click', show);
  _closeBtn.addEventListener('click', hide);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) hide(); });

  _tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      _tabs.forEach(t => t.classList.toggle('active', t === tab));
      _panels.forEach(p => p.classList.toggle('active', p.dataset.panel === target));
      if (target === 'current') _loadCurrent();
      if (target === 'market')  _loadMarket();
      if (target === 'save')    _loadSavePanel();
    });
  });

  // 市场搜索 / 行业筛选
  const industrySelect = document.getElementById('solution-industry-filter');
  industrySelect.innerHTML = `<option value="all">全部行业</option>` +
    INDUSTRIES.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
  industrySelect.addEventListener('change', _loadMarket);
  let searchTimer;
  document.getElementById('solution-search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(_loadMarket, 300);
  });

  // 载入确认弹窗
  document.getElementById('solution-load-close').addEventListener('click', _closeLoadModal);
  document.getElementById('solution-load-cancel').addEventListener('click', _closeLoadModal);
  document.getElementById('solution-load-confirm').addEventListener('click', _confirmLoad);

  _syncEntryBadge();
}

export function show() {
  _overlay.classList.remove('hidden');
  _tabs[0].click();
}

export function hide() {
  _overlay.classList.add('hidden');
}

/**
 * 当前方案的展示位在「我的」里（解决方案条目的副文本），载入 / 清除后刷新它。
 * 顶栏原来那个角标随入口按钮一起收进了「我的」。
 */
function _syncEntryBadge() {
  refreshAccount();
}

// ── Tab 1：当前方案 ─────────────────────────────────────────────────────────

async function _loadCurrent() {
  const panel = document.getElementById('solution-current-panel');
  panel.innerHTML = `<div class="skill-empty">加载中…</div>`;

  let current = null;
  try {
    current = (await (await fetch('/api/solutions/current')).json()).data;
  } catch { /* 下面按"无方案"渲染 */ }

  if (!current || !current.slug && !current.name) {
    panel.innerHTML = `
      <div class="skill-empty">
        当前未载入任何解决方案。<br>
        可以到「方案市场」载入一套，或在「保存当前方案」里把现在的配置打包发布。
      </div>`;
    return;
  }

  const industry = (INDUSTRIES.find(i => i[0] === current.industry) || [null, current.industry])[1];
  const devices = current.devices || [];
  const needs = current.needsConfig || [];

  panel.innerHTML = `
    <div class="solution-current-card">
      <div class="solution-current-head">
        <div>
          <div class="solution-current-name">${_esc(current.name || current.slug)}</div>
          <div class="solution-current-meta">
            ${current.slug ? _esc(current.slug) + ' · ' : ''}v${_esc(current.version || '')}
            ${industry ? ' · ' + _esc(industry) : ''}
            ${current.appliedAt ? ' · 载入于 ' + new Date(current.appliedAt * 1000).toLocaleString() : ''}
            ${current.versionAligned ? ' · 已对齐容器版本' : ''}
          </div>
        </div>
        <button class="skill-btn skill-btn-sm" id="solution-clear-current" title="只清除标记，不改动已载入的配置">清除标记</button>
      </div>

      <div class="solution-section-label">包含内容</div>
      <div class="solution-pills">
        ${(current.includes || []).map(b => `<span class="solution-pill">${_esc(BLOCK_LABELS[b] || b)}</span>`).join('')}
      </div>

      ${devices.length ? `
        <div class="solution-section-label">所需驱动（记录版本）</div>
        <div class="solution-driver-list">
          ${devices.map(d => `
            <div class="solution-driver-row">
              <span class="solution-driver-name">${_esc(d.name || d.serverName)}</span>
              <span class="solution-pill solution-pill-sm">${_esc(d.category || '')}</span>
              <span class="solution-driver-image">${_esc(d.image || d.registryImage || '')}</span>
            </div>`).join('')}
        </div>` : ''}

      ${needs.length ? `
        <div class="solution-section-label solution-warn">待补填字段（打包时按卡片声明脱敏）</div>
        <div class="solution-needs-list" id="solution-needs-list">
          ${needs.map(p => `<button class="solution-need-item" data-path="${_esc(p)}">${_esc(p)}</button>`).join('')}
        </div>` : ''}
    </div>`;

  panel.querySelector('#solution-clear-current').addEventListener('click', async () => {
    if (!confirm('清除"当前方案"标记？已载入的画布 / 技能 / Prompt / 任务不会被改动。')) return;
    await fetch('/api/solutions/current', { method: 'DELETE' });
    _syncEntryBadge();
    _loadCurrent();
  });

  panel.querySelectorAll('.solution-need-item').forEach(btn => {
    btn.addEventListener('click', () => _openConfigForPath(btn.dataset.path, devices));
  });
}

/**
 * 把 needsConfig 里的 "d0:asr_start:card-x:api_key" 落到具体的配置弹窗。
 * ref → 本机 mcpId 靠 devices[].serverName 与 /api/mcp 的 server_name 对上。
 */
async function _openConfigForPath(path, devices) {
  const parts = path.split(':');
  if (parts.length < 3) return;
  const ref = parts[0];
  const toolName = parts[1];
  // 中间可能有 instance id：d0:tool:prop（3 段）或 d0:tool:cardId:prop（4 段）
  const instanceId = parts.length >= 4 ? parts[2] : '';

  const dev = devices.find(d => d.ref === ref);
  if (!dev) { alert('无法定位该字段所属设备，请手动到卡片上配置'); return; }

  let mcps = [];
  try { mcps = (await (await fetch('/api/mcp')).json()).data || []; } catch { /* below */ }
  const mcp = mcps.find(m => m.server_name && m.server_name === dev.serverName)
           || mcps.find(m => dev.port && (m.url || '').includes(`:${dev.port}`));
  if (!mcp) { alert(`设备 ${dev.name || dev.serverName} 当前不在线，无法打开配置`); return; }

  const tool = (mcp.tools || []).find(t => typeof t === 'object' && t.name === toolName);
  const schema = tool ? tool.configSchema : null;
  if (!schema) { alert(`${toolName} 当前没有可编辑的配置项`); return; }

  hide();
  if (instanceId) openInstanceConfigModal(mcp.id, toolName, instanceId, schema);
  else            openToolConfigModal(mcp.id, toolName, schema);
}

// ── Tab 2：方案市场 ─────────────────────────────────────────────────────────

async function _loadMarket() {
  const list = document.getElementById('solution-market-list');
  const empty = document.getElementById('solution-market-empty');
  const search = document.getElementById('solution-search').value.trim();
  const industry = document.getElementById('solution-industry-filter').value;

  list.innerHTML = `<div class="skill-empty">加载中…</div>`;
  empty.classList.add('hidden');

  const params = new URLSearchParams({ limit: '30' });
  if (search) params.set('search', search);
  if (industry) params.set('industry', industry);

  let items = [];
  try {
    const json = await (await fetch(`/api/solutions/market?${params}`)).json();
    if (json.code !== 200) {
      list.innerHTML = `<div class="skill-empty">无法连接方案市场：${_esc(json.error || '')}</div>`;
      return;
    }
    items = json.data || [];
  } catch (e) {
    list.innerHTML = `<div class="skill-empty">无法连接方案市场：${_esc(e.message)}</div>`;
    return;
  }

  _marketCache = items;
  empty.classList.toggle('hidden', items.length > 0);
  list.innerHTML = items.map(s => {
    const industryLabel = (INDUSTRIES.find(i => i[0] === s.industry) || [null, s.industry])[1];
    const drivers = s.requiredDrivers || [];
    return `
      <div class="skill-card solution-market-card">
        <div class="skill-card-header">
          <span class="skill-card-icon">${s.icon || '◈'}</span>
          <div class="skill-card-info">
            <span class="skill-card-name">${_esc(s.name)}</span>
            <span class="skill-card-meta">${_esc(industryLabel || '')} · v${_esc(s.version)} · ${_esc(s.author?.name || '匿名')} · ↓${s.downloads || 0}</span>
          </div>
          <button class="skill-btn skill-btn-sm skill-btn-primary" data-load="${_esc(s.slug)}">载入</button>
        </div>
        <p class="skill-card-desc">${_esc(s.oneLiner)}</p>
        <div class="solution-pills">
          ${(s.includes || []).map(b => `<span class="solution-pill solution-pill-sm">${_esc(BLOCK_LABELS[b] || b)}</span>`).join('')}
          ${drivers.map(d => `<span class="solution-pill solution-pill-driver">${_esc(d.name || d.serverName)}</span>`).join('')}
        </div>
      </div>`;
  }).join('');

  list.querySelectorAll('[data-load]').forEach(btn => {
    btn.addEventListener('click', () => _startLoad(btn.dataset.load));
  });
}

// ── 载入流程：preflight → 缺驱动一键安装 → 覆盖确认 → apply ────────────────

// 与画布共用同一个 per-tab session id（见 session.js），否则 preflight 会把自己
// 持有的编辑锁误判成"别人正在编辑"
function _sessionId() { return sessionId(); }

async function _preflight(slug) {
  const res = await fetch('/api/solutions/preflight', {
    method: 'POST', headers: rcHeaders(),
    body: JSON.stringify({ slug, session_id: _sessionId() }),
  });
  return res.json();
}

async function _startLoad(slug) {
  _alignVersions = false;   // 每次重新进入载入流程都从"不对齐"开始
  const body = document.getElementById('solution-load-body');
  document.getElementById('solution-load-overlay').classList.remove('hidden');
  body.innerHTML = `<div class="skill-empty">检查中…</div>`;
  document.getElementById('solution-load-confirm').classList.add('hidden');

  const json = await _preflight(slug);
  if (json.code !== 200) {
    body.innerHTML = `<div class="solution-load-error">${_esc(json.error || '检查失败')}</div>`;
    return;
  }
  _loadTarget = { slug, ...json.data };
  _renderLoadModal(json.data);
}

function _renderLoadModal(data) {
  const body = document.getElementById('solution-load-body');
  const confirmBtn = document.getElementById('solution-load-confirm');
  const sol = data.solution || {};
  const dev = data.devices || {};
  const ow = data.overwrite || {};

  document.getElementById('solution-load-title').textContent = `载入「${sol.name || sol.slug}」`;

  const missing = dev.missing || [];
  const installable = dev.installable || [];
  const matched = dev.matched || [];

  let html = '';

  if (data.canvasEditor) {
    html += `<div class="solution-load-error">
      有其他会话正在编辑画布（${_esc(data.canvasEditor)}）。请让对方退出编辑后再载入，
      否则它的自动保存会把刚载入的画布覆盖回去。
    </div>`;
  }

  html += `<div class="solution-section-label">所需驱动</div><div class="solution-driver-list">`;
  html += matched.map(d => `
    <div class="solution-driver-row">
      <span class="solution-driver-ok">✓</span>
      <div class="solution-driver-main">
        <div class="solution-driver-line">
          <span class="solution-driver-name">${_esc(d.name || d.serverName)}</span>
          <span class="solution-pill solution-pill-sm">${_esc(d.category || '')}</span>
          <span class="solution-driver-state">已就绪</span>
        </div>
        ${_versionCell(d)}
      </div>
    </div>`).join('');
  html += installable.map(d => `
    <div class="solution-driver-row solution-driver-row-warn">
      <span class="solution-driver-warn">!</span>
      <div class="solution-driver-main">
        <div class="solution-driver-line">
          <span class="solution-driver-name">${_esc(d.name || d.serverName)}</span>
          <span class="solution-pill solution-pill-sm">${_esc(d.category || '')}</span>
          <span class="solution-driver-state">未启动</span>
        </div>
        ${_versionCell(d)}
        <div class="solution-driver-sub">${_esc(d.localImage || d.image || '')}</div>
      </div>
      <button class="skill-btn skill-btn-sm skill-btn-primary" data-install="${_esc(d.localDriverId)}">一键安装</button>
    </div>`).join('');
  html += missing.map(d => `
    <div class="solution-driver-row solution-driver-row-error">
      <span class="solution-driver-err">✕</span>
      <div class="solution-driver-main">
        <div class="solution-driver-line">
          <span class="solution-driver-name">${_esc(d.name || d.serverName)}</span>
          <span class="solution-pill solution-pill-sm">${_esc(d.category || '')}</span>
          <span class="solution-driver-state">本机缺少</span>
        </div>
        <div class="solution-driver-sub">${_esc(d.registryImage || d.serverName)}</div>
      </div>
    </div>`).join('');
  html += `</div>`;

  // 版本对齐：默认不对齐（打包只记录版本，不做判断），勾上则把相关容器
  // 重新部署到方案记录的 tag。Agent Core 自身不在此列 —— 它就是正在处理这次
  // 请求的容器，重启会让载入半路断掉，只能提示用户手动升级。
  const misaligned = data.misaligned || [];
  const alignable = [...matched, ...installable].filter(d => d.alignImage);
  html += `<div class="solution-section-label">容器版本</div>`;
  html += `<label class="solution-check" id="solution-align-wrap">
      <input type="checkbox" id="solution-align-versions" ${alignable.length ? '' : 'disabled'}>
      <span>对齐方案记录的容器版本${alignable.length ? '' : '（方案未记录可用的镜像 tag）'}</span>
    </label>`;
  if (misaligned.length) {
    html += `<div class="solution-hint solution-warn">
      勾选后会先把这些容器重新部署到方案记录的 tag，再载入：
      ${misaligned.map(d => `<code>${_esc(d.name || d.serverName)} ${_esc(d.runningTag || '未知')} → ${_esc(d.packageTag)}</code>`).join('、')}
    </div>`;
  } else if (alignable.length) {
    html += `<div class="solution-hint">当前各容器版本已与方案记录一致（或无法读取 Docker 状态）。</div>`;
  }
  if (sol.coreVersion && data.selfVersion && sol.coreVersion !== data.selfVersion) {
    html += `<div class="solution-hint solution-warn">
      方案打包自 Agent Core <code>${_esc(sol.coreVersion)}</code>，本机是
      <code>${_esc(data.selfVersion)}</code>。Agent Core 自身不会自动对齐
      （重启会中断本次载入），需要时请到「设置 → 部署」手动升级。
    </div>`;
  }

  if (missing.length) {
    html += `<div class="solution-load-error">
      本机缺少上述驱动，无法载入。请先到「设置 → 部署」同步镜像仓库并安装
      ${missing.map(d => `<code>${_esc(d.registryImage || d.serverName)}</code>`).join('、')}
      ，安装完成后再回到这里载入。
      <button class="skill-btn skill-btn-sm" id="solution-sync-registry">同步镜像仓库</button>
    </div>`;
  }

  // 覆盖清单
  html += `<div class="solution-section-label solution-warn">载入会覆盖以下内容</div>
           <ul class="solution-overwrite-list">`;
  if (ow.canvas) {
    html += `<li>当前画布：${ow.canvas.cards} 张卡片、${ow.canvas.connections} 条连线、${ow.canvas.toolConfigs} 份卡片配置将被整体替换</li>`;
  }
  if (ow.skills) {
    html += ow.skills.length
      ? `<li>已激活技能将改为方案指定的一套（当前激活：${ow.skills.map(s => _esc(s.name || s.slug)).join('、')}；不会卸载，只是停用）</li>`
      : `<li>技能：当前没有激活的技能，方案里的技能会被安装并激活</li>`;
  }
  (ow.prompt || []).forEach(p => {
    html += `<li>${_esc(BLOCK_LABELS[p.block] || p.block)}：${_esc(p.path)} ${p.exists ? '将被覆盖' : '将被创建'}</li>`;
  });
  if (ow.tasks) {
    html += ow.tasks.length
      ? `<li>任务：现有 ${ow.tasks.length} 个任务将被清除（${ow.tasks.map(t => _esc(t.goal)).join('；')}），改为方案中的任务</li>`
      : `<li>任务：当前无活跃任务，方案中的任务会被创建</li>`;
  }
  html += `</ul>`;

  if ((sol.needsConfig || []).length) {
    html += `<div class="solution-section-label solution-warn">载入后需要补填</div>
      <div class="solution-needs-list">
        ${sol.needsConfig.map(p => `<span class="solution-need-item solution-need-item-static">${_esc(p)}</span>`).join('')}
      </div>`;
  }

  body.innerHTML = html;

  body.querySelectorAll('[data-install]').forEach(btn => {
    btn.addEventListener('click', () => _installDriver(btn.dataset.install, btn));
  });
  const syncBtn = body.querySelector('#solution-sync-registry');
  if (syncBtn) {
    syncBtn.addEventListener('click', async () => {
      syncBtn.disabled = true;
      syncBtn.textContent = '同步中…';
      try { await fetch('/api/drivers/sync', { method: 'POST' }); } catch { /* 下面重查 */ }
      _startLoad(_loadTarget.slug);
    });
  }

  // 勾选状态要跨重渲染保留：装完一个驱动就会重新 preflight + 重画，
  // 用户不该因此丢掉刚勾上的"对齐版本"。
  const alignBox = body.querySelector('#solution-align-versions');
  if (alignBox) {
    alignBox.checked = _alignVersions && !alignBox.disabled;
    alignBox.addEventListener('change', () => { _alignVersions = alignBox.checked; });
  }

  const ready = !missing.length && !installable.length && !data.canvasEditor;
  confirmBtn.classList.toggle('hidden', !ready);
  confirmBtn.disabled = !ready;
}

/** 版本格挡：本机在跑的 tag vs 方案记录的 tag。 */
function _versionCell(d) {
  if (!d.packageTag) return '';
  if (d.aligned === false) {
    return `<span class="solution-ver solution-ver-diff" title="本机 ${_esc(d.runningTag || '未知')} → 方案 ${_esc(d.packageTag)}">${_esc(d.runningTag || '未知')} → ${_esc(d.packageTag)}</span>`;
  }
  if (d.aligned === true) {
    return `<span class="solution-ver">${_esc(d.packageTag)}</span>`;
  }
  return `<span class="solution-ver solution-ver-unknown" title="读不到容器状态">方案 ${_esc(d.packageTag)}</span>`;
}

/** 部署一个已在驱动清单里、但还没跑起来的驱动，然后等它自注册成 MCP。 */
async function _installDriver(driverId, btn) {
  btn.disabled = true;
  btn.textContent = '安装中…';
  try {
    const res = await fetch(`/api/drivers/${encodeURIComponent(driverId)}/deploy`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    const json = await res.json();
    if (json.code !== 200) {
      btn.disabled = false;
      btn.textContent = '重试安装';
      alert(`部署失败：${json.message || '未知错误'}`);
      return;
    }
  } catch (e) {
    btn.disabled = false;
    btn.textContent = '重试安装';
    alert(`部署失败：${e.message}`);
    return;
  }

  // 容器起来后驱动会自己 POST /api/mcp 注册（见 phanthymotus-driver/common
  // 的 vendor_runtime.py），所以这里轮询 preflight 直到该设备变成 matched。
  btn.textContent = '等待驱动上线…';
  for (let i = 0; i < 40; i++) {           // 最多约 2 分钟
    await new Promise(r => setTimeout(r, 3000));
    const json = await _preflight(_loadTarget.slug);
    if (json.code !== 200) continue;
    const stillPending = (json.data.devices.installable || [])
      .some(d => d.localDriverId === driverId);
    if (!stillPending) {
      _loadTarget = { slug: _loadTarget.slug, ...json.data };
      _renderLoadModal(json.data);
      return;
    }
  }
  btn.disabled = false;
  btn.textContent = '重试安装';
  alert('驱动已部署但仍未上线，请到「设置 → 部署」查看容器日志。');
}

async function _confirmLoad() {
  const confirmBtn = document.getElementById('solution-load-confirm');
  const body = document.getElementById('solution-load-body');
  confirmBtn.disabled = true;

  try {
    if (_alignVersions) {
      confirmBtn.textContent = '对齐版本中…';
      const ok = await _alignAllVersions(confirmBtn);
      if (!ok) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认载入';
        return;
      }
    }

    confirmBtn.textContent = '载入中…';
    const res = await fetch('/api/solutions/apply', {
      method: 'POST', headers: rcHeaders(),
      body: JSON.stringify({
        slug: _loadTarget.slug, confirm: true, session_id: _sessionId(),
        align_versions: _alignVersions,
      }),
    });
    const json = await res.json();
    if (json.code !== 200) {
      body.innerHTML = `<div class="solution-load-error">${_esc(json.error || '载入失败')}</div>` + body.innerHTML;
      confirmBtn.disabled = false;
      confirmBtn.textContent = '确认载入';
      return;
    }
    const needs = json.data.needsConfig || [];
    const failed = json.data.applied?.skills?.failed || [];
    _closeLoadModal();
    await reloadFromServer();
    _syncEntryBadge();
    _tabs[0].click();
    let msg = '解决方案已载入。';
    if (_alignVersions) msg += '\n相关容器已对齐到方案记录的版本。';
    if (needs.length) msg += `\n有 ${needs.length} 个脱敏字段需要补填，见「当前方案」。`;
    if (failed.length) msg += `\n以下技能安装失败：${failed.map(f => f.slug).join('、')}`;
    if (json.data.warning) msg += `\n${json.data.warning}`;
    alert(msg);
  } catch (e) {
    confirmBtn.disabled = false;
    confirmBtn.textContent = '确认载入';
    alert(`载入失败：${e.message}`);
  }
}

/**
 * 逐个把版本不一致的容器重新部署到方案记录的 tag。
 *
 * 一个个来而不是并发：拉镜像很吃带宽和磁盘，几个驱动同时 pull 在机器人这种
 * 硬件上容易把自己拖死。每部署完一个都重新 preflight，一是拿到最新的对齐
 * 状态，二是确认容器真的重新注册上来了。
 */
async function _alignAllVersions(statusBtn) {
  let pending = (_loadTarget.misaligned || []).map(d => d.ref);
  const total = pending.length;
  if (!total) return true;

  for (let done = 0; done < total; done++) {
    const ref = pending[0];
    const dev = (_loadTarget.misaligned || []).find(d => d.ref === ref) || {};
    const label = dev.name || dev.serverName || ref;
    statusBtn.textContent = `对齐 ${label}（${done + 1}/${total}）…`;

    let json;
    try {
      const res = await fetch('/api/solutions/align-device', {
        method: 'POST', headers: rcHeaders(),
        body: JSON.stringify({ slug: _loadTarget.slug, ref }),
      });
      json = await res.json();
    } catch (e) {
      alert(`对齐 ${label} 失败：${e.message}`);
      return false;
    }
    if (json.code !== 200) {
      alert(`对齐 ${label} 失败：${json.error || '未知错误'}`);
      return false;
    }

    // 等容器重新起来并注册；轮询 preflight 直到这个 ref 不再出现在 misaligned
    let settled = false;
    for (let i = 0; i < 40; i++) {          // 最多约 2 分钟
      await new Promise(r => setTimeout(r, 3000));
      const pf = await _preflight(_loadTarget.slug);
      if (pf.code !== 200) continue;
      _loadTarget = { slug: _loadTarget.slug, ...pf.data };
      const still = (pf.data.misaligned || []).some(d => d.ref === ref);
      const notReady = (pf.data.devices.installable || []).some(d => d.ref === ref);
      if (!still && !notReady) { settled = true; break; }
    }
    if (!settled) {
      _renderLoadModal(_loadTarget);
      alert(`${label} 已按方案版本重新部署，但还没恢复上线。请到「设置 → 部署」看容器日志。`);
      return false;
    }
    pending = (_loadTarget.misaligned || []).map(d => d.ref);
    if (!pending.length) break;
  }
  return true;
}

function _closeLoadModal() {
  document.getElementById('solution-load-overlay').classList.add('hidden');
}

// ── Tab 3：保存当前方案 ─────────────────────────────────────────────────────

async function _loadSavePanel() {
  const panel = document.getElementById('solution-save-panel');
  panel.innerHTML = `<div class="skill-empty">加载中…</div>`;

  if (!isRcLoggedIn()) {
    // 与技能 modal 的「我的技能」未登录态用同一套 .skill-rc-login 结构，
    // 两处提示看起来才是一件事
    panel.innerHTML = `
      <div class="skill-rc-login">
        <p class="skill-rc-login-hint">发布解决方案需要先登录 Resource Center</p>
        <button class="skill-btn skill-btn-primary" id="sol-goto-login">去「我的」登录</button>
      </div>`;
    panel.querySelector('#sol-goto-login').addEventListener('click', () => showAccount());
    return;
  }

  let data;
  try {
    const json = await (await fetch('/api/solutions/packable', { headers: rcHeaders() })).json();
    if (json.code !== 200) {
      panel.innerHTML = `<div class="skill-empty">${_esc(json.error || '读取失败')}</div>`;
      return;
    }
    data = json.data;
  } catch (e) {
    panel.innerHTML = `<div class="skill-empty">读取失败：${_esc(e.message)}</div>`;
    return;
  }
  _packable = data;

  if (!data.canvas.cards) {
    panel.innerHTML = `<div class="skill-empty">画布为空。解决方案必须包含画布，请先在画布上搭好拓扑。</div>`;
    return;
  }

  const sensitive = (data.configFields || []).filter(f => f.sensitive);
  const localOnly = (data.configFields || []).filter(f => f.localOnly && !f.sensitive);
  const others    = (data.configFields || []).filter(f => !f.sensitive && !f.localOnly);

  // 适用机型由后端从画布上的硬件驱动推出（没有硬件驱动就是空）—— 作者不能手填，
  // 否则方案会声称支持一台打包这边根本没有驱动的机器
  const robots = data.robotTypes || [];

  panel.innerHTML = `
    <div class="solution-save-form">
      <div class="solution-section-label">打包内容</div>

      <div class="solution-check-group">
        <div class="solution-check-group-title">画布（必选）</div>
        <label class="solution-check solution-check-locked">
          <input type="checkbox" checked disabled>
          <span>${data.canvas.cards} 张卡片、${data.canvas.devices.length} 个设备 —— 解决方案必须包含画布</span>
        </label>
        ${data.canvas.unresolved.length ? `
          <div class="solution-load-error">
            画布上有卡片引用了未注册的设备（${data.canvas.unresolved.map(_esc).join('、')}），
            请先删除这些卡片再打包。
          </div>` : ''}
      </div>

      <div class="solution-check-group">
        <div class="solution-check-group-title">技能（仅当前激活）</div>
        ${data.skills.available.length ? data.skills.available.map(s => `
          <label class="solution-check">
            <input type="checkbox" class="sol-skill" value="${_esc(s.slug)}" checked>
            <span>${s.icon || '◆'} ${_esc(s.name || s.slug)} <em>v${_esc(s.version || '')}</em></span>
          </label>`).join('') : `<div class="solution-hint">当前没有"已激活且已上架"的技能</div>`}
        ${data.skills.offMarket.length ? `
          <div class="solution-hint solution-warn">
            以下技能只能在本机使用，无法打包 —— 只能保存技能广场中已上架的技能，
            请先在「技能 → 我的技能」发布并通过审核：
            ${data.skills.offMarket.map(s => `<code>${_esc(s.name || s.slug)}</code>`).join('、')}
          </div>` : ''}
      </div>

      <div class="solution-check-group">
        <div class="solution-check-group-title">Prompt 设定</div>
        ${data.prompt.map(p => `
          <label class="solution-check ${p.exists ? '' : 'solution-check-disabled'}">
            <input type="checkbox" class="sol-prompt" value="${_esc(p.block.split('.')[1])}" ${p.exists ? '' : 'disabled'}>
            <span>${_esc(BLOCK_LABELS[p.block] || p.block)} <em>${p.exists ? p.chars + ' 字符' : '文件不存在'}</em></span>
          </label>`).join('')}
      </div>

      <div class="solution-check-group">
        <div class="solution-check-group-title">任务</div>
        <label class="solution-check ${data.tasks.length ? '' : 'solution-check-disabled'}">
          <input type="checkbox" id="sol-tasks" ${data.tasks.length ? '' : 'disabled'}>
          <span>打包 ${data.tasks.length} 个活跃任务${data.tasks.length ? '：' + data.tasks.map(t => _esc(t.goal)).join('；') : '（当前无任务）'}</span>
        </label>
      </div>

      <div class="solution-check-group">
        <div class="solution-check-group-title">打包时会清空的字段</div>
        ${sensitive.length ? `
          <div class="solution-hint">以下字段由卡片声明为敏感，一定会被清空：</div>
          ${sensitive.map(f => `
            <label class="solution-check solution-check-locked">
              <input type="checkbox" checked disabled>
              <span><code>${_esc(f.path)}</code></span>
            </label>`).join('')}` : `<div class="solution-hint">卡片没有声明任何敏感字段</div>`}
        ${localOnly.length ? `
          <div class="solution-hint">以下字段只对本机有效（渠道 / 声卡设备），载入方需要重选：</div>
          ${localOnly.map(f => `
            <label class="solution-check solution-check-locked">
              <input type="checkbox" checked disabled>
              <span><code>${_esc(f.path)}</code></span>
            </label>`).join('')}` : ''}
        ${others.length ? `
          <div class="solution-hint">如果下面还有不该外传的值，勾选它一并清空：</div>
          ${others.map(f => `
            <label class="solution-check">
              <input type="checkbox" class="sol-redact" value="${_esc(f.path)}">
              <span><code>${_esc(f.path)}</code></span>
            </label>`).join('')}` : ''}
      </div>

      <div class="solution-section-label">方案信息</div>
      <div class="skill-form-row">
        <div class="skill-form-field">
          <label>名称 <span class="required">*</span></label>
          <input type="text" id="sol-name" required>
        </div>
        <div class="skill-form-field">
          <label>Slug <span class="required">*</span></label>
          <input type="text" id="sol-slug" required pattern="[a-z0-9-]+" placeholder="tour-guide-g1">
        </div>
      </div>
      <div class="skill-form-row">
        <div class="skill-form-field">
          <label>行业 <span class="required">*</span></label>
          <select id="sol-industry">${INDUSTRIES.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select>
        </div>
        <div class="skill-form-field skill-form-field-sm">
          <label>图标</label>
          <input type="text" id="sol-icon" placeholder="◈">
        </div>
        <div class="skill-form-field">
          <label>版本 <span class="required">*</span></label>
          <input type="text" id="sol-version" required value="1.0.0">
        </div>
      </div>
      <div class="skill-form-field">
        <label>一句话简介 <span class="required">*</span></label>
        <input type="text" id="sol-oneliner" required maxlength="80">
      </div>
      <div class="skill-form-field">
        <label>详细描述 <span class="required">*</span></label>
        <textarea id="sol-description" rows="3" required></textarea>
      </div>
      <div class="skill-form-row">
        <div class="skill-form-field">
          <label>适用机型</label>
          <div class="solution-robots">
            ${robots.length
              ? robots.map(m => `<span class="solution-pill solution-pill-driver">${_esc(m)}</span>`).join('')
              : `<span class="solution-robots-none">None</span>`}
          </div>
          <div class="solution-hint">${robots.length
            ? '由画布上的硬件驱动自动确定'
            : '画布上没有硬件驱动，方案不声明适用机型'}</div>
        </div>
        <div class="skill-form-field">
          <label>标签（逗号分隔）</label>
          <input type="text" id="sol-tags" placeholder="导览, 语音交互">
        </div>
      </div>

      <p class="skill-form-error hidden" id="sol-error"></p>
      <div class="skill-form-footer">
        <span class="solution-hint">发布后为草稿，需在 Resource Center 提交审核</span>
        <button type="button" class="skill-btn skill-btn-primary" id="sol-publish">发布到方案市场</button>
      </div>
    </div>`;

  panel.querySelector('#sol-publish').addEventListener('click', _publish);
}

function _collectInclude() {
  const panel = document.getElementById('solution-save-panel');
  return {
    canvas: true,
    skills: [...panel.querySelectorAll('.sol-skill:checked')].map(el => el.value),
    prompt: [...panel.querySelectorAll('.sol-prompt:checked')].map(el => el.value),
    tasks:  !!panel.querySelector('#sol-tasks')?.checked,
  };
}

async function _publish() {
  const panel = document.getElementById('solution-save-panel');
  const errorEl = panel.querySelector('#sol-error');
  const btn = panel.querySelector('#sol-publish');
  errorEl.classList.add('hidden');

  const meta = {
    name:        panel.querySelector('#sol-name').value.trim(),
    slug:        panel.querySelector('#sol-slug').value.trim(),
    industry:    panel.querySelector('#sol-industry').value,
    icon:        panel.querySelector('#sol-icon').value.trim() || null,
    version:     panel.querySelector('#sol-version').value.trim(),
    oneLiner:    panel.querySelector('#sol-oneliner').value.trim(),
    description: panel.querySelector('#sol-description').value.trim(),
    // 机型不来自表单，来自 /packable 推导出的画布硬件驱动
    robotTypes:  _packable?.robotTypes || [],
    tags:        _splitList(panel.querySelector('#sol-tags').value),
  };
  if (!meta.name || !meta.slug || !meta.oneLiner || !meta.description) {
    errorEl.textContent = '名称 / Slug / 一句话简介 / 详细描述都是必填';
    errorEl.classList.remove('hidden');
    return;
  }
  if (!/^[a-z0-9-]+$/.test(meta.slug)) {
    errorEl.textContent = 'Slug 只能包含小写字母、数字和连字符';
    errorEl.classList.remove('hidden');
    return;
  }

  btn.disabled = true;
  btn.textContent = '发布中…';
  try {
    const json = await rcFetch('/api/solutions/publish', {
      method: 'POST',
      body: JSON.stringify({
        meta,
        include: _collectInclude(),
        extra_redact: [...panel.querySelectorAll('.sol-redact:checked')].map(el => el.value),
      }),
    });
    if (json.code !== 200) {
      const detail = json.detail ? `（${[].concat(json.detail).join('、')}）` : '';
      errorEl.textContent = `${json.error || '发布失败'}${detail}`;
      errorEl.classList.remove('hidden');
      return;
    }
    alert('已作为草稿发布到方案市场。请到 Resource Center 的「我的方案」提交审核。');
  } catch (e) {
    errorEl.textContent = `发布失败：${e.message}`;
    errorEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.textContent = '发布到方案市场';
  }
}

// ── Util ────────────────────────────────────────────────────────────────────

function _splitList(raw) {
  return raw ? raw.split(',').map(s => s.trim()).filter(Boolean) : [];
}

function _esc(str) {
  if (str === 0) return '0';
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
