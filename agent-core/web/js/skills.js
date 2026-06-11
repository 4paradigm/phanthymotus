/**
 * skills.js — 技能管理 modal（安装/卸载/浏览/详情/编辑/发布）
 */

let _overlay, _closeBtn, _tabs, _panels;
let _installedList, _installedEmpty, _browseList, _browseEmpty, _searchInput;

export function initSkills() {
  _overlay        = document.getElementById('skill-overlay');
  _closeBtn       = document.getElementById('skill-close');
  _tabs           = _overlay.querySelectorAll('.skill-tab');
  _panels         = _overlay.querySelectorAll('.skill-panel');
  _installedList  = document.getElementById('skill-installed-list');
  _installedEmpty = document.getElementById('skill-installed-empty');
  _browseList     = document.getElementById('skill-browse-list');
  _browseEmpty    = document.getElementById('skill-browse-empty');
  _searchInput    = document.getElementById('skill-search');

  // Open / close
  document.getElementById('btn-skills').addEventListener('click', show);
  _closeBtn.addEventListener('click', hide);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) hide(); });

  // Tabs
  _tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      _tabs.forEach(t => t.classList.toggle('active', t === tab));
      _panels.forEach(p => p.classList.toggle('active', p.dataset.panel === target));
      if (target === 'browse') _loadBrowse();
    });
  });

  // Search
  let _searchTimer;
  _searchInput.addEventListener('input', () => {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(_loadBrowse, 300);
  });
}

export function show() {
  _overlay.classList.remove('hidden');
  _tabs[0].click();
  _loadInstalled();
}

export function hide() {
  _overlay.classList.add('hidden');
}

// ── Installed tab ─────────────────────────────────────────────────────────

async function _loadInstalled() {
  try {
    const res = await fetch('/api/skills');
    const json = await res.json();
    const skills = json.data || [];
    _renderInstalled(skills);
  } catch { _renderInstalled([]); }
}

function _renderInstalled(skills) {
  _installedEmpty.classList.toggle('hidden', skills.length > 0);
  if (!skills.length) {
    _installedList.innerHTML = '';
    return;
  }
  _installedList.innerHTML = skills.map(s => `
    <div class="skill-card ${s.active ? 'skill-card-active' : ''}" data-slug="${_esc(s.slug)}">
      <div class="skill-card-header">
        <span class="skill-card-icon">${s.icon || '◆'}</span>
        <div class="skill-card-info">
          <span class="skill-card-name">${_esc(s.name)}</span>
          <span class="skill-card-meta">${_esc(s.category)} · v${_esc(s.version)}${s.author ? ' · ' + _esc(s.author) : ''}</span>
        </div>
        <div class="skill-card-actions">
          <span class="skill-card-status-tag ${s.active ? 'active' : ''}" data-slug="${_esc(s.slug)}">${s.active ? '激活' : '未激活'}</span>
          <button class="skill-card-uninstall-btn" data-slug="${_esc(s.slug)}" title="卸载">✕</button>
        </div>
      </div>
      <p class="skill-card-desc">${_esc(s.oneLiner)}</p>
    </div>
  `).join('');

  // Bind click handlers (card body → detail)
  _installedList.querySelectorAll('.skill-card').forEach(card => {
    card.addEventListener('click', (e) => {
      // Don't navigate if clicking action buttons
      if (e.target.closest('.skill-card-actions')) return;
      _showSkillDetail(card.dataset.slug);
    });
  });

  // Bind status tag toggle
  _installedList.querySelectorAll('.skill-card-status-tag').forEach(tag => {
    tag.addEventListener('click', async (e) => {
      e.stopPropagation();
      const slug = tag.dataset.slug;
      const isActive = tag.classList.contains('active');
      const action = isActive ? 'deactivate' : 'activate';
      await fetch(`/api/skills/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug }),
      });
      _loadInstalled();
    });
  });

  // Bind uninstall buttons
  _installedList.querySelectorAll('.skill-card-uninstall-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const slug = btn.dataset.slug;
      if (!confirm(`确定卸载技能 "${slug}"？`)) return;
      const res = await fetch('/api/skills/uninstall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug }),
      });
      const data = await res.json();
      if (data.code === 200) _loadInstalled();
      else alert(data.error || '卸载失败');
    });
  });
}

// ── Browse tab (from Resource Center) ─────────────────────────────────────

let _browseCache = [];

async function _loadBrowse() {
  const q = _searchInput.value.trim();
  try {
    let rcUrl = '';
    try {
      const cfgRes = await fetch('/api/config');
      const cfgData = await cfgRes.json();
      rcUrl = cfgData?.data?.services?.resource_center?.url || 'https://motus.phanthy.com';
    } catch { rcUrl = 'https://motus.phanthy.com'; }

    const params = new URLSearchParams();
    if (q) params.set('search', q);
    params.set('limit', '20');

    const res = await fetch(`${rcUrl}/api/skills?${params}`);
    const json = await res.json();
    const skills = json.data || [];
    _browseCache = skills;
    _renderBrowse(skills);
  } catch (e) {
    _browseList.innerHTML = `<div class="skill-empty">无法连接技能广场: ${e.message}</div>`;
    _browseEmpty.classList.add('hidden');
  }
}

function _renderBrowse(skills) {
  _browseEmpty.classList.toggle('hidden', skills.length > 0);
  if (!skills.length) {
    _browseList.innerHTML = '';
    return;
  }
  _browseList.innerHTML = skills.map(s => `
    <div class="skill-card" data-slug="${_esc(s.slug)}">
      <div class="skill-card-header">
        <span class="skill-card-icon">${s.icon || '◆'}</span>
        <div class="skill-card-info">
          <span class="skill-card-name">${_esc(s.name)}</span>
          <span class="skill-card-meta">${_esc(s.category)} · v${s.version} · ${s.author?.name || '匿名'}</span>
        </div>
      </div>
      <p class="skill-card-desc">${_esc(s.oneLiner)}</p>
    </div>
  `).join('');

  // Bind click handlers
  _browseList.querySelectorAll('.skill-card').forEach(card => {
    card.addEventListener('click', () => {
      const skill = _browseCache.find(s => s.slug === card.dataset.slug);
      if (skill) _showBrowseSkillDetail(skill);
    });
  });
}

// ── Skill Detail View ──────────────────────────────────────────────────────

async function _showSkillDetail(slug) {
  try {
    const res = await fetch(`/api/skills/${encodeURIComponent(slug)}`);
    const json = await res.json();
    if (json.code !== 200) { alert(json.error || '获取失败'); return; }
    _renderSkillDetail(json.data, 'installed');
  } catch (e) { alert('获取技能详情失败: ' + e.message); }
}

function _showBrowseSkillDetail(skill) {
  _renderSkillDetail(skill, 'browse');
}

function _renderSkillDetail(skill, context) {
  const container = context === 'installed' ? _installedList : _browseList;
  const emptyEl = context === 'installed' ? _installedEmpty : _browseEmpty;
  emptyEl.classList.add('hidden');

  const requiredToolsHtml = (skill.requiredTools || []).length
    ? `<div class="skill-detail-section">
        <div class="skill-detail-section-label">依赖工具</div>
        <div class="skill-detail-tools">${(skill.requiredTools || []).map(t => `<span class="skill-detail-tool-pill">${_esc(t)}</span>`).join('')}</div>
      </div>` : '';

  const configSchemaHtml = skill.configSchema
    ? `<div class="skill-detail-section">
        <div class="skill-detail-section-label">配置模式</div>
        <pre class="skill-detail-schema">${_esc(JSON.stringify(skill.configSchema, null, 2))}</pre>
      </div>` : '';

  const installedAtHtml = skill.installedAt
    ? `<div class="skill-detail-section">
        <div class="skill-detail-section-label">安装时间</div>
        <div class="skill-detail-text">${_esc(skill.installedAt.replace('T', ' ').slice(0, 19))}</div>
      </div>` : '';

  const actionsHtml = context === 'browse' ? `
    <div class="skill-detail-actions">
      <button class="skill-btn skill-btn-primary" id="skill-detail-install">安装</button>
    </div>
  ` : '';

  container.innerHTML = `
    <button class="skill-detail-back" id="skill-detail-back">← 返回</button>
    <div class="skill-detail-header">
      <div class="skill-detail-icon-lg">${skill.icon || '◆'}</div>
      <div class="skill-detail-meta">
        <div class="skill-detail-title">${_esc(skill.name)}</div>
        <div class="skill-detail-badges">
          <span class="skill-detail-badge version">v${_esc(skill.version || '1.0.0')}</span>
          ${skill.category ? `<span class="skill-detail-badge">${_esc(skill.category)}</span>` : ''}
          ${skill.author ? `<span class="skill-detail-badge">${_esc(typeof skill.author === 'object' ? skill.author.name : skill.author)}</span>` : ''}
          ${skill.active ? '<span class="skill-detail-badge active">激活中</span>' : ''}
        </div>
      </div>
    </div>
    ${skill.description ? `<div class="skill-detail-section"><div class="skill-detail-section-label">描述</div><div class="skill-detail-text">${_esc(skill.description)}</div></div>` : ''}
    ${skill.instruction ? `<div class="skill-detail-section"><div class="skill-detail-section-label">指令内容</div><pre class="skill-detail-instruction">${_esc(skill.instruction)}</pre></div>` : ''}
    ${requiredToolsHtml}
    ${configSchemaHtml}
    ${installedAtHtml}
    ${actionsHtml}
  `;

  // Back button
  container.querySelector('#skill-detail-back').addEventListener('click', () => {
    if (context === 'installed') _loadInstalled();
    else _renderBrowse(_browseCache);
  });

  // Action buttons
  if (context === 'browse') {
    container.querySelector('#skill-detail-install').addEventListener('click', async () => {
      try {
        const res = await fetch('/api/skills/install', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ slug: skill.slug }),
        });
        const data = await res.json();
        if (data.code === 200) {
          _tabs[0].click();
          _loadInstalled();
        } else {
          alert(data.error || '安装失败');
        }
      } catch (e) { alert('安装失败: ' + e.message); }
    });
  }
}

// ── Legacy global handlers (kept for backward compat) ─────────────────────

window.__skillInstall = async function(slug) {
  try {
    const res = await fetch('/api/skills/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug }),
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { code: res.status, error: text.slice(0, 100) }; }
    if (data.code === 200) {
      _tabs[0].click();
      _loadInstalled();
    } else {
      alert(data.error || '安装失败');
    }
  } catch (e) { alert('安装失败: ' + e.message); }
};

window.__skillUninstall = async function(slug) {
  if (!confirm(`确定卸载技能 "${slug}"？`)) return;
  try {
    const res = await fetch('/api/skills/uninstall', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug }),
    });
    const data = await res.json();
    if (data.code === 200) _loadInstalled();
    else alert(data.error || '卸载失败');
  } catch (e) { alert('卸载失败: ' + e.message); }
};

// ── Util ──────────────────────────────────────────────────────────────────

function _esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
