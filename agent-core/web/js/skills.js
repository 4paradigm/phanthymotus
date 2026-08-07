/**
 * skills.js — 技能管理 modal（安装/卸载/浏览/详情/编辑/发布/我的技能/RC登录）
 */

let _overlay, _closeBtn, _tabs, _panels;
let _installedList, _installedEmpty, _browseList, _browseEmpty, _searchInput;
let _mineList, _mineEmpty, _mineContent, _loginForm;

// ── RC Auth State ────────────────────────────────────────────────────────────

function _getRcToken() { return localStorage.getItem('rc_token'); }
function _setRcToken(token, role) {
  localStorage.setItem('rc_token', token);
  if (role) localStorage.setItem('rc_role', role);
}
function _clearRcToken() {
  localStorage.removeItem('rc_token');
  localStorage.removeItem('rc_role');
}
function _isRcLoggedIn() { return !!_getRcToken(); }

function _rcHeaders() {
  const h = { 'Content-Type': 'application/json' };
  const token = _getRcToken();
  if (token) h['X-RC-Token'] = token;
  return h;
}

// ── Init ─────────────────────────────────────────────────────────────────────

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
  _mineList       = document.getElementById('skill-mine-list');
  _mineEmpty      = document.getElementById('skill-mine-empty');
  _mineContent    = document.getElementById('skill-mine-content');
  _loginForm      = document.getElementById('skill-rc-login-form');

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
      if (target === 'installed') _loadInstalled();
      if (target === 'browse') _loadBrowse();
      if (target === 'mine') _loadMine();
    });
  });

  // Search
  let _searchTimer;
  _searchInput.addEventListener('input', () => {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(_loadBrowse, 300);
  });

  // RC Login form
  document.getElementById('skill-rc-login-form-el').addEventListener('submit', _handleRcLogin);

  // My Skills create button
  document.getElementById('skill-mine-create-btn').addEventListener('click', () => _openSkillForm(null));

  // Skill form modal
  document.getElementById('skill-form-close').addEventListener('click', _closeSkillForm);
  document.getElementById('skill-form-cancel').addEventListener('click', _closeSkillForm);
  document.getElementById('skill-form-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'skill-form-overlay') _closeSkillForm();
  });
  document.getElementById('skill-form-el').addEventListener('submit', _handleSkillFormSubmit);

  // Update login status display
  _updateLoginStatus();
}

export function show() {
  _overlay.classList.remove('hidden');
  _tabs[0].click();
  _loadInstalled();
}

export function hide() {
  _overlay.classList.add('hidden');
}

// ── RC Login ─────────────────────────────────────────────────────────────────

function _updateLoginStatus() {
  const el = document.getElementById('skill-rc-login-status');
  if (_isRcLoggedIn()) {
    el.innerHTML = `<span class="rc-logged-in">RC 已登录</span> <button class="rc-logout-btn" id="rc-logout-btn">退出</button>`;
    el.querySelector('#rc-logout-btn').addEventListener('click', _handleRcLogout);
  } else {
    el.innerHTML = '';
  }
}

async function _handleRcLogin(e) {
  e.preventDefault();
  const email = document.getElementById('skill-rc-email').value.trim();
  const password = document.getElementById('skill-rc-password').value;
  const errorEl = document.getElementById('skill-rc-login-error');
  errorEl.classList.add('hidden');

  try {
    const res = await fetch('/api/skills/rc/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ identifier: email, password }),
    });
    const data = await res.json();
    if (data.code === 200 && data.data?.token) {
      _setRcToken(data.data.token, data.data.role);
      _updateLoginStatus();
      _loadMine();
    } else {
      errorEl.textContent = data.error || '登录失败';
      errorEl.classList.remove('hidden');
    }
  } catch (err) {
    errorEl.textContent = '网络错误: ' + err.message;
    errorEl.classList.remove('hidden');
  }
}

function _handleRcLogout() {
  _clearRcToken();
  _updateLoginStatus();
  _loadMine();
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

// ── My Skills tab ─────────────────────────────────────────────────────────

let _installedSlugs = new Set();

async function _fetchInstalledSlugs() {
  try {
    const res = await fetch('/api/skills');
    const json = await res.json();
    _installedSlugs = new Set((json.data || []).map(s => s.slug));
  } catch { _installedSlugs = new Set(); }
}

async function _loadMine() {
  if (!_isRcLoggedIn()) {
    _loginForm.classList.remove('hidden');
    _mineContent.classList.add('hidden');
    return;
  }
  _loginForm.classList.add('hidden');
  _mineContent.classList.remove('hidden');

  try {
    await _fetchInstalledSlugs();
    const res = await fetch('/api/skills/rc/mine', { headers: _rcHeaders() });
    const json = await res.json();
    if (json.code === 401) {
      _clearRcToken();
      _updateLoginStatus();
      _loadMine();
      return;
    }
    const skills = json.data || [];
    _renderMine(skills);
  } catch { _renderMine([]); }
}

const STATUS_LABELS = {
  draft: { label: '草稿', cls: 'status-draft' },
  pending: { label: '待审核', cls: 'status-pending' },
  published: { label: '已发布', cls: 'status-published' },
  rejected: { label: '已拒绝', cls: 'status-rejected' },
};

function _renderMine(skills) {
  _mineEmpty.classList.toggle('hidden', skills.length > 0);
  if (!skills.length) {
    _mineList.innerHTML = '';
    return;
  }
  _mineList.innerHTML = skills.map(s => {
    const st = STATUS_LABELS[s.status] || STATUS_LABELS.draft;
    const canSubmit = ['draft', 'rejected'].includes(s.status);
    const isInstalled = _installedSlugs.has(s.slug);
    return `
    <div class="skill-card" data-id="${_esc(s.id)}">
      <div class="skill-card-header">
        <span class="skill-card-icon">${s.icon || '◆'}</span>
        <div class="skill-card-info">
          <span class="skill-card-name">${_esc(s.name)}</span>
          <span class="skill-card-meta">${_esc(s.slug)} · v${_esc(s.version)}</span>
        </div>
        <div class="skill-card-actions">
          <span class="skill-mine-status ${st.cls}">${st.label}</span>
        </div>
      </div>
      <p class="skill-card-desc">${_esc(s.oneLiner)}</p>
      <div class="skill-mine-actions">
        ${isInstalled
          ? `<button class="skill-btn skill-btn-sm" data-action="uninstall" data-slug="${_esc(s.slug)}">卸载</button>`
          : `<button class="skill-btn skill-btn-sm skill-btn-primary" data-action="install" data-slug="${_esc(s.slug)}">安装</button>`
        }
        <button class="skill-btn skill-btn-sm" data-action="edit" data-id="${_esc(s.id)}">编辑</button>
        ${canSubmit ? `<button class="skill-btn skill-btn-sm skill-btn-warn" data-action="submit" data-id="${_esc(s.id)}">提交审核</button>` : ''}
        <button class="skill-btn skill-btn-sm skill-btn-danger" data-action="delete" data-id="${_esc(s.id)}">删除</button>
      </div>
    </div>`;
  }).join('');

  // Bind actions
  _mineList.querySelectorAll('[data-action="install"]').forEach(btn => {
    btn.addEventListener('click', () => _installMineSkill(btn.dataset.slug));
  });
  _mineList.querySelectorAll('[data-action="uninstall"]').forEach(btn => {
    btn.addEventListener('click', () => _uninstallMineSkill(btn.dataset.slug));
  });
  _mineList.querySelectorAll('[data-action="edit"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const skill = skills.find(s => s.id === btn.dataset.id);
      if (skill) _openSkillForm(skill);
    });
  });
  _mineList.querySelectorAll('[data-action="submit"]').forEach(btn => {
    btn.addEventListener('click', () => _submitForReview(btn.dataset.id));
  });
  _mineList.querySelectorAll('[data-action="delete"]').forEach(btn => {
    btn.addEventListener('click', () => _deleteMineSkill(btn.dataset.id));
  });
}

async function _submitForReview(id) {
  if (!confirm('确定提交审核？')) return;
  try {
    const res = await fetch(`/api/skills/rc/mine/${id}/submit`, {
      method: 'POST',
      headers: _rcHeaders(),
    });
    const data = await res.json();
    if (data.code === 200) _loadMine();
    else alert(data.error || '提交失败');
  } catch (e) { alert('提交失败: ' + e.message); }
}

async function _deleteMineSkill(id) {
  if (!confirm('确定删除此技能？')) return;
  try {
    const res = await fetch(`/api/skills/rc/mine/${id}`, {
      method: 'DELETE',
      headers: _rcHeaders(),
    });
    const data = await res.json();
    if (data.code === 200) _loadMine();
    else alert(data.error || '删除失败');
  } catch (e) { alert('删除失败: ' + e.message); }
}

async function _installMineSkill(slug) {
  try {
    const res = await fetch('/api/skills/install', {
      method: 'POST',
      headers: _rcHeaders(),
      body: JSON.stringify({ slug }),
    });
    const data = await res.json();
    if (data.code === 200) {
      _loadMine(); // refresh to show "卸载" button
    } else {
      alert(data.error || '安装失败');
    }
  } catch (e) { alert('安装失败: ' + e.message); }
}

async function _uninstallMineSkill(slug) {
  try {
    const res = await fetch('/api/skills/uninstall', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug }),
    });
    const data = await res.json();
    if (data.code === 200) {
      _loadMine(); // refresh to show "安装" button
    } else {
      alert(data.error || '卸载失败');
    }
  } catch (e) { alert('卸载失败: ' + e.message); }
}

// ── Skill Create/Edit Form ───────────────────────────────────────────────────

let _editingSkill = null;

function _openSkillForm(skill) {
  _editingSkill = skill;
  const title = document.getElementById('skill-form-title');
  title.textContent = skill ? '编辑技能' : '创建技能';

  // Fill form
  document.getElementById('sf-name').value = skill?.name || '';
  document.getElementById('sf-slug').value = skill?.slug || '';
  document.getElementById('sf-slug').disabled = !!skill;
  document.getElementById('sf-category').value = skill?.category || 'utility';
  document.getElementById('sf-icon').value = skill?.icon || '';
  document.getElementById('sf-version').value = skill?.version || '1.0.0';
  document.getElementById('sf-oneliner').value = skill?.oneLiner || '';
  document.getElementById('sf-description').value = skill?.description || '';
  document.getElementById('sf-instruction').value = skill?.instruction || '';
  document.getElementById('sf-tools').value = (skill?.requiredTools || []).join(', ');
  document.getElementById('sf-configschema').value = skill?.configSchema ? JSON.stringify(skill.configSchema, null, 2) : '';

  // If editing a published skill, auto-increment version
  if (skill?.status === 'published') {
    const parts = (skill.version || '1.0.0').split('.');
    parts[parts.length - 1] = String(Number(parts[parts.length - 1]) + 1);
    document.getElementById('sf-version').value = parts.join('.');
  }

  document.getElementById('skill-form-error').classList.add('hidden');
  document.getElementById('skill-form-overlay').classList.remove('hidden');
}

function _closeSkillForm() {
  document.getElementById('skill-form-overlay').classList.add('hidden');
  _editingSkill = null;
}

async function _handleSkillFormSubmit(e) {
  e.preventDefault();
  const errorEl = document.getElementById('skill-form-error');
  errorEl.classList.add('hidden');

  const body = {
    name: document.getElementById('sf-name').value.trim(),
    slug: document.getElementById('sf-slug').value.trim(),
    category: document.getElementById('sf-category').value,
    icon: document.getElementById('sf-icon').value.trim() || null,
    version: document.getElementById('sf-version').value.trim(),
    oneLiner: document.getElementById('sf-oneliner').value.trim(),
    description: document.getElementById('sf-description').value.trim(),
    instruction: document.getElementById('sf-instruction').value.trim(),
    requiredTools: document.getElementById('sf-tools').value
      ? document.getElementById('sf-tools').value.split(',').map(s => s.trim()).filter(Boolean)
      : [],
    configSchema: null,
  };

  // Parse config schema
  const schemaStr = document.getElementById('sf-configschema').value.trim();
  if (schemaStr) {
    try { body.configSchema = JSON.parse(schemaStr); }
    catch { errorEl.textContent = '配置模式 JSON 格式错误'; errorEl.classList.remove('hidden'); return; }
  }

  const submitBtn = document.getElementById('skill-form-submit-btn');
  submitBtn.disabled = true;
  submitBtn.textContent = '保存中…';

  try {
    let url, method;
    if (_editingSkill) {
      url = `/api/skills/rc/mine/${_editingSkill.id}`;
      method = 'PUT';
    } else {
      url = '/api/skills/rc/mine';
      method = 'POST';
    }

    const res = await fetch(url, {
      method,
      headers: _rcHeaders(),
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.code === 200) {
      _closeSkillForm();
      _loadMine();
    } else {
      errorEl.textContent = data.error || '保存失败';
      errorEl.classList.remove('hidden');
    }
  } catch (err) {
    errorEl.textContent = '网络错误: ' + err.message;
    errorEl.classList.remove('hidden');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = '保存';
  }
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
