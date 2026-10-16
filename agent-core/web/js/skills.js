/**
 * skills.js — 技能管理 modal（安装/卸载/浏览/详情/编辑/发布/我的技能）
 * 账号登录统一在「我的」（js/account.js）里，这里只留一个入口。
 */

import { isRcLoggedIn as _isRcLoggedIn, rcHeaders as _rcHeaders, showAccount } from './account.js';
import { openWorkbench } from './workbench.js';

let _overlay, _closeBtn, _tabs, _panels;
let _installedList, _installedEmpty, _browseList, _browseEmpty, _searchInput;
let _mineList, _mineEmpty, _mineContent, _loginForm;

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

  // 登录统一在「我的」里做，这里只留一个入口
  document.getElementById('skill-rc-login-btn')?.addEventListener('click', () => showAccount());

  // My Skills create button
  document.getElementById('skill-mine-create-btn').addEventListener('click', () => _openSkillForm(null));

  // Skill form modal
  document.getElementById('skill-form-close').addEventListener('click', _closeSkillForm);
  document.getElementById('skill-form-cancel').addEventListener('click', _closeSkillForm);
  document.getElementById('skill-form-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'skill-form-overlay') _closeSkillForm();
  });
  document.getElementById('skill-form-el').addEventListener('submit', _handleSkillFormSubmit);
  _initPackages();

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
    await _loadPending();
  } catch { _renderInstalled([]); }
}

async function _loadPending() {
  const list = document.getElementById('skill-pending-list');
  try {
    const response = await fetch('/api/skills/package/pending');
    if (!response.ok) throw new Error();
    const {drafts = [], unavailable = []} = (await response.json()).data;
    list.replaceChildren();
    if (drafts.length) _installedEmpty.classList.add('hidden');
    for (const draft of drafts) {
      const card = document.createElement('div'); card.className = 'skill-card';
      card.dataset.draftId = draft.id;
      card.innerHTML = `<div class="skill-card-header"><span class="skill-card-icon">◇</span>
        <div class="skill-card-info"><span class="skill-card-name">${_esc(draft.name)}</span>
        <span class="skill-card-meta">待验证 · ${_esc(draft.target.label)}</span></div>
        <button class="skill-btn skill-btn-primary">继续验证</button></div>`;
      card.querySelector('button').onclick = async event => {
        const button = event.currentTarget;
        button.disabled = true;
        try {
          const response = await fetch('/api/skills/package/resume', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({card_id: draft.target.instance_id, draft_id: draft.id})});
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || '无法继续验证');
          await _openPackageWorkbench(result.data.target, result.data.ui);
        } catch (error) { alert(error.message); button.disabled = false; }
      };
      list.append(card);
    }
    if (unavailable.length) {
      const note = document.createElement('p'); note.className = 'skill-detail-text';
      note.textContent = '部分设备未连接，待验证技能将在连接后显示。'; list.append(note);
    }
  } catch { list.textContent = '暂时无法读取待验证技能，请重新打开重试。'; }
}

async function _openPackageWorkbench(target, ui) {
  await openWorkbench({mcp: target.mcp_id, tool: target.tool, instanceId: target.instance_id},
    target.workbench, ui, () => show());
  hide();
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
      // token 失效：交给「我的」清理并提示，这里只回到未登录态
      showAccount('登录已过期，请重新登录');
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
  if (context === 'installed') document.getElementById('skill-pending-list').replaceChildren();
  // `attachment` is the whole test. The page used to also sniff a
  // `taught-<32 hex>` slug; the server resolves legacy slugs from the
  // prefix each card declares, so the browser does not need the convention.
  const packagedSkill = !!skill.attachment;

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
  ` : packagedSkill ? `
    <div class="skill-detail-actions"><button class="skill-btn" id="skill-package-refresh">更新调用说明</button><button class="skill-btn skill-btn-primary" id="skill-package-export">下载技能包</button></div>
    <p class="skill-detail-text" id="skill-package-export-status" role="status"></p>` : '';

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
    ${_compatibilityHtml(skill.compatibility)}
    ${packagedSkill && context === 'installed' ? '<div id="skill-session" class="skill-detail-section"></div>' : ''}
    ${skill.instruction ? actionSkill
      ? `<details class="skill-detail-section"><summary class="skill-detail-text">调用说明</summary><pre class="skill-detail-instruction">${_esc(skill.instruction)}</pre></details>`
      : `<div class="skill-detail-section"><div class="skill-detail-section-label">指令内容</div><pre class="skill-detail-instruction">${_esc(skill.instruction)}</pre></div>` : ''}
    ${requiredToolsHtml}
    ${configSchemaHtml}
    ${installedAtHtml}
    ${actionsHtml}
  `;
  _overlay.querySelector('.skill-modal-body').scrollTop = 0;
  if (packagedSkill && context === 'installed') _loadSession(skill.slug, container.querySelector('#skill-session'));

  // Back button
  container.querySelector('#skill-detail-back').addEventListener('click', () => {
    if (context === 'installed') _loadInstalled();
    else _renderBrowse(_browseCache);
  });

  // Action buttons
  container.querySelector('#skill-package-refresh')?.addEventListener('click', async event => {
    const button = event.currentTarget, status = container.querySelector('#skill-package-export-status');
    button.disabled = true; status.textContent = '正在核对来源并更新说明…';
    try {
      const response = await fetch(`/api/skills/package/refresh/${encodeURIComponent(skill.slug)}`, {method: 'POST'});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || '更新失败');
      await _showSkillDetail(skill.slug);
      container.querySelector('#skill-package-export-status').textContent = '调用说明与型号信息已更新。';
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  });
  container.querySelector('#skill-package-export')?.addEventListener('click', async event => {
    const button = event.currentTarget, status = container.querySelector('#skill-package-export-status');
    button.disabled = true; status.textContent = '正在读取完整技能包…';
    try {
      const response = await fetch(`/api/skills/package/export/${encodeURIComponent(skill.slug)}`);
      if (!response.ok) { const error = await response.json(); throw new Error(error.detail || '导出失败'); }
      const url = URL.createObjectURL(await response.blob()), link = document.createElement('a');
      link.href = url; link.download = `${skill.slug}.motus-skill`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status.textContent = '已下载完整技能包。导入其他设备后需重新试运行。';
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  });
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

async function _loadSession(slug, area) {
  try {
    const response = await fetch(`/api/skills/package/execution/${encodeURIComponent(slug)}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '无法读取自动执行设置');
    if (!result.data.supported) return _loadTemporarySession(slug, area);
    const current = result.data;
    area.innerHTML = `<div class="skill-detail-section-label">对话调用</div>
      <label class="skill-detail-text"><input type="checkbox" id="skill-auto-execute" ${current.enabled ? 'checked' : ''} ${current.device_enabled ? 'disabled' : ''}> 允许自动执行</label>
      <p class="skill-detail-text">${current.device_enabled ? '已按卡片设置允许本设备所有已验证技能自动执行。' : '开启后，直接说技能名称即可执行，无需逐次确认。'}</p>
      <p class="skill-detail-text">设置保存在本设备，停止项目或重启后保留；每次仍检查设备、起点及控制权。</p>
      <p class="skill-detail-text" role="status" id="skill-auto-execute-status">${current.binding_changed ? '技能或设备绑定已变化，请核对后重新保存设置。' : ''}</p>`;
    const checkbox = area.querySelector('#skill-auto-execute');
    checkbox.onchange = async () => {
      checkbox.disabled = true;
      try {
        const response = await fetch(`/api/skills/package/execution/${encodeURIComponent(slug)}`, {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: checkbox.checked})});
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || '设置失败');
        await _loadSession(slug, area);
      } catch (error) {
        area.querySelector('#skill-auto-execute-status').textContent = error.message;
        checkbox.checked = current.enabled; checkbox.disabled = false;
      }
    };
  } catch (error) { area.textContent = error.message; }
}

async function _loadTemporarySession(slug, area) {
  try {
    const response = await fetch(`/api/skills/package/session/${encodeURIComponent(slug)}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '无法读取会话授权');
    if (!result.data.supported) return;
    const current = result.data;
    area.innerHTML = `<div class="skill-detail-section-label">对话调用</div>
      <p class="skill-detail-text">${current.active ? `本次已允许直接执行 · 剩余 ${Math.ceil(current.remaining_seconds/60)} 分钟` : '默认每次确认。可为本机此技能授权一次，后续直接说技能名称。'}</p>
      <p class="skill-detail-text">使用条件：${_esc(current.conditions)}</p>
      ${current.active ? '' : '<label class="skill-detail-text"><input type="checkbox" id="skill-session-confirm"> 已确认移动路径、活动空间及接管，允许本次直接调用（最长 30 分钟）。</label>'}
      <div class="skill-detail-actions"><button class="skill-btn" id="skill-session-toggle" ${current.active ? '' : 'disabled'}>${current.active ? '结束本次授权' : '允许本次直接调用'}</button></div>
      <p class="skill-detail-text">停止、设备重连或动作与标定变化后需重新授权。</p><p role="status" id="skill-session-status" class="skill-detail-text"></p>`;
    const button = area.querySelector('#skill-session-toggle');
    area.querySelector('#skill-session-confirm')?.addEventListener('change', event => {button.disabled = !event.target.checked;});
    button.onclick = async () => {
      button.disabled = true;
      try {
        const response = await fetch(`/api/skills/package/session/${encodeURIComponent(slug)}`, {method: 'POST',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: !current.active,
            workspace_clear: !current.active, take_control: !current.active})});
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || '设置失败');
        await _loadSession(slug, area);
      } catch (error) {area.querySelector('#skill-session-status').textContent = error.message; button.disabled = false;}
    };
  } catch (error) { area.textContent = error.message; }
}

/** `compatibility` is opaque to Core. The device that wrote it knows what its
 * fields mean, so it supplies the pills to show in `labels`; we render those
 * and fall back to the raw scalars. An earlier version mapped `dual_arm` to
 * 双臂 and `fixed_hand` to 固定手 right here — kinematics vocabulary in the
 * host, for one robot shape, which a second shape could only extend by editing
 * this file. */
function _compatibilityHtml(value) {
  if (!value || typeof value !== 'object') return '';
  const labels = (Array.isArray(value.labels) ? value.labels : _scalarLabels(value))
    .filter(v => typeof v === 'string' && v.trim()).slice(0, 12);
  if (!labels.length && !Object.keys(value).length) return '';
  return `<div class="skill-detail-section"><div class="skill-detail-section-label">适用设备</div>
    <div class="skill-detail-tools">${labels.map(v => `<span class="skill-detail-tool-pill">${_esc(v)}</span>`).join('')}</div>
    <details><summary class="skill-detail-text">兼容信息</summary><pre class="skill-detail-schema">${_esc(JSON.stringify(value, null, 2))}</pre></details></div>`;
}

/** Scalar fields, in declaration order, for a device that sends no `labels`. */
function _scalarLabels(value) {
  return Object.entries(value)
    .filter(([key, v]) => key !== 'tool' && (typeof v === 'string' || typeof v === 'number'))
    .map(([, v]) => String(v));
}

function _initPackages() {
  const get = id => document.getElementById('skill-package-' + id);
  let file, targets = [], busy = false, revision = 0;
  const close = () => { if (!busy) get('overlay').classList.add('hidden'); };
  get('close').onclick = close; get('cancel').onclick = close;
  get('open').onclick = () => { get('overlay').classList.remove('hidden'); get('file').focus(); };
  get('file').onchange = async () => {
    const version = ++revision;
    file = null; targets = []; get('install').disabled = true; get('target').disabled = true;
    get('target').replaceChildren(); get('summary').replaceChildren(); get('error').textContent = '';
    const selected = get('file').files[0];
    if (!selected) return;
    try {
      if (selected.size > 16*1024*1024) throw new Error('技能包须小于 16 MiB');
      const body = new FormData(); body.append('file', selected);
      const response = await fetch('/api/skills/package/inspect', {method: 'POST', body});
      const result = await response.json();
      if (version !== revision) return;
      if (!response.ok) throw new Error(result.detail || '无法读取技能包');
      const value = result.data;
      get('summary').innerHTML = `<h3 class="skill-detail-title">${_esc(value.manifest.skill?.name || '技能包')}</h3>` + _compatibilityHtml(value.manifest.compatibility);
      // Already filtered server-side by declared protocol and tool: routing a
      // package is a transport decision, not something the page should redo by
      // reaching into the manifest.
      targets = value.targets || [];
      get('target').replaceChildren(new Option('请选择目标设备…', ''), ...targets.map(t => new Option(t.label, t.instance_id)));
      if (targets.length === 1) get('target').value = targets[0].instance_id;
      get('target').disabled = !targets.length;
      if (!targets.length) throw new Error('画布中没有支持此技能包的设备卡片，请先连接对应驱动。');
      file = selected; get('install').disabled = !get('target').value;
    } catch (error) { if (version === revision) get('error').textContent = error.message; }
  };
  get('target').onchange = () => { get('install').disabled = !file || !get('target').value; };
  get('install').onclick = async () => {
    if (busy || !file) return;
    busy = true; get('install').disabled = true; get('file').disabled = true;
    get('error').textContent = '正在校验并传入目标设备…';
    try {
      const body = new FormData(); body.append('file', file); body.append('card_id', get('target').value);
      const response = await fetch('/api/skills/package/install', {method: 'POST', body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || '导入失败');
      if (result.data.reused) {
        get('overlay').classList.add('hidden');
        file = null; get('file').value = '';
        await _showSkillDetail(result.data.slug);
        document.getElementById('skill-package-export-status').textContent = '已复用本机相同技能包的验证记录，无需重复试运行。';
        return;
      }
      const target = result.data.target;
      const uiResponse = await fetch(`/api/mcp/${encodeURIComponent(target.mcp_id)}/call`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tool: target.tool, arguments: {action: 'ui', instance_id: target.instance_id}}),
      });
      const uiResult = await uiResponse.json();
      if (!uiResponse.ok || uiResult.code !== 200) throw new Error('技能包已导入，请从目标卡片打开工作台继续检查。');
      const content = uiResult.data;
      const ui = Array.isArray(content) ? JSON.parse(content.find(c => c.type === 'text').text) : content;
      await _openPackageWorkbench(target, ui);
      get('overlay').classList.add('hidden'); hide();
      file = null; get('file').value = ''; get('summary').replaceChildren(); get('error').textContent = '';
    } catch (error) { get('error').textContent = error.message; }
    finally { busy = false; get('install').disabled = !file; get('file').disabled = false; }
  };
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
