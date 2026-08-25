/**
 * account.js — 「我的」：Resource Center 账号 + 技能广场 / 解决方案市场入口。
 *
 * 桌面端是顶栏「设置」右边的 modal，移动端是第四个底 tab 的整页面板 ——
 * 两边渲染同一份 HTML（`_render()` 写进两个容器），避免两套逻辑各自漂移。
 *
 * 这里同时是 Resource Center 账号态的唯一出处：token 存在 localStorage.rc_token，
 * skills.js / solutions.js 都从这里取，401 也统一在 `rcFetch` 里处理 ——
 * 以前 Resource Center 的 401 原文（Unauthorized）会被十几处 alert 直接抛给用户，而登录
 * 表单埋在「设置 → 技能 → 我的技能」里，用户根本找不到。
 */

const TOKEN_KEY = 'rc_token';
const ROLE_KEY  = 'rc_role';
const USER_KEY  = 'rc_user_id';

let _overlay, _closeBtn;
let _user = null;         // {id, role, name, email}，未登录为 null
let _mode = 'login';      // login | register
let _notice = '';         // 顶部提示（如"登录已过期"）
let _busy = false;
let _formError = '';
let _rcUrl = '';

// ── Token（唯一出处，skills.js / solutions.js 都 import 这里）──────────────

export function getRcToken() { return localStorage.getItem(TOKEN_KEY) || ''; }

export function setRcToken(token, role, userId) {
  localStorage.setItem(TOKEN_KEY, token);
  if (role) localStorage.setItem(ROLE_KEY, role);
  if (userId) localStorage.setItem(USER_KEY, userId);
}

export function clearRcToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ROLE_KEY);
  localStorage.removeItem(USER_KEY);
  _user = null;
}

/** 登录时记下的身份，用于 Resource Center 暂时问不到 session 时兜底显示。 */
function _storedUser() {
  const id = localStorage.getItem(USER_KEY);
  if (!id) return null;
  return { id, role: localStorage.getItem(ROLE_KEY) || '', name: null, email: null };
}

export function isRcLoggedIn() { return !!getRcToken(); }

/** 带 Resource Center token 的请求头。 */
export function rcHeaders(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  const token = getRcToken();
  if (token) h['X-RC-Token'] = token;
  return h;
}

/**
 * 统一的 Resource Center 请求：自动带 token，遇 401 就清 token 并把用户带到「我的」。
 * 返回后端的 JSON（`{code, data|error}`）；401 时也会返回，调用方可以直接
 * 忽略 —— 提示已经由这里给出了。
 */
export async function rcFetch(url, opts = {}) {
  const res = await fetch(url, { ...opts, headers: rcHeaders(opts.headers || {}) });
  let json;
  try { json = await res.json(); } catch { json = { code: res.status, error: '响应解析失败' }; }
  if (json.code === 401 || res.status === 401) {
    const hadToken = isRcLoggedIn();
    clearRcToken();
    showAccount(hadToken ? '登录已过期，请重新登录' : '该操作需要先登录');
  }
  return json;
}

// ── Init ────────────────────────────────────────────────────────────────────

export function initAccount() {
  _overlay  = document.getElementById('account-overlay');
  if (!_overlay) return;
  _closeBtn = document.getElementById('account-close');

  document.getElementById('btn-account').addEventListener('click', () => showAccount());
  _closeBtn.addEventListener('click', hideAccount);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) hideAccount(); });

  // 两个容器共用一套事件委托
  ['account-content', 'mobile-account-content'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', _onClick);
    if (el) el.addEventListener('submit', _onSubmit);
  });

  _loadRcUrl();
  refreshAccount();
}

/**
 * 打开「我的」。桌面开 modal；移动端切到第四个 tab（面板本身常驻 DOM）。
 * notice 会显示在顶部，用于"登录已过期"这类被动跳转的说明。
 */
export function showAccount(notice = '') {
  _notice = notice;
  _formError = '';
  const isMobileView = window.matchMedia('(max-width: 768px)').matches;
  if (isMobileView) {
    document.querySelector('.tabbar-btn[data-tab="account"]')?.click();
  } else {
    _overlay?.classList.remove('hidden');
  }
  refreshAccount();
}

export function hideAccount() {
  _overlay?.classList.add('hidden');
  _notice = '';
}

/** 重新拉身份与两个市场的摘要，然后重绘。 */
export async function refreshAccount() {
  _render();                       // 先用现有状态画一次，避免闪空白
  if (getRcToken()) {
    try {
      const res = await fetch('/api/account/session', { headers: rcHeaders() });
      const json = await res.json();
      if (json.code === 200) {
        _user = json.data;
      } else if (json.code === 401) {
        // token 真的失效了：静默回到未登录态。这是"过期 token 到处报
        // Unauthorized"的根治点 —— 在用户动手之前就把状态纠正过来。
        clearRcToken();
        if (!_notice) _notice = '登录已过期，请重新登录';
      } else {
        // 问不到（Resource Center 还没升级到支持 Bearer 的 session、或临时不可用）：
        // 保留 token，用登录时记下的身份兜底，别把用户踢下线。
        _user = _storedUser();
      }
    } catch {
      _user = _storedUser();       // 网络问题同样不清 token
    }
  } else {
    _user = null;
  }
  _syncBadges();
  _render();
  _loadEntrySummaries();
}

async function _loadRcUrl() {
  try {
    const json = await (await fetch('/api/account/rc-url')).json();
    _rcUrl = json.data?.url || '';
  } catch { _rcUrl = ''; }
}

// ── 渲染 ────────────────────────────────────────────────────────────────────

function _render() {
  const html = `
    ${_notice ? `<div class="account-notice">${_esc(_notice)}</div>` : ''}
    ${_user ? _renderIdentity() : _renderAuthForm()}
    <div class="account-section-label">内容</div>
    <div class="settings-list account-entry-list">
      <button class="settings-item" data-entry="skills">
        <span class="settings-item-text">
          <span class="settings-item-label">技能广场</span>
          <span class="settings-item-sub" id="account-skills-sub">安装、激活、发布技能</span>
        </span>
        <span class="settings-item-arrow">›</span>
      </button>
      <button class="settings-item" data-entry="solutions">
        <span class="settings-item-text">
          <span class="settings-item-label">解决方案市场</span>
          <span class="settings-item-sub" id="account-solutions-sub">整套画布 / 技能 / Prompt / 任务</span>
        </span>
        <span class="settings-item-arrow">›</span>
      </button>
    </div>
    <p class="account-hint">浏览、安装、加载技能与解决方案不需要登录；发布、管理自己的内容才需要。</p>`;

  ['account-content', 'mobile-account-content'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  });
}

function _renderIdentity() {
  const label = _user.email || _user.name || _user.id;
  // id 是 cuid，窄屏上整串会把"复制"挤到第二行；只显示头尾，完整值给 title 与复制
  const shortId = _user.id.length > 12
    ? `${_user.id.slice(0, 6)}…${_user.id.slice(-4)}`
    : _user.id;
  return `
    <div class="account-card account-card-signed">
      <div class="account-avatar">${_esc((label || '?').slice(0, 1).toUpperCase())}</div>
      <div class="account-identity">
        <div class="account-identity-name">${_esc(label)}</div>
        <div class="account-identity-meta">
          <span class="account-role">${_esc(_user.role || 'viewer')}</span>
          <code class="account-id" title="${_esc(_user.id)}">${_esc(shortId)}</code>
          <button class="account-copy-btn" data-copy="${_esc(_user.id)}" title="复制完整 ID">复制</button>
        </div>
      </div>
      <button class="skill-btn skill-btn-sm" data-action="logout">退出登录</button>
    </div>`;
}

function _renderAuthForm() {
  const isRegister = _mode === 'register';
  return `
    <div class="account-card">
      <div class="account-card-head">
        <div class="account-avatar account-avatar-empty">?</div>
        <div>
          <div class="account-identity-name">未登录</div>
          <div class="account-identity-sub">登录后可发布技能与解决方案</div>
        </div>
      </div>
      <form class="account-form" data-form="${isRegister ? 'register' : 'login'}">
        <input type="text" name="identifier" placeholder="邮箱账号" autocomplete="username" required>
        <input type="password" name="password" placeholder="${isRegister ? '密码（至少 8 位）' : '密码'}"
               autocomplete="${isRegister ? 'new-password' : 'current-password'}" required
               ${isRegister ? 'minlength="8"' : ''}>
        ${isRegister ? `<input type="text" name="name" placeholder="昵称（可选）" autocomplete="nickname">` : ''}
        ${_formError ? `<p class="account-form-error">${_esc(_formError)}</p>` : ''}
        <button type="submit" class="account-submit-btn" ${_busy ? 'disabled' : ''}>
          ${_busy ? '处理中…' : (isRegister ? '注册并登录' : '登录')}
        </button>
      </form>
      <div class="account-switch">
        ${isRegister
          ? `<span>已有账号？<button class="account-link" data-action="to-login">去登录</button></span>`
          : `<span>还没有账号？<button class="account-link" data-action="to-register">注册</button></span>`}
      </div>
      ${_rcUrl ? `
        <div class="account-web-link">
          <a class="account-link" href="${_esc(_rcUrl)}" target="_blank" rel="noreferrer">Resource Center 网页版 ↗</a>
        </div>` : ''}
    </div>`;
}

/** 两个入口的副文本：已装技能数 / 当前载入的方案。 */
async function _loadEntrySummaries() {
  try {
    const skills = (await (await fetch('/api/skills')).json()).data || [];
    const active = skills.filter(s => s.active).length;
    _setText('account-skills-sub',
      skills.length ? `已安装 ${skills.length} · 激活 ${active}` : '还没有安装技能');
  } catch { /* 保持默认文案 */ }

  try {
    const current = (await (await fetch('/api/solutions/current')).json()).data;
    _setText('account-solutions-sub', current && (current.name || current.slug)
      ? `当前：${current.name || current.slug}${current.version ? ' v' + current.version : ''}`
      : '尚未载入解决方案');
  } catch { /* 保持默认文案 */ }
}

/** 顶栏按钮与移动端 tab 上的账号角标。 */
function _syncBadges() {
  const label = _user ? (_user.email || _user.name || _user.id).split('@')[0] : '';
  const badge = document.getElementById('account-badge');
  if (badge) {
    badge.textContent = label;
    badge.classList.toggle('hidden', !label);
  }
  // 移动端 tab 上只用一个圆点表示"已登录"，标签放不下文字
  const tabBadge = document.getElementById('account-tab-badge');
  if (tabBadge) {
    tabBadge.textContent = '';
    tabBadge.classList.toggle('hidden', !_user);
  }
}

// ── 交互 ────────────────────────────────────────────────────────────────────

function _onClick(e) {
  const entry = e.target.closest('[data-entry]');
  if (entry) {
    // 两个市场都不需要登录就能浏览，直接点隐藏的原按钮，沿用它们自己的初始化
    hideAccount();
    document.getElementById(entry.dataset.entry === 'skills' ? 'btn-skills' : 'btn-solutions')?.click();
    return;
  }

  const copyBtn = e.target.closest('[data-copy]');
  if (copyBtn) {
    navigator.clipboard?.writeText(copyBtn.dataset.copy);
    copyBtn.textContent = '已复制';
    setTimeout(() => { copyBtn.textContent = '复制'; }, 1500);
    return;
  }

  const action = e.target.closest('[data-action]')?.dataset.action;
  if (action === 'logout') {
    clearRcToken();
    _notice = '';
    _mode = 'login';
    refreshAccount();
  } else if (action === 'to-register') {
    _mode = 'register'; _formError = ''; _render();
  } else if (action === 'to-login') {
    _mode = 'login'; _formError = ''; _render();
  }
}

async function _onSubmit(e) {
  const form = e.target.closest('[data-form]');
  if (!form) return;
  e.preventDefault();
  if (_busy) return;

  const kind = form.dataset.form;   // login | register
  const body = {
    identifier: form.identifier.value.trim(),
    password:   form.password.value,
  };
  if (kind === 'register' && form.name) body.name = form.name.value.trim();

  _busy = true; _formError = ''; _render();
  try {
    const res = await fetch(`/api/account/${kind}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = await res.json();
    if (json.code !== 200 || !json.data?.token) {
      _formError = json.error || (kind === 'register' ? '注册失败' : '登录失败');
      _busy = false; _render();
      return;
    }
    setRcToken(json.data.token, json.data.role, json.data.userId);
    _notice = '';
    _mode = 'login';
    _busy = false;
    await refreshAccount();
  } catch (err) {
    _formError = `网络错误：${err.message}`;
    _busy = false;
    _render();
  }
}

// ── Util ────────────────────────────────────────────────────────────────────

function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function _esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
