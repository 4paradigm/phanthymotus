/**
 * app.js — Entry point.
 * Mounts canvas, sidebar, deploy panel, settings panel, and activity log.
 */

import { getToken, setToken, verifyToken } from './auth.js';
import { initSidebar, renderSidebar } from './sidebar.js';
import { initCanvas, updateCanvasMcps } from './canvas.js';
import { initDeployPanel, showDeployConfirmModal } from './deploy-panel.js';
import { connectMotus } from './motus-stream.js';
import { initActivityLog }   from './activity-log.js';
import { initDetailPanel }   from './detail-panel.js';
import { initMonitorMode }   from './monitor-mode.js';
import { initSkills }        from './skills.js';
import { initSolutions }     from './solutions.js';
import { initAccount }       from './account.js';
import { initHistory }       from './history.js';
import { initNetwork }       from './network.js';
import { initChannels }      from './channels.js';
import { initMobile }        from './mobile.js';
import { initPerformance }   from './performance.js';
import { initUsage }         from './usage.js';
import './agent-definition.js';

let _allMcps   = [];
let _topicStatuses = {};
const _pingedIds = new Set();

async function main() {
  // Auth gate: check if auth is required
  const token = getToken();
  const noTokenValid = await verifyToken('');  // If no-token passes, auth is disabled
  if (!noTokenValid && (!token || !(await verifyToken(token)))) {
    _showLoginScreen();
    return;
  }

  // Auth passed — show app
  document.getElementById('app').style.display = '';

  initMobile();
  initSidebar();
  initDetailPanel();
  initMonitorMode();
  initDeployPanel();
  initSkills();
  initSolutions();
  initAccount();
  initHistory();
  initNetwork();
  initChannels();
  initPerformance();
  initUsage();

  // Settings dropdown (web topbar)
  _initSettingsDropdown();

  initActivityLog();

  // Connect motus WebSocket for activity log
  connectMotus();

  // Fetch MCP data first so canvas cards can resolve tool types
  _allMcps = await _fetchMcps();

  // Initialize canvas and fetch topic statuses in parallel; ping all MCPs concurrently
  await Promise.all([
    initCanvas(_allMcps),
    fetchTopicStatuses(),
    _pingNewMcps(_allMcps),
  ]);

  // Render once after all data is ready
  updateModelLabel();
  renderSidebar(_allMcps, _topicStatuses);
  updateCanvasMcps(_allMcps);

  // Poll every 10s
  setInterval(async () => {
    const fresh = await _fetchMcps();
    // Preserve online status from previous ping results
    const oldMap = Object.fromEntries(_allMcps.map(m => [m.id, m]));
    for (const m of fresh) {
      if (m.online == null && oldMap[m.id]?.online != null) {
        m.online = oldMap[m.id].online;
      }
    }
    _allMcps = fresh;
    await fetchTopicStatuses();
    renderSidebar(_allMcps, _topicStatuses);
    updateCanvasMcps(_allMcps);
    _pingNewMcps(_allMcps);
  }, 10000);

  checkForUpdate();
}

async function _fetchMcps() {
  try {
    const res  = await fetch('/api/mcp');
    const json = await res.json();
    return json.data || [];
  } catch { return []; }
}

async function fetchTopicStatuses() {
  try {
    const res = await fetch('/api/topics/status');
    const json = await res.json();
    if (json.code === 200) _topicStatuses = json.data || {};
  } catch { /* 静默失败 */ }
}

// Priority order for update banner display
const _UPDATE_PRIORITY = ['core', 'perception', 'actucore', 'driver'];

async function checkForUpdate() {
  try {
    // Sync registry to ensure manifest has latest image tags
    await fetch('/api/drivers/sync', { method: 'POST' });
    const res  = await fetch('/api/drivers');
    const json = await res.json();
    if (json.code !== 200 || !json.data) return;

    // Find services that have a newer image available vs what's running
    const updatable = json.data.filter(d => {
      if (!d.running && d.category !== 'core') return false;  // core is always running if responding
      if (!d.image || !d.running_image) return false;
      const latestTag  = _tagFromImage(d.image);
      const runningTag = _tagFromImage(d.running_image);
      return latestTag && runningTag && latestTag !== runningTag;
    }).map(d => ({
      id:         d.id,
      name:       d.name,
      category:   d.category || 'driver',
      image:      d.image,
      currentTag: _tagFromImage(d.running_image),
      latestTag:  _tagFromImage(d.image),
    }));

    if (!updatable.length) return;

    // Sort by priority: core > perception > driver
    updatable.sort((a, b) => {
      const pa = _UPDATE_PRIORITY.indexOf(a.category);
      const pb = _UPDATE_PRIORITY.indexOf(b.category);
      return (pa === -1 ? 99 : pa) - (pb === -1 ? 99 : pb);
    });

    showUpdateBanner(updatable);
  } catch { /* 静默失败 */ }
}

function _tagFromImage(image) {
  return image && image.includes(':') ? image.split(':').pop() : '';
}

function showUpdateBanner(updatable) {
  const banner = document.getElementById('update-banner');
  const text   = updatable.length === 1
    ? `${updatable[0].name} 发现新版本 ${updatable[0].latestTag}`
    : `${updatable.length} 个服务有新版本可用`;
  document.getElementById('update-banner-text').textContent = text;
  banner.classList.remove('hidden');
  document.getElementById('btn-update').onclick = () => confirmAndUpdate(updatable);
}

async function confirmAndUpdate(updatable) {
  const coreItem = updatable.find(u => u.category === 'core');
  if (coreItem) {
    // Core requires confirm modal since it restarts the whole page
    const items = updatable.map(u => ({
      label: u.name, currentTag: u.currentTag, newTag: u.latestTag,
    }));
    showDeployConfirmModal(items, () => _doUpdate(coreItem.image, coreItem.latestTag));
  } else {
    // Non-core services: deploy directly, show progress in banner
    _deployServices(updatable);
  }
}

async function _deployServices(services) {
  const btn  = document.getElementById('btn-update');
  const text = document.getElementById('update-banner-text');
  btn.disabled = true;

  for (let i = 0; i < services.length; i++) {
    const svc = services[i];
    const prefix = services.length > 1 ? `[${i + 1}/${services.length}] ` : '';
    text.textContent = `${prefix}${svc.name} 正在升级…`;

    try {
      const res = await fetch(`/api/drivers/${svc.id}/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: svc.image }),
      });
      const json = await res.json();
      if (json.code !== 200) {
        text.textContent = `${svc.name} 升级失败：${json.message || '未知错误'}`;
        btn.disabled = false;
        return;
      }
    } catch {
      text.textContent = `${svc.name} 请求失败，请检查网络`;
      btn.disabled = false;
      return;
    }
  }

  // All done
  const names = services.map(s => s.name).join('、');
  text.textContent = `${names} 升级完成`;
  btn.disabled = false;
  setTimeout(() => {
    document.getElementById('update-banner').classList.add('hidden');
  }, 3000);
}

async function _doUpdate(image, tag) {
  const btn  = document.getElementById('btn-update');
  const text = document.getElementById('update-banner-text');
  btn.disabled = true;
  text.textContent = '正在启动升级…';

  try {
    const res  = await fetch('/api/system/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
    const json = await res.json();
    if (json.code !== 200) {
      text.textContent = `升级失败：${json.message || '未知错误'}`;
      btn.disabled = false;
      return;
    }
  } catch {
    text.textContent = '请求失败，请检查网络';
    btn.disabled = false;
    return;
  }

  const poll = setInterval(async () => {
    try {
      const r = await fetch('/api/system/update-status');
      const j = await r.json();
      const d = j.data || {};
      if (d.error) {
        clearInterval(poll);
        text.textContent = `升级失败：${d.error}`;
        btn.disabled = false;
      } else if (d.step) {
        text.textContent = d.step;
      }
    } catch {
      clearInterval(poll);
      _startReconnectLoop(tag);
    }
  }, 1500);
}

function _startReconnectLoop(expectedTag) {
  const text = document.getElementById('update-banner-text');
  let elapsed = 0;
  let attempts = 0;
  text.textContent = `容器切换中（0s），请稍后…`;

  const timer = setInterval(() => {
    elapsed += 10;
    text.textContent = `容器切换中（${elapsed}s），请稍后…`;
  }, 10000);

  const reconnect = setInterval(async () => {
    attempts++;
    try {
      const res  = await fetch('/api/system/update-check');
      const json = await res.json();
      if (json.code === 200) {
        clearInterval(timer);
        clearInterval(reconnect);
        const newTag = json.data?.current_tag || expectedTag;
        text.textContent = `升级成功，版本：${newTag}`;
        setTimeout(() => location.reload(), 1500);
      }
    } catch {
      // Server might be down OR SSL cert changed after restart — force reload after 60s
      if (attempts >= 6) {
        clearInterval(timer);
        clearInterval(reconnect);
        location.reload();
      }
    }
  }, 10000);
}

async function _pingNewMcps(mcps) {
  const toPing = (mcps || []).filter(m => m.id && !_pingedIds.has(m.id));
  if (!toPing.length) return;
  toPing.forEach(m => _pingedIds.add(m.id));
  await Promise.all(toPing.map(m => _pingOne(m)));
  renderSidebar(_allMcps, _topicStatuses);
  updateCanvasMcps(_allMcps);
}

async function _pingOne(mcp) {
  try {
    const r = await fetch(`/api/mcp/${mcp.id}/ping`, { method: 'POST' });
    const j = await r.json();
    if (j.data) {
      mcp.online = j.data.online;
      // Only update tools/resources from ping if online and non-empty (avoid overwriting cached data with empty response)
      if (j.data.online && j.data.tools?.length) {
        mcp.tools       = j.data.tools;
        mcp.resources   = j.data.resources ?? mcp.resources;
        mcp.render_hint = j.data.render_hint ?? mcp.render_hint;
        mcp.topic_out   = j.data.topic_out ?? mcp.topic_out;
        mcp.topic_in    = j.data.topic_in  ?? mcp.topic_in;
      }
      if (j.data.server_name) mcp.server_name = j.data.server_name;
      if (j.data.ws_path)     mcp.ws_path     = j.data.ws_path;
    }
  } catch { /* silent */ }
}

async function updateModelLabel() {
  // no-op: model label removed from topbar
}

function _initSettingsDropdown() {
  const btn = document.getElementById('topbar-settings-btn');
  const dropdown = document.getElementById('topbar-settings-dropdown');
  if (!btn || !dropdown) return;

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('hidden');
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('#topbar-settings')) {
      dropdown.classList.add('hidden');
    }
  });

  dropdown.querySelectorAll('.settings-dropdown-item').forEach(item => {
    item.addEventListener('click', () => {
      const targetId = item.dataset.target;
      document.getElementById(targetId)?.click();
      dropdown.classList.add('hidden');
    });
  });

  // Reset modal
  _initResetModal();
}

function _initResetModal() {
  const overlay = document.getElementById('reset-overlay');
  const btnOpen = document.getElementById('btn-reset');
  const btnClose = document.getElementById('reset-close');
  const btnCancel = document.getElementById('reset-cancel');
  const btnConfirm = document.getElementById('reset-confirm');
  if (!overlay || !btnOpen) return;

  const close = () => overlay.classList.add('hidden');

  btnOpen.addEventListener('click', () => overlay.classList.remove('hidden'));
  btnClose?.addEventListener('click', close);
  btnCancel?.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  btnConfirm?.addEventListener('click', async () => {
    const restartServices = document.getElementById('reset-restart-services')?.checked || false;
    const body = {
      restart_services: restartServices,
      chat_history: document.getElementById('reset-chat-history')?.checked || false,
      system_prompt: document.getElementById('reset-system-prompt')?.checked || false,
      identity: document.getElementById('reset-identity')?.checked || false,
      memory: document.getElementById('reset-memory')?.checked || false,
      skills: document.getElementById('reset-skills')?.checked || false,
    };
    if (!Object.values(body).some(v => v)) return;

    btnConfirm.disabled = true;
    btnConfirm.textContent = '执行中...';
    try {
      const res = await fetch('/api/config/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || payload.ok === false) {
        throw new Error(payload.detail || payload.message || `HTTP ${res.status}`);
      }
      if (payload.warning) alert(payload.warning);
      overlay.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = false);
      if (restartServices) {
        // Show restart waiting overlay
        close();
        _showRestartWaiting();
      } else {
        btnConfirm.textContent = '已完成';
        setTimeout(() => { close(); btnConfirm.textContent = '确认执行'; btnConfirm.disabled = false; }, 1000);
      }
    } catch (e) {
      console.error('[reset] failed:', e);
      alert('重置失败: ' + e.message);
      btnConfirm.textContent = '失败';
      setTimeout(() => { btnConfirm.textContent = '确认执行'; btnConfirm.disabled = false; }, 2000);
    }
  });
}

function _showRestartWaiting() {
  let el = document.getElementById('restart-waiting-overlay');
  if (!el) {
    el = document.createElement('div');
    el.id = 'restart-waiting-overlay';
    el.className = 'modal-overlay';
    el.innerHTML = `
      <div class="restart-waiting">
        <div class="restart-waiting-spinner"></div>
        <p class="restart-waiting-text">服务重启中，请稍候...</p>
        <p class="restart-waiting-sub" id="restart-waiting-status">等待服务关闭...</p>
      </div>
    `;
    document.body.appendChild(el);
  }
  el.classList.remove('hidden');

  // Phase 1: Wait for agent-core to go down (up to 10s)
  // Phase 2: Then poll until it's back up
  let attempts = 0;
  let wentDown = false;
  const poll = setInterval(async () => {
    attempts++;
    const statusEl = document.getElementById('restart-waiting-status');
    try {
      const r = await fetch('/api/config', { signal: AbortSignal.timeout(2000) });
      if (r.ok && wentDown) {
        // Phase 2 complete: service is back
        clearInterval(poll);
        if (statusEl) statusEl.textContent = '服务已恢复，正在刷新...';
        setTimeout(() => location.reload(), 500);
        return;
      }
      // Still up, waiting to go down
      if (!wentDown && statusEl) statusEl.textContent = '等待服务关闭...';
    } catch (_) {
      // Service is down
      if (!wentDown) {
        wentDown = true;
        if (statusEl) statusEl.textContent = '服务已关闭，等待恢复...';
      } else {
        if (statusEl) statusEl.textContent = `等待恢复中... (${attempts * 2}s)`;
      }
    }
    if (attempts > 45) { // 90s timeout
      clearInterval(poll);
      if (statusEl) statusEl.textContent = '超时，请手动刷新页面';
    }
  }, 2000);
}

function _showLoginScreen() {
  const app = document.getElementById('app');
  if (app) app.style.display = 'none';

  let loginEl = document.getElementById('login-screen');
  if (!loginEl) {
    loginEl = document.createElement('div');
    loginEl.id = 'login-screen';
    loginEl.className = 'login-screen';
    loginEl.innerHTML = `
      <div class="login-card">
        <div class="login-brand">
          <img class="login-logo" src="https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/embodied_logo.svg" alt="PhanthyMotus">
          <div class="login-brand-text">
            <span class="brand-name">Phanthy</span><span class="brand-name-accent">Motus</span>
          </div>
        </div>
        <p class="login-subtitle">Enter access token to continue</p>
        <input type="password" class="login-input" id="login-token-input" placeholder="Access Token" autocomplete="off" />
        <button class="login-btn" id="login-btn">Login</button>
        <p class="login-hint" id="login-hint"></p>
        <p class="login-footer">Token is shown in server console on startup</p>
      </div>
    `;
    document.body.appendChild(loginEl);
  }
  loginEl.style.display = 'flex';

  const input = document.getElementById('login-token-input');
  const btn = document.getElementById('login-btn');
  const hint = document.getElementById('login-hint');

  async function doLogin() {
    const val = input.value.trim();
    if (!val) { hint.textContent = 'Please enter a token'; return; }
    hint.textContent = 'Verifying...';
    btn.disabled = true;
    const valid = await verifyToken(val);
    if (valid) {
      setToken(val);
      loginEl.style.display = 'none';
      if (app) app.style.display = '';
      main();
    } else {
      hint.textContent = 'Invalid token';
      btn.disabled = false;
    }
  }

  btn.addEventListener('click', doLogin);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') doLogin(); });
  input.focus();
}

main();
