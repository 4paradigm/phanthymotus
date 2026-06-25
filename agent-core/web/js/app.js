/**
 * app.js — Entry point.
 * Mounts canvas, sidebar, deploy panel, settings panel, and activity log.
 */

import { initSidebar, renderSidebar } from './sidebar.js';
import { initCanvas, updateCanvasMcps } from './canvas.js';
import { initDeployPanel, showDeployConfirmModal } from './deploy-panel.js';
import { connectMotus } from './motus-stream.js';
import { initActivityLog }   from './activity-log.js';
import { initDetailPanel }   from './detail-panel.js';
import { initMonitorMode }   from './monitor-mode.js';
import { initSkills }        from './skills.js';
import { initHistory }       from './history.js';
import './agent-definition.js';

let _allMcps   = [];
let _topicStatuses = {};
const _pingedIds = new Set();

async function main() {
  initSidebar();
  initDetailPanel();
  initMonitorMode();
  initDeployPanel();
  initSkills();
  initHistory();

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
    _allMcps = await _fetchMcps();
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
const _UPDATE_PRIORITY = ['core', 'perception', 'driver'];

async function checkForUpdate() {
  try {
    // Sync registry to ensure manifest has latest image tags
    await fetch('/api/drivers/sync', { method: 'POST' });
    const res  = await fetch('/api/drivers');
    const json = await res.json();
    if (json.code !== 200 || !json.data) return;

    // Find services that have a newer image available vs what's running
    const updatable = json.data.filter(d => {
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
  text.textContent = `容器切换中（0s），请稍后…`;

  const timer = setInterval(() => {
    elapsed += 10;
    text.textContent = `容器切换中（${elapsed}s），请稍后…`;
  }, 10000);

  const reconnect = setInterval(async () => {
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
    } catch { /* 服务还没起来，继续等 */ }
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

main();
