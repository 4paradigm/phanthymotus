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

async function checkForUpdate() {
  try {
    const res  = await fetch('/api/system/update-check');
    const json = await res.json();
    if (json.data && !json.data.up_to_date) {
      showUpdateBanner(json.data);
    }
  } catch { /* 静默失败 */ }
}

function showUpdateBanner({ latest_tag, latest_image }) {
  const banner = document.getElementById('update-banner');
  document.getElementById('update-banner-text').textContent = `发现新版本 ${latest_tag}`;
  banner.classList.remove('hidden');
  document.getElementById('btn-update').onclick = () => confirmAndUpdate(latest_image, latest_tag);
}

async function confirmAndUpdate(image, tag) {
  showDeployConfirmModal(
    [{ label: 'Agent Core', currentTag: '当前版本', newTag: tag }],
    () => _doUpdate(image, tag)
  );
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
      Object.assign(mcp, {
        online:      j.data.online,
        tools:       j.data.tools       ?? mcp.tools,
        resources:   j.data.resources   ?? mcp.resources,
        render_hint: j.data.render_hint ?? mcp.render_hint,
        server_name: j.data.server_name ?? mcp.server_name,
        topic_out:   j.data.topic_out   ?? mcp.topic_out,
        topic_in:    j.data.topic_in    ?? mcp.topic_in,
        ws_path:     j.data.ws_path     ?? mcp.ws_path,
      });
    }
  } catch { /* silent */ }
}

async function updateModelLabel() {
  // no-op: model label removed from topbar
}

main();
