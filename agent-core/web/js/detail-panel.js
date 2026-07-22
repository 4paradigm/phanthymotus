/**
 * detail-panel.js — Right-side detail panel.
 *
 * Two modes:
 *  - Topic: subscribes to /ws/bus/{topic} and renders live stream using existing renderers.
 *  - Node:  shows MCP service info (tools, status, URL).
 */

import { ActivityRenderer } from './renderers/activity.js';
import { TextRenderer }     from './renderers/text.js';
import { VideoRenderer }    from './renderers/video.js';
import { ImageRenderer }    from './renderers/image.js';
import { AudioRenderer }    from './renderers/audio.js';
import { LidarRenderer }    from './renderers/lidar.js';
import { SkeletonRenderer } from './renderers/skeleton.js';
import { CameraRenderer, DepthRenderer } from './renderers/camera.js';
import { HTMSGRenderer }    from './renderers/htmsg.js';
import { openDetailPanelMobile, closeDetailPanelMobile } from './mobile.js';

const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, ImageRenderer, AudioRenderer, LidarRenderer, HTMSGRenderer, SkeletonRenderer, TextRenderer, ActivityRenderer];

let _panel    = null;
let _renderer = null;
let _ws       = null;

export function initDetailPanel() {
  _panel = document.getElementById('detail-panel');
  document.getElementById('detail-close').addEventListener('click', _closePanel);
}

export function showTopicDetail(topicPath, format) {
  _cleanup();

  _panel.classList.remove('hidden');
  openDetailPanelMobile();
  document.getElementById('detail-title').textContent    = topicPath;
  document.getElementById('detail-subtitle').textContent = format ? `format: ${format}` : 'live stream';

  const body = document.getElementById('detail-body');
  body.innerHTML = '';

  const hint     = format || 'activity';
  const Renderer = RENDERERS.find(r => r.canRender(hint)) || ActivityRenderer;

  _renderer = Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
  _renderer.mount(body, 'detail');
  body.style.position = 'relative';
  const streamStatus = document.createElement('div');
  streamStatus.className = 'detail-stream-status';
  streamStatus.style.cssText = 'position:absolute;left:10px;top:10px;z-index:10;padding:5px 8px;border-radius:4px;background:rgba(0,0,0,.72);color:#fff;font-size:12px;max-width:calc(100% - 20px)';
  streamStatus.textContent = '正在连接数据流…';
  body.appendChild(streamStatus);
  let receivedData = false;
  let streamFailed = false;

  const showStreamStatus = (message, failed = false) => {
    if (!streamStatus.isConnected) return;
    streamFailed = streamFailed || failed;
    streamStatus.style.display = '';
    streamStatus.style.background = failed ? 'rgba(170,24,24,.88)' : 'rgba(0,0,0,.72)';
    streamStatus.textContent = message;
  };

  // Connect WebSocket — /ws/bus/* is proxied through agent-core
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsHost = location.host;
  const wsUrl = `${proto}://${wsHost}/ws/bus${topicPath}`;
  _ws = new WebSocket(wsUrl);
  _ws.binaryType = 'arraybuffer';
  _ws.onopen = () => showStreamStatus('已连接，等待首帧数据…');
  _ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // Binary frame — pass directly to renderer (audio PCM, sensor binary, etc.)
      if (ev.data.byteLength === 0) return;
      receivedData = true;
      streamStatus.style.display = 'none';
      _renderer?.onData?.(ev.data, hint);
    } else {
      // Text frame — JSON messages
      try {
        const parsed = JSON.parse(ev.data);
        if (parsed.type === 'meta') {
          showStreamStatus('已连接，等待首帧数据…');
          return;
        }
        if (parsed.type === 'ping') {
          if (!receivedData) showStreamStatus('已连接，但尚未收到数据');
          return;
        }
        if (parsed.type === 'error') {
          console.warn('[detail-panel] WS error:', parsed.message);
          const message = parsed.message || '未知错误';
          const displayMessage = /not registered/i.test(message)
            ? '数据源尚未启动：请停止后重新开启智能控制'
            : `数据流错误：${message}`;
          showStreamStatus(displayMessage, true);
          return;
        }
      } catch {}
      receivedData = true;
      streamStatus.style.display = 'none';
      const buf = new TextEncoder().encode(ev.data).buffer;
      _renderer?.onData?.(buf, hint);
    }
  };
  _ws.onclose = () => {
    console.debug('[detail-panel] WS closed:', topicPath);
    if (!receivedData && !streamFailed) showStreamStatus('数据流已关闭，未收到任何数据', true);
  };
  _ws.onerror = (e) => {
    console.warn('[detail-panel] WS error:', topicPath, e);
    showStreamStatus('无法连接数据流，请检查 topic 和驱动状态', true);
  };
}

export async function showNodeDetail(mcp) {
  _cleanup();

  _panel.classList.remove('hidden');
  openDetailPanelMobile();
  document.getElementById('detail-title').textContent    = mcp.server_name || mcp.name;
  document.getElementById('detail-subtitle').textContent = mcp.url || '';

  const body = document.getElementById('detail-body');

  const status      = mcp.online === true ? '在线' : mcp.online === false ? '离线' : '未知';
  const statusColor = mcp.online === true ? 'var(--green)' : mcp.online === false ? 'var(--red)' : 'var(--text-dim)';
  const tools       = (mcp.tools || []).map(t => typeof t === 'string' ? t : t.name).filter(Boolean);
  const topicOut    = (mcp.topic_out || []).map(t => t.topic).filter(Boolean);
  const topicIn     = (mcp.topic_in  || []).map(t => t.topic).filter(Boolean);

  // For inspection nodes: fetch registered topics from /api/topics
  let inspectionSection = '';
  if (mcp.category === 'inspection') {
    try {
      const res  = await fetch('/api/topics');
      const json = await res.json();
      const items = json.data || [];
      const rows = items.map(t => {
        const statusDot = t.status === 'active' ? '🟢' : t.status === 'online' ? '🟡' : '⚫';
        return `<div class="insp-topic-row">
          <span class="insp-topic-status">${statusDot}</span>
          <span class="insp-topic-path" title="${t.topic}">${t.topic}</span>
          <span class="insp-topic-fmt">${t.format || ''}</span>
          <span class="insp-topic-src">${t.mcp_id || ''}</span>
        </div>`;
      }).join('');

      inspectionSection = `
        <div class="node-info-row" style="align-items:flex-start;flex-direction:column;gap:4px">
          <span class="node-info-label">工作机制</span>
          <span class="node-info-value" style="color:var(--text-dim);font-size:0.78rem;line-height:1.5">
            订阅驱动注册的 ROS2 DDS topic，通过<br>
            <code style="background:var(--bg2);padding:1px 4px;border-radius:3px;font-size:0.75rem">WS /ws/bus/{topic}</code> 实时推送数据流
          </span>
        </div>
        <div class="node-info-row" style="align-items:flex-start;flex-direction:column;gap:6px">
          <span class="node-info-label">已注册 Topics（${items.length}）</span>
          ${items.length ? `<div class="insp-topic-list">${rows}</div>` : `<span class="node-info-value" style="color:var(--text-dim)">暂无 — 驱动 ping 成功后自动注册</span>`}
        </div>`;
    } catch { /* silent */ }
  }

  body.innerHTML = `
    <div class="node-info">
      <div class="node-info-row">
        <span class="node-info-label">状态</span>
        <span class="node-info-value" style="color:${statusColor}">${status}</span>
      </div>
      <div class="node-info-row">
        <span class="node-info-label">协议</span>
        <span class="node-info-value">${mcp.transport || 'http'}</span>
      </div>
      <div class="node-info-row">
        <span class="node-info-label">地址</span>
        <span class="node-info-value">${mcp.url || '—'}</span>
      </div>
      ${topicOut.length ? `
      <div class="node-info-row">
        <span class="node-info-label">输出 topic</span>
        <span class="node-info-value">${topicOut.join('<br>')}</span>
      </div>` : ''}
      ${topicIn.length ? `
      <div class="node-info-row">
        <span class="node-info-label">输入 topic</span>
        <span class="node-info-value">${topicIn.join('<br>')}</span>
      </div>` : ''}
      ${inspectionSection}
      ${tools.length ? `
      <div class="node-info-tools">
        <div class="node-info-label" style="margin-bottom:6px">工具</div>
        ${tools.map(t => `<span class="tool-chip">${t}</span>`).join('')}
      </div>` : ''}
    </div>`;
}

function _closePanel() {
  _cleanup();
  _panel.classList.add('hidden');
  closeDetailPanelMobile();
}

function _cleanup() {
  if (_renderer) {
    _renderer.unmount?.();
    _renderer = null;
  }
  if (_ws) {
    _ws.close();
    _ws = null;
  }
}
