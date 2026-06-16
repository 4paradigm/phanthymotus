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
  document.getElementById('detail-title').textContent    = topicPath;
  document.getElementById('detail-subtitle').textContent = format ? `format: ${format}` : 'live stream';

  const body = document.getElementById('detail-body');
  body.innerHTML = '';

  const hint     = format || 'activity';
  const Renderer = RENDERERS.find(r => r.canRender(hint)) || ActivityRenderer;

  _renderer = Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
  _renderer.mount(body, 'detail');

  // Connect WebSocket — /ws/bus/* is proxied through agent-core
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsHost = location.host;
  const wsUrl = `${proto}://${wsHost}/ws/bus${topicPath}`;
  _ws = new WebSocket(wsUrl);
  _ws.binaryType = 'arraybuffer';
  _ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // Binary frame — pass directly to renderer (audio PCM, sensor binary, etc.)
      if (ev.data.byteLength === 0) return;
      _renderer?.onData?.(ev.data, hint);
    } else {
      // Text frame — JSON messages
      try {
        const parsed = JSON.parse(ev.data);
        if (parsed.type === 'ping' || parsed.type === 'meta') return;
        if (parsed.type === 'error') {
          console.warn('[detail-panel] WS error:', parsed.message);
          return;
        }
      } catch {}
      const buf = new TextEncoder().encode(ev.data).buffer;
      _renderer?.onData?.(buf, hint);
    }
  };
  _ws.onclose = () => { console.debug('[detail-panel] WS closed:', topicPath); };
  _ws.onerror = (e) => { console.warn('[detail-panel] WS error:', topicPath, e); };
}

export async function showNodeDetail(mcp) {
  _cleanup();

  _panel.classList.remove('hidden');
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
