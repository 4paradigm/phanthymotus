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
import { ControlRenderer }  from './renderers/control.js';
import { SkeletonRenderer } from './renderers/skeleton.js';
import { CameraRenderer, DepthRenderer, DepthZlibRenderer } from './renderers/camera.js';
import { openDetailPanelMobile, closeDetailPanelMobile } from './mobile.js';

const RENDERERS = [VideoRenderer, CameraRenderer, DepthRenderer, DepthZlibRenderer, ImageRenderer, AudioRenderer, LidarRenderer, SkeletonRenderer, ControlRenderer, TextRenderer, ActivityRenderer];

let _panel    = null;
let _renderer = null;
let _ws       = null;
let _status   = null;   // overlay that says why nothing is on screen
let _staleTimer = null;
let _frames   = 0;
let _textLike = false;  // 文本/活动流；其余都是画布
let _retryTimer = null;
let _session  = 0;      // 换话题时自增，用来丢弃在途的重连

// How long without a frame before the panel stops implying the stream is live.
// A <canvas> keeps its last painted pixels forever, so a stalled stream is
// indistinguishable from a running one unless something says so.
const STALE_MS = 10000;

// 连接中／等待数据这类提示始终居中：那时面板上本来就没有内容可遮。
// 断线后多久重连一次。
//
// 服务端在话题还没注册时会发一条 error 然后**关掉**连接
// （api/inspection.py 的 bus_ws）。没有重连的话，面板就永久停在"这个 topic
// 没有发出任何数据"上 —— 哪怕几秒后卡片启动了、话题注册了、数据开始流，它也
// 不会知道，只有关掉重开才会重新连上。
//
// 而"先打开面板、再启动卡片"恰恰是最自然的操作顺序。
const RETRY_MS = 3000;

const _CENTERED = 'position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);' +
  'text-align:center;font-size:13px;padding:0 16px;pointer-events:none;z-index:2';

export function initDetailPanel() {
  _panel = document.getElementById('detail-panel');
  document.getElementById('detail-close').addEventListener('click', _closePanel);
}

/**
 * Show a line of text over the renderer, or hide it with `null`.
 *
 * Without this the panel has exactly one way to express every failure — a black
 * rectangle. "Not registered", "registered but nothing has ever published",
 * "publisher stopped an hour ago" and "the renderer threw" all looked the same,
 * and the monitor tab showing the *same* topic as a picture (its canvas still
 * holding a frame painted while the stream was alive) made it read as a
 * rendering bug in this panel.
 */
function _setStatus(text, tone = 'dim') {
  if (!_status) return;
  _status.textContent = text || '';
  _status.style.display = text ? 'block' : 'none';
  _status.style.color = tone === 'warn' ? 'var(--orange, #d77757)' : 'var(--text-dim, #888)';
}

function _armStaleTimer() {
  clearTimeout(_staleTimer);
  // 文本和活动流不需要这条提示，所以连定时器都不武装。
  //
  // 它存在的理由只对画布成立：canvas 会把最后一帧永远留在屏幕上，停掉的流和
  // 活着的流长得一模一样，不说一声就分不出来。日志不是这样 —— 每一行自带
  // 时间戳，停没停它自己就说明了。对着一段会自己说话的内容再压一层提示，
  // 唯一的效果是盖住一行数据。
  if (_textLike) return;
  _staleTimer = setTimeout(() => {
    if (_frames > 0) {
      _setStatus(`已暂停 — ${Math.round(STALE_MS / 1000)} 秒没有新数据，画面是最后一帧`, 'warn');
    }
  }, STALE_MS);
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
  _textLike = Renderer === TextRenderer || Renderer === ActivityRenderer;

  _renderer = Object.assign(Object.create(Object.getPrototypeOf(Renderer)), Renderer);
  _renderer.mount(body, 'detail');

  _frames = 0;
  _status = document.createElement('div');
  _status.className = 'detail-status';
  _status.style.cssText = _CENTERED;
  body.style.position = body.style.position || 'relative';
  body.appendChild(_status);
  _setStatus('正在连接…');

  // Connect WebSocket — /ws/bus/* is proxied through agent-core.
  // Skip it when the topic is unresolved: `/ws/bus` with nothing after it
  // matches no route (`/ws/bus/{topic:path}`), so the handshake fails and
  // uvicorn logs a 500 that looks like a server fault rather than "no topic".
  if (!topicPath || topicPath === '/') {
    console.debug('[detail-panel] no topic yet, not opening a bus WS');
    _setStatus('这张卡片还没有解析出输出 topic — 先启动它', 'warn');
    return;
  }
  _connect(topicPath, hint, _session);
}

/**
 * 连一次 /ws/bus，断了就再连 —— 只要面板还停在同一个话题上。
 *
 * `session` 是打开这个面板时的代次。换了话题或关了面板，_cleanup 会把代次推进，
 * 在途的重连醒来发现自己过期就安静退出，不会往下一个话题的面板上写东西。
 */
function _connect(topicPath, hint, session) {
  if (session !== _session) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsHost = location.host;
  const wsUrl = `${proto}://${wsHost}/ws/bus${topicPath}`;
  _ws = new WebSocket(wsUrl);
  _ws.binaryType = 'arraybuffer';
  const _onFrame = (payload) => {
    _frames++;
    _setStatus(null);
    _armStaleTimer();
    _renderer?.onData?.(payload, hint);
  };
  _ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // Binary frame — pass directly to renderer (audio PCM, sensor binary, etc.)
      if (ev.data.byteLength === 0) return;
      _onFrame(ev.data);
    } else {
      // Text frame — JSON messages
      try {
        const parsed = JSON.parse(ev.data);
        if (parsed.type === 'ping') return;
        if (parsed.type === 'meta') {
          // Connected, but a frame is what proves anything is publishing.
          if (_frames === 0) _setStatus('已连接，等待数据…');
          return;
        }
        if (parsed.type === 'error') {
          console.warn('[detail-panel] WS error:', parsed.message);
          _setStatus(parsed.message || '订阅失败', 'warn');
          return;
        }
      } catch {}
      _onFrame(new TextEncoder().encode(ev.data).buffer);
    }
  };
  _ws.onclose = () => {
    console.debug('[detail-panel] WS closed:', topicPath);
    clearTimeout(_staleTimer);
    if (_frames === 0) {
      _setStatus(`还没有数据 — ${Math.round(RETRY_MS / 1000)} 秒后重试`, 'dim');
    }
    _scheduleRetry(topicPath, hint, session);
  };
  _ws.onerror = (e) => {
    console.warn('[detail-panel] WS error:', topicPath, e);
    // onerror 之后总会跟一个 onclose，重连交给它，这里只管说话。
    if (_frames === 0) _setStatus('连接失败，正在重试…', 'warn');
  };
}

function _scheduleRetry(topicPath, hint, session) {
  clearTimeout(_retryTimer);
  if (session !== _session) return;
  _retryTimer = setTimeout(() => _connect(topicPath, hint, session), RETRY_MS);
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
  // 先推进代次：在途的重连醒来会发现自己过期而退出。清 timer 之外还要这一步，
  // 因为 _connect 可能已经在跑、timer 已经烧掉了。
  _session += 1;
  clearTimeout(_retryTimer);
  _retryTimer = null;
  clearTimeout(_staleTimer);
  _staleTimer = null;
  _frames = 0;
  _textLike = false;
  if (_status) {
    _status.remove();
    _status = null;
  }
  if (_renderer) {
    _renderer.unmount?.();
    _renderer = null;
  }
  if (_ws) {
    // Drop the handlers first: onclose fires asynchronously and would otherwise
    // write "连接已断开" over the panel the next topic has already mounted.
    _ws.onmessage = _ws.onclose = _ws.onerror = null;
    _ws.close();
    _ws = null;
  }
}
