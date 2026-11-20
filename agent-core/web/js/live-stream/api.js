// MCP boundary for the live-stream card. Parse business-level responses, not only HTTP 200.
function safeError(code) {
  const messages = {
    CAMERA_UNAVAILABLE: '摄像头还没准备好，请检查 Bumi 摄像头。',
    MIC_UNAVAILABLE: '麦克风还没有声音，请检查 Bumi 麦克风。',
    APPROVAL_REQUIRED: '本场直播信息已失效，请获取新的直播信息。',
    MISSING_URL: '请先导入本场直播信息。',
    INVALID_STREAM_URL: '直播信息不正确，请重新复制。',
    STREAM_ACTIVE: '请先结束当前直播，再更换平台。',
    RTMP_CONNECT_TIMEOUT: '连接直播平台超时，请检查网络或更新直播信息。',
    RTMP_WRITE_TIMEOUT: '网络传输中断，请检查连接状态。',
    RTMP_PROCESS_EXITED: '直播连接断开，请在平台确认是否已停播。',
    CONFIG_SAVE_FAILED: '无法保存直播设置，请联系设备管理员。',
    ENCODER_START_FAILED: '机器人无法启动视频编码，请联系设备管理员。',
  };
  return messages[code] || '操作未完成，请检查机器人和平台状态后重试。';
}

function unwrap(payload) {
  if (payload?.code !== 200) throw new Error('机器人暂时没有响应，请检查连接。');
  let body = payload.data;
  if (Array.isArray(body) && body[0]?.text) {
    try { body = JSON.parse(body[0].text); }
    catch { throw new Error('机器人返回了无法识别的状态'); }
  } else if (body && typeof body === 'object' && 'result' in body) {
    body = body.result;
  }
  if (!body || typeof body !== 'object') throw new Error('机器人没有返回状态');
  if (body.ok === false || body.error_code || body.state === 'error') {
    throw new Error(safeError(body.error_code));
  }
  return body;
}

async function fetchJson(url, options) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error('机器人请求失败，请确认网络和登录状态');
  return response.json();
}

export async function callLiveStream(mcpId, action) {
  const payload = await fetchJson(
    `/api/mcp/${encodeURIComponent(mcpId)}/call?timeout_s=12`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: 'live_stream', arguments: { action } }) },
  );
  return unwrap(payload);
}

export async function authorizeStoredLiveStream(mcpId, cardId) {
  // Called only after the operator confirms this broadcast in the Canvas.
  // The previously stored secret never travels back through JavaScript.
  const path = `/api/canvas/tool-config/${encodeURIComponent(mcpId)}/live_stream/${encodeURIComponent(cardId)}/authorize-start`;
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.code !== 200 || payload.ready_to_go_live !== true) {
    throw new Error('当前直播信息无法重新授权，请检查设备状态或重新获取本场推流码。');
  }
  return true;
}

export async function startApprovedLiveStream(mcpId, cardId) {
  // Public broadcast is dispatched from authenticated Core, never raw MCP.
  const path = `/api/canvas/tool-config/${encodeURIComponent(mcpId)}/live_stream/${encodeURIComponent(cardId)}/go-live`;
  const response = await fetch(path, { method: 'POST' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.code !== 200 || !['starting', 'streaming'].includes(payload.state)) {
    throw new Error('直播没有开始，请检查机器人摄像头、麦克风和开播授权。');
  }
  return { state: payload.state };
}

export async function saveLiveStreamConfig(mcpId, cardId, url) {
  const path = `/api/canvas/tool-config/${encodeURIComponent(mcpId)}/live_stream/${encodeURIComponent(cardId)}`;
  // Never add the URL to client-side logging or error messages.
  const response = await fetch(path, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rtmp_url: url }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.code !== 200) {
    throw new Error('直播信息没有保存成功，请检查内容和机器人状态。');
  }
  if (payload.applied !== true) {
    throw new Error('当前 Dashboard 版本无法确认机器人已接收配置，请升级后重试。');
  }
  if (payload.ready_to_go_live !== true) {
    throw new Error('请从平台获取本场新的直播信息，旧信息不能重复开播。');
  }
  return true;
}
