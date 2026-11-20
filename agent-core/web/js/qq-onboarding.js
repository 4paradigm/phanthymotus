const statusText = {
  starting: '正在向腾讯 QQ 请求二维码…',
  waiting: '请使用 QQ 扫码创建并连接 QQ 机器人',
  configuring: '已授权，正在保存 QQ 渠道…',
  gateway_pending: '渠道已保存，正在连接 QQ Gateway。',
  canvas_pending: 'QQ Gateway 已连接。在 Canvas 开启智能控制后，消息才能进入助手。',
  canvas_setup: '渠道已保存。请在 Canvas 把 QQ 输入和回复卡片连接到同一个 Decision Core。',
  connected: 'QQ 已连接，Canvas 消息输入已就绪。',
  expired: '二维码已过期，请重新扫码。',
  failed: '无法完成 QQ 连接，请检查网络后重试。',
  cancelled: '已取消扫码。',
};
const finalStates = new Set(['connected', 'expired', 'failed', 'cancelled']);
const qrPrefix = 'data:image/svg+xml;base64,';

async function request(action, sessionId) {
  const response = await fetch('/api/channel/qq/qr/' + action, {
    method: 'POST',
    cache: 'no-store',
    keepalive: action === 'cancel',
    ...(sessionId ? {
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    } : {}),
  });
  if (!response.ok) throw new Error(response.status === 403
    ? '请先启用 Dashboard 身份验证。' : statusText.failed);
  return response.json();
}

export function openQQOnboarding(container, onClose) {
  container.innerHTML = `
    <div class="channel-qq-onboarding">
      <p role="status"></p>
      <img alt="QQ 连接二维码" hidden width="196" height="196" />
      <div class="channel-form-actions">
        <button class="btn-primary" type="button" data-retry hidden>重新扫码</button>
        <button class="btn-ghost" type="button" data-close>取消</button>
      </div>
    </div>`;
  const label = container.querySelector('[role=status]');
  const image = container.querySelector('img');
  const retry = container.querySelector('[data-retry]');
  const button = container.querySelector('[data-close]');
  let sessionId;
  let timer;
  let closed = false;

  function render(value) {
    label.textContent = statusText[value.status] || statusText.failed;
    retry.hidden = !['expired', 'failed', 'cancelled'].includes(value.status);
    button.textContent = value.channel_id || finalStates.has(value.status) ? '关闭' : '取消';
    image.hidden = !(value.status === 'waiting' && value.qr_data_url?.startsWith(qrPrefix));
    if (image.hidden) image.removeAttribute('src');
    else image.src = value.qr_data_url;
    if (!finalStates.has(value.status)) timer = setTimeout(() => advance('status'), 1500);
  }

  async function advance(action) {
    retry.hidden = true;
    try {
      const value = await request(action, action === 'status' ? sessionId : undefined);
      sessionId = value.session_id;
      if (closed) {
        await request('cancel', sessionId);
        return;
      }
      render(value);
    } catch (error) {
      if (!closed) {
        label.textContent = error.message;
        image.hidden = true;
        image.removeAttribute('src');
        retry.hidden = false;
      }
    }
  }

  function cancel() {
    if (closed) return;
    closed = true;
    clearTimeout(timer);
    window.removeEventListener('pagehide', cancel);
    if (sessionId) request('cancel', sessionId).catch(() => {});
    onClose();
  }

  retry.onclick = async () => {
    retry.hidden = true;
    if (sessionId) await request('cancel', sessionId).catch(() => {});
    if (!closed) {
      label.textContent = statusText.starting;
      advance('start');
    }
  };
  button.onclick = cancel;
  window.addEventListener('pagehide', cancel, { once: true });
  label.textContent = statusText.starting;
  advance('start');
  return cancel;
}
