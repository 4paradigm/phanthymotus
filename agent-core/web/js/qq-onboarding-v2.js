const endpoint = '/api/channel';
const TERMINAL = new Set(['connected', 'configured', 'action_required', 'failed', 'expired', 'cancelled']);

let root = null;
let host = null;
let finishCallback = () => {};
let sessionId = '';
let sessionStatus = '';
let pollTimer = null;
let polling = false;

async function request(path, method='GET', body) {
  const response = await fetch(endpoint + path, {
    method,
    headers: {'Content-Type': 'application/json'},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    cache: 'no-store',
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.message || '操作失败，请重试。';
    throw new Error(typeof detail === 'string' ? detail : detail.message || '操作失败，请重试。');
  }
  return payload;
}

function stopPolling() {
  if (pollTimer !== null) clearInterval(pollTimer);
  pollTimer = null;
  polling = false;
}

function setText(selector, value) {
  const node = root?.querySelector(selector);
  if (node) node.textContent = value || '';
}

function renderState(state) {
  if (!root || !state) return;
  sessionStatus = state.status || 'starting';
  setText('[data-qq-status]', state.message || '正在准备连接…');

  const image = root.querySelector('[data-qq-qr]');
  const qrPanel = root.querySelector('[data-qq-qr-panel]');
  const spinner = root.querySelector('[data-qq-spinner]');
  const refresh = root.querySelector('[data-qq-refresh]');
  const cancel = root.querySelector('[data-qq-cancel]');
  const finish = root.querySelector('[data-qq-finish]');
  const success = root.querySelector('[data-qq-success]');
  const error = root.querySelector('[data-qq-error]');

  const waiting = ['starting', 'waiting_scan'].includes(sessionStatus);
  const completed = ['connected', 'configured', 'action_required'].includes(sessionStatus);
  const failed = ['failed', 'expired', 'cancelled'].includes(sessionStatus);

  if (state.qr_data_url && sessionStatus === 'waiting_scan') {
    image.src = state.qr_data_url;
    image.hidden = false;
    spinner.hidden = true;
    qrPanel.hidden = false;
  } else if (['starting', 'configuring'].includes(sessionStatus)) {
    image.hidden = true;
    spinner.hidden = false;
    qrPanel.hidden = false;
  } else {
    image.hidden = true;
    spinner.hidden = true;
    qrPanel.hidden = true;
  }

  success.hidden = !completed;
  if (completed) {
    const title = root.querySelector('[data-qq-success-title]');
    const detail = root.querySelector('[data-qq-success-detail]');
    if (sessionStatus === 'connected') {
      title.textContent = 'QQ 已连接';
      detail.textContent = state.message || '消息输入订阅已就绪。现在可以在 QQ 中发送消息测试。';
    } else if (sessionStatus === 'configured') {
      title.textContent = 'QQ 已授权，配置已保存';
      detail.textContent = state.message || '消息输入订阅尚未就绪。你可以返回渠道列表查看运行状态。';
    } else {
      title.textContent = 'QQ 授权已完成';
      detail.textContent = '现有画布需要选择一个助手后才能开始对话。';
    }
  }
  error.hidden = !failed;
  refresh.hidden = !failed;
  cancel.hidden = !waiting;
  finish.hidden = !completed;
  if (failed) {
    error.textContent = state.message || '连接未完成，请重新扫码。';
  } else {
    error.textContent = '';
  }
  if (completed) {
    finish.textContent = sessionStatus === 'connected' ? '完成' : '返回渠道列表';
    finish.dataset.connected = sessionStatus === 'connected' ? 'true' : 'false';
  }
  if (TERMINAL.has(sessionStatus)) stopPolling();
}

async function pollStatus() {
  if (!sessionId || polling || !root) return;
  polling = true;
  try {
    const state = await request('/qq/onboard/status?session_id=' + encodeURIComponent(sessionId));
    renderState(state);
  } catch (err) {
    stopPolling();
    renderState({
      status: 'expired',
      message: err.message || '扫码会话已失效，请重新开始。',
    });
  } finally {
    polling = false;
  }
}

async function startOnboarding() {
  stopPolling();
  sessionId = '';
  sessionStatus = 'starting';
  renderState({
    status: 'starting',
    message: '正在生成安全连接二维码…',
  });
  const refresh = root.querySelector('[data-qq-refresh]');
  refresh.disabled = true;
  try {
    const state = await request('/qq/onboard/start', 'POST');
    sessionId = state.session_id || '';
    if (!sessionId) throw new Error('没有获取到扫码会话，请重试。');
    renderState(state);
    if (!TERMINAL.has(state.status)) {
      pollTimer = setInterval(pollStatus, 1500);
      await pollStatus();
    }
  } catch (err) {
    renderState({status: 'failed', message: err.message || '无法生成二维码，请重试。'});
  } finally {
    refresh.disabled = false;
  }
}

export function openQQWizard({onConnected}={}) {
  if (document.getElementById('qq-guided-setup')) return;
  finishCallback = onConnected || (() => {});
  host = document.getElementById('channel-list');
  const add = document.getElementById('channel-add-btn');
  if (!host) return;

  host.hidden = true;
  if (add) add.hidden = true;
  root = document.createElement('section');
  root.id = 'qq-guided-setup';
  root.className = 'qq-onboarding qq-qr-onboarding';
  host.closest('.channel-modal')?.classList.add('qq-onboarding-active');
  root.innerHTML = [
    '<header class="qq-onboard-top"><div><span class="qq-onboard-kicker">QQ 连接</span><h2>扫码连接 QQ</h2></div>',
    '<button class="qq-onboard-close" type="button" data-qq-close aria-label="关闭">×</button></header>',
    '<p class="qq-qr-intro">打开手机 QQ 扫描二维码，按照手机上的提示确认。授权完成后，系统会自动配置并连接。</p>',
    '<div class="qq-qr-panel" data-qq-qr-panel><div class="qq-qr-spinner" data-qq-spinner aria-label="正在生成二维码"></div>',
    '<img data-qq-qr alt="QQ 扫码连接二维码" hidden></div>',
    '<p class="qq-qr-status" data-qq-status aria-live="polite">正在生成安全连接二维码…</p>',
    '<div class="qq-qr-success" data-qq-success hidden><span>✓</span><strong data-qq-success-title></strong><p data-qq-success-detail></p></div>',
    '<p class="qq-qr-error" data-qq-error role="alert" hidden></p>',
    '<div class="qq-qr-actions"><button class="btn-ghost" data-qq-cancel>取消</button>',
    '<button class="btn-ghost" data-qq-refresh hidden>重新获取二维码</button>',
    '<button class="btn-primary" data-qq-finish hidden>完成</button></div>',
    '<p class="qq-qr-footnote">无需填写机器人编号或密钥。连接凭证由系统在服务端安全保存。</p>',
    '<a class="qq-qr-help" href="https://q.qq.com/qqbot/dashboard/" target="_blank" rel="noopener noreferrer">打开 QQ 官方平台 ↗</a>',
  ].join('');
  host.parentNode.insertBefore(root, host);

  function close({refresh=false}={}) {
    stopPolling();
    const pending = sessionId && ['starting', 'waiting_scan'].includes(sessionStatus);
    if (pending) {
      request('/qq/onboard/cancel', 'POST', {session_id: sessionId}).catch(() => {});
    }
    root?.remove();
    root = null;
    host.hidden = false;
    host.closest('.channel-modal')?.classList.remove('qq-onboarding-active');
    if (add) add.hidden = false;
    sessionId = '';
    if (refresh) finishCallback();
  }

  root.querySelector('[data-qq-close]').onclick = () => close();
  root.querySelector('[data-qq-cancel]').onclick = () => close();
  root.querySelector('[data-qq-refresh]').onclick = () => startOnboarding();
  root.querySelector('[data-qq-finish]').onclick = () => close({refresh:true});
  startOnboarding();
}
