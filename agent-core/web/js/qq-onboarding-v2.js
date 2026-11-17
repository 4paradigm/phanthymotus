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
  root.querySelector('[data-qq-wizard-title]').hidden = completed;
  root.classList.toggle('qq-qr-result', completed);

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
    const icon = root.querySelector('[data-qq-result-icon]');
    if (sessionStatus === 'connected') {
      title.textContent = '连接成功';
      icon.textContent = '✓';
    } else if (sessionStatus === 'configured') {
      title.textContent = '已授权，尚未就绪';
      icon.textContent = '…';
    } else {
      title.textContent = '需要选择助手';
      icon.textContent = '!';
    }
    success.classList.toggle('qq-qr-success--pending', sessionStatus !== 'connected');
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
    '<header class="qq-onboard-top"><h2 data-qq-wizard-title>使用 QQ 扫码创建并连接机器人</h2>',
    '<button class="qq-onboard-close" type="button" data-qq-close aria-label="关闭">×</button></header>',
    '<div class="qq-qr-panel" data-qq-qr-panel><div class="qq-qr-spinner" data-qq-spinner aria-label="正在生成二维码"></div>',
    '<img data-qq-qr alt="QQ 扫码连接二维码" hidden></div>',
    '<span class="qq-qr-status" data-qq-status role="status" aria-live="polite">正在生成安全连接二维码…</span>',
    '<div class="qq-qr-success" data-qq-success hidden><span data-qq-result-icon aria-hidden="true">✓</span><strong data-qq-success-title></strong></div>',
    '<p class="qq-qr-error" data-qq-error role="alert" hidden></p>',
    '<div class="qq-qr-actions"><button class="btn-ghost" data-qq-cancel>取消</button>',
    '<button class="btn-ghost" data-qq-refresh hidden>重新获取二维码</button>',
    '<button class="btn-primary" data-qq-finish hidden>完成</button></div>',
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
