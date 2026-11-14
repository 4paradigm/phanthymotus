/** Guided QQ onboarding: no Docker instructions, no persisted browser secrets. */
let panel = null;
let channelId = '';
let refreshMode = false;
let knownQQChannels = new Map();
let ownerAlreadyBound = false;
let onReady = () => {};
let otherPlatform = () => {};

async function api(url, options = {}) {
  const res = await fetch('/api/channel' + url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail || body.message || '请求失败';
    const message = typeof detail === 'string' ? detail : (detail.message || '请求失败');
    const error = new Error(message);
    error.status = res.status;
    error.details = detail;
    throw error;
  }
  return body;
}
function post(url, obj) {
  return api(url, { method: 'POST', body: JSON.stringify(obj) });
}
function warn(message) {
  const target = panel?.querySelector('[data-qq-error]');
  if (target) target.textContent = message || '';
}
function busy(button, active) {
  if (button) button.disabled = active;
}
function step(number) {
  panel.querySelectorAll('[data-qq-step]').forEach(node => {
    node.hidden = Number(node.dataset.qqStep) !== number;
  });
  panel.querySelector('[data-qq-progress]').textContent = '步骤 ' + (number + 1) + ' / 5';
  warn('');
}
function get(selector) { return panel.querySelector(selector); }
function showStatus(target, value) {
  const node = get(target);
  if (node) node.textContent = value;
}
function sourceMode() {
  // Credential entry is identical for both choices. The backend owns storage.
  get('[data-qq-secret-field]').hidden = false;
}
async function initialStatus() {
  try {
    const state = await api('/qq/setup/status');
    knownQQChannels = new Map((state.channels || []).map(ch => [String(ch.app_id), ch]));
    const select = get('[data-qq-existing]');
    select.innerHTML = '';
    for (const ch of (state.channels || [])) {
      const opt = document.createElement('option');
      opt.value = ch.channel_id;
      opt.dataset.credentialReady = String(ch.credential_ready);
      opt.dataset.credentialMode = ch.credential_mode || '';
      opt.dataset.appId = ch.app_id;
      opt.dataset.ownerBound = String(ch.owner_bound);
      opt.dataset.connected = String(ch.status === 'connected');
      opt.textContent = 'QQ AppID ' + ch.app_id + ' · ' +
        (ch.credential_ready ? (ch.status || '已配置') : '需重新提供凭证');
      select.appendChild(opt);
    }
    get('[data-qq-existing-block]').hidden = !select.options.length;
    showStatus('[data-qq-precheck]', select.options.length
      ? '检测到已有 QQ 机器人。若已断线，请优先恢复原渠道，不需要重新添加。' : '');
  } catch (err) {
    showStatus('[data-qq-precheck]', '无法读取现有配置：' + err.message);
  }
}
async function connect() {
  const button = get('[data-qq-connect]');
  if (button.disabled) return;
  const password = get('[data-qq-secret]');
  const appId = get('[data-qq-appid]').value.trim();
  const source = get('[name="qq-credential-mode"]:checked')?.value || 'persistent';

  // Validate before any network request or channel creation. Never send a
  // supplied secret as an environment-variable name.
  if (!/^[0-9]{5,32}$/.test(appId)) {
    warn('请输入 QQ 开放平台提供的数字 AppID（5–32 位）。');
    return;
  }
  if (password.value.trim().length < 8 || password.value.trim().length > 512) {
    warn('请输入 QQ 平台提供的 AppSecret（8–512 字符）。');
    return;
  }

  const existing = knownQQChannels.get(appId);
  if (existing && !channelId) {
    channelId = existing.channel_id;
    refreshMode = true;
  }

  busy(button, true);
  warn('');
  try {
    const result = refreshMode
      ? await api('/qq/manage/' + encodeURIComponent(channelId) + '/credential', {
          method: 'POST',
          body: JSON.stringify({ app_secret:password.value, restore_basic:true,
                                 persist:source === 'persistent' }),
        })
      : await post('/qq/guided-setup', {
          app_id:appId, source,
          app_secret:password.value,
          receive_images:true,
        });
    channelId = result.channel_id || channelId;
    ownerAlreadyBound = !!knownQQChannels.get(appId)?.owner_bound;
    const nextButton = get('[data-qq-to-pair]');
    nextButton.textContent = ownerAlreadyBound
      ? '下一步：默认权限（管理员已绑定）' : '下一步：绑定管理员';
    get('[data-qq-to-pair]').hidden = !result.connected;
    showStatus('[data-qq-connection-result]', result.connected
      ? 'QQ Gateway 已连接。'
      : '已记录渠道，但尚未连通。请核对凭证、腾讯平台授权与网络连接。');
    showStatus('[data-qq-flow-result]', result.default_flow?.ready
      ? '基础 QQ 消息工作流已配置。'
      : '已保留现有 Canvas；如未配置消息卡片，请在画布中绑定 QQ Request 和 Reply。');
    if (!result.connected) {
      refreshMode = true;
      knownQQChannels.set(appId, {channel_id:channelId, app_id:appId});
      button.textContent = '重新连接';
      warn(result.detail || '连接未成功，请检查 AppID、AppSecret 和腾讯平台权限；重试不会重复创建机器人。');
      return;
    }
    refreshMode = false;
    onReady();
  } catch (err) {
    if (err.status === 409 && err.details?.channel_id) {
      channelId = err.details.channel_id;
      refreshMode = true;
      warn('这个 QQ 机器人已经存在。输入 AppSecret 可恢复原渠道，不会重新创建。');
    } else {
      warn(err.message);
    }
  } finally {
    password.value = '';
    busy(button, false);
  }
}
async function startPair() {
  const button = get('[data-qq-pair-start]');
  busy(button, true);
  try {
    const result = await post('/qq/pair/start', { channel_id: channelId });
    showStatus('[data-qq-pair-code]', '绑定 ' + result.code);
    showStatus('[data-qq-pair-status]', '验证码 5 分钟内有效。请在 QQ 私聊机器人发送上面的文字。');
    get('[data-qq-pair-check]').hidden = false;
  } catch (err) { warn(err.message); }
  finally { busy(button, false); }
}
async function checkPair() {
  try {
    const state = await api('/qq/pair/status?channel_id=' + encodeURIComponent(channelId));
    if (state.status === 'verified') {
      showStatus('[data-qq-pair-status]', '已收到验证消息，用户尾号 ' + state.user_suffix + '。请在此确认。');
      get('[data-qq-pair-confirm]').hidden = false;
    } else {
      showStatus('[data-qq-pair-status]', state.status === 'expired'
        ? '验证码已过期，请重新生成。' : '尚未收到对应 QQ 私聊，请检查机器人的连接状态。');
    }
  } catch (err) { warn(err.message); }
}
async function confirmPair() {
  const button = get('[data-qq-pair-confirm]');
  busy(button, true);
  try {
    await post('/qq/pair/confirm', { channel_id: channelId });
    showStatus('[data-qq-pair-status]', '管理员已绑定。');
    step(3);
    onReady();
  } catch (err) { warn(err.message); }
  finally { busy(button, false); }
}
async function savePolicy() {
  const button = get('[data-qq-save-policy]');
  busy(button, true);
  try {
    await api('/' + encodeURIComponent(channelId) + '/access-policy', {
      method: 'PUT',
      body: JSON.stringify({
        default_permission: 'chat',
      }),
    });
    step(4);
    await diagnostics();
  } catch (err) { warn(err.message); }
  finally { busy(button, false); }
}
async function diagnostics() {
  try {
    const d = await api('/qq/setup/diagnostics?channel_id=' + encodeURIComponent(channelId));
    const lines = [
      'Gateway: ' + d.gateway,
      '凭证引用: ' + (d.credential_ready ? '可用' : '不可用（临时凭证需重新输入）'),
      'Canvas 消息输入: ' + (d.flow_input ? '已配置' : '未配置'),
      'Canvas 回复输出: ' + (d.flow_output ? '已配置' : '未配置'),
      'QQ 管理员: ' + (d.owner_bound ? '已绑定' : '未绑定'),
      '真实验收：请使用手机 QQ 发送一条消息，核对是否实际收到回复。',
    ];
    showStatus('[data-qq-diagnostics]', lines.join('\n'));
  } catch (err) { warn(err.message); }
}

export function openQQWizard({ onConnected, onOtherPlatform } = {}) {
  if (document.getElementById('qq-guided-setup')) return;
  onReady = onConnected || (() => {});
  otherPlatform = onOtherPlatform || (() => {});
  channelId = '';
  refreshMode = false;
  ownerAlreadyBound = false;
  knownQQChannels = new Map();
  const host = document.getElementById('channel-list');
  if (!host) return;
  // Keep the official channel listing intact, but hide it while the QQ wizard
  // is active. Showing its empty-state under step 2 was redundant and confusing.
  const addButton = document.getElementById('channel-add-btn');
  host.hidden = true;
  if (addButton) addButton.hidden = true;
  const closeWizard = () => {
    panel?.remove();
    panel = null;
    host.hidden = false;
    if (addButton) addButton.hidden = false;
  };
  panel = document.createElement('section');
  panel.id = 'qq-guided-setup';
  panel.className = 'qq-guided-setup';
  panel.innerHTML = [
    '<header class="qq-wizard-head"><strong>添加通讯渠道 · QQ</strong><span data-qq-progress>步骤 1 / 5</span><button type="button" class="btn-ghost btn-sm" data-qq-close>关闭</button></header>',
    '<p class="qq-wizard-error" role="alert" data-qq-error></p>',
    '<div data-qq-step="0">',
    '<h3>① 获取 QQ 官方机器人</h3>',
    '<p>在腾讯 QQ 开放平台创建机器人，按官方要求完成开发者验证，获得 AppID 和 AppSecret。</p>',
    '<p><a href="https://q.qq.com/" target="_blank" rel="noopener noreferrer">打开 QQ 开放平台</a> · <a href="https://github.com/tencent-connect/bot-docs" target="_blank" rel="noopener noreferrer">查看官方接入文档</a></p>',
    '<div data-qq-existing-block hidden><label>已有 QQ 渠道</label><select data-qq-existing></select><button class="btn-ghost btn-sm" type="button" data-qq-use-existing>管理已配置机器人</button></div>',
    '<p class="channel-users-hint" data-qq-precheck></p>',
    '<div class="channel-form-actions"><button class="btn-ghost" data-qq-other>返回渠道选择</button><button class="btn-primary" data-qq-next>我已获取机器人凭证</button></div>',
    '</div>',
    '<div data-qq-step="1" hidden>',
    '<h3>② 连接 QQ Gateway</h3>',
    '<label>AppID <input data-qq-appid inputmode="numeric" autocomplete="off" placeholder="QQ 开放平台提供的 AppID" /></label>',
    '<label data-qq-secret-field>AppSecret <input data-qq-secret type="password" autocomplete="new-password" spellcheck="false" placeholder="QQ 开放平台提供的 AppSecret" /></label>',
    '<fieldset><legend>连接保持方式</legend>',
    '<label><input type="radio" name="qq-credential-mode" value="persistent" checked /> 保存并自动连接（推荐，重启无需重填）</label>',
    '<label><input type="radio" name="qq-credential-mode" value="runtime" /> 仅本次使用（进程重启后需要重新输入）</label>',
    '</fieldset>',
    '<p class="channel-users-hint">长期凭证由 Agent Core 在独立受保护目录保管，不写入 Channel 数据库或浏览器存储。</p>',
    '<p data-qq-connection-result></p><p data-qq-flow-result></p>',
    '<div class="channel-form-actions"><button class="btn-primary" data-qq-connect>保存连接并检测</button><button class="btn-ghost" data-qq-to-pair hidden>下一步：绑定管理员</button></div>',
    '</div>',
    '<div data-qq-step="2" hidden>',
    '<h3>③ 绑定管理员 QQ</h3>',
    '<p>使用你自己的 QQ 私聊刚创建的机器人。不要将管理员验证码发送到 QQ 群。</p>',
    '<button class="btn-primary" data-qq-pair-start>生成一次性绑定码</button>',
    '<p class="qq-pair-code" data-qq-pair-code></p><p data-qq-pair-status></p>',
    '<div class="channel-form-actions"><button class="btn-ghost" data-qq-pair-check hidden>检查验证状态</button><button class="btn-primary" data-qq-pair-confirm hidden>确认绑定此 QQ 为管理员</button></div>',
    '</div>',
    '<div data-qq-step="3" hidden>',
    '<h3>④ 确认 QQ 用户权限</h3>',
    '<p class="channel-users-hint">普通私聊用户和 QQ 群成员固定仅聊天，无需审批；已绑定的管理员默认获得全部 Agent Core 工具。</p>',
    '<p class="channel-users-hint">以后可以在 QQ 渠道管理中查看如何邀请其他用户添加机器人。</p>',
    '<button class="btn-primary" data-qq-save-policy>确认并检查</button>',
    '</div>',
    '<div data-qq-step="4" hidden>',
    '<h3>⑤ 验收与完成</h3>',
    '<pre class="qq-wizard-diagnostics" data-qq-diagnostics></pre>',
    '<div class="channel-form-actions"><button class="btn-ghost" data-qq-recheck>刷新检查</button><button class="btn-ghost" data-qq-start-project>启动默认对话流程</button><button class="btn-primary" data-qq-finish>完成</button></div>',
    '</div>',
  ].join('');
  host.parentNode.insertBefore(panel, host);
  get('[data-qq-close]').onclick = closeWizard;
  get('[data-qq-next]').onclick = () => step(1);
  get('[data-qq-other]').onclick = () => { closeWizard(); otherPlatform(); };
  get('[data-qq-use-existing]').onclick = () => {
    const selected = get('[data-qq-existing]').selectedOptions[0];
    channelId = selected?.value || '';
    if (!channelId) return;
    ownerAlreadyBound = selected.dataset.ownerBound === 'true';
    if (selected.dataset.connected !== 'true') {
      refreshMode = true;
      step(1);
      get('[data-qq-appid]').value = selected.dataset.appId;
      get('[data-qq-appid]').disabled = true;
      get('[name="qq-credential-mode"][value="persistent"]').checked = true;
      sourceMode();
      warn('检测到原有 QQ 渠道断线。重新输入 AppSecret 即可恢复，不会重复创建机器人或重新绑定管理员。');
      return;
    }
    step(ownerAlreadyBound ? 3 : 2);
  };
  panel.querySelectorAll('[name="qq-credential-mode"]').forEach(el => el.onchange = sourceMode);
  get('[data-qq-connect]').onclick = connect;
  get('[data-qq-to-pair]').onclick = () => step(ownerAlreadyBound ? 3 : 2);
  get('[data-qq-pair-start]').onclick = startPair;
  get('[data-qq-pair-check]').onclick = checkPair;
  get('[data-qq-pair-confirm]').onclick = confirmPair;
  get('[data-qq-save-policy]').onclick = savePolicy;
  get('[data-qq-recheck]').onclick = diagnostics;
  get('[data-qq-start-project]').onclick = async () => {
    const btn = get('[data-qq-start-project]');
    busy(btn, true);
    try {
      const response = await fetch('/api/config/start-project', {method:'POST'});
      const result = await response.json().catch(() => ({}));
      if (!response.ok || result.ok === false) throw Error(result.detail || '启动失败');
      warn('');
      showStatus('[data-qq-diagnostics]', get('[data-qq-diagnostics]').textContent + '\\nAgent Core 项目已启动，请从 QQ 发送测试消息。');
    } catch(error) { warn('启动失败：' + error.message + '。请检查 LLM 配置和 Canvas。'); }
    finally { busy(btn, false); }
  };
  get('[data-qq-finish]').onclick = () => {
    closeWizard();
    onReady();
  };
  initialStatus();
}
