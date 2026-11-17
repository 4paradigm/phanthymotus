/** QQ-specific settings surface. Does not replace the original Channels UI.
 * All editable SDK controls are scoped to a QQ channel and require Dashboard auth.
 */
const API = '/api/channel/qq/manage';
const SCOPES = [
  ['c2c', '私聊消息'],
  ['group', '群内 @机器人（需腾讯群消息权限）'],
  ['guild', 'QQ 频道 @机器人（需平台授权）'],
  ['guild_dm', 'QQ 频道私信（需平台授权）'],
];
const FLAGS = [
  ['enable_inbound_media', '接收图片和附件'],
  ['enable_outbound_media', '发送图片和附件'],
  ['enable_typing', '输入状态提示'],
  ['enable_session_resume', '断线恢复 Gateway 会话'],
  ['enable_markdown', 'Markdown（需平台审批）'],
  ['enable_voice_transcription', '语音转写（需 STT 后端）'],
  ['enable_guild_plain_messages', '频道普通消息（需平台权限）'],
  ['enable_interactions', '交互按钮事件'],
  ['enable_approvals', '可信交互审批'],
];
const txt = (tag, content, cls = '') => {
  const el = document.createElement(tag);
  el.textContent = content;
  if (cls) el.className = cls;
  return el;
};
function elCheckbox(value, title, checked = false) {
  const label = txt('label', '', 'qq-settings-checkbox');
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.value = value;
  input.checked = checked;
  label.append(input, document.createTextNode(' ' + title));
  return label;
}
async function jsonRequest(url, {method = 'GET', body, timeoutMs = 105000} = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method, signal: controller.signal, headers: {'Content-Type':'application/json'},
      ...(body ? {body: JSON.stringify(body)} : {}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail || data.message || ('HTTP ' + response.status);
      throw Error(typeof detail === 'string' ? detail : (detail.message || JSON.stringify(detail)));
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

export function initQQSettingsUI() {
  const host = document.getElementById('qq-management');
  if (!host || host.querySelector('#qq-config-settings')) return;
  const panel = document.createElement('section');
  panel.id = 'qq-config-settings';
  panel.className = 'qq-config-settings';
  const overview = txt('div', '', 'qq-overview');
  overview.dataset.qqOverview = '';
  const overviewFields = {};
  for (const [key, label] of [
    ['connection', '连接状态'], ['private', '私聊用户'],
    ['group', '群聊消息'],
  ]) {
    const cell = txt('div', '', 'qq-overview-item');
    cell.appendChild(txt('span', label, 'qq-overview-label'));
    const value = txt('strong', '读取中…', 'qq-overview-value');
    value.dataset.qqOverviewValue = key;
    overviewFields[key] = value;
    cell.appendChild(value);
    overview.appendChild(cell);
  }
  const state = txt('p', '', 'channel-users-hint qq-credential-status');
  state.dataset.qqSettingsState = '';
  const message = txt('p', '', 'channel-users-hint');
  message.setAttribute('role', 'status');
  const advanced = document.createElement('details');
  advanced.dataset.qqAdvanced = '';
  advanced.className = 'qq-settings-section qq-advanced-settings';
  advanced.appendChild(txt('summary', '高级设置 · QQ SDK 事件与能力'));
  advanced.appendChild(txt('p', '只有已获得腾讯授权的事件才应开启；修改 Gateway 订阅会触发重连，失败时回滚旧配置。', 'channel-users-hint'));
  const scopes = txt('div', '', 'qq-settings-options');
  for (const [id, label] of SCOPES) scopes.appendChild(elCheckbox(id, label));
  const flags = txt('div', '', 'qq-settings-options');
  for (const [id, label] of FLAGS) flags.appendChild(elCheckbox(id, label));
  const advancedSave = txt('button', '保存高级配置并验证连接', 'btn-ghost btn-sm');
  advancedSave.type = 'button';
  advanced.append(txt('strong', '事件订阅'), scopes, txt('strong', 'SDK 能力'), flags, advancedSave);

  const recovery = txt('section', '', 'qq-settings-section');
  recovery.hidden = true;
  const recoveryField = document.createElement('input');
  recoveryField.type = 'password';
  recoveryField.autocomplete = 'new-password';
  recoveryField.placeholder = '请输入机器人连接密钥';
  const recoverButton = txt('button', '恢复 QQ 连接', 'btn-ghost btn-sm');
  recoverButton.type = 'button';
  const persistOption = elCheckbox('persist', '保存并自动重连（推荐）', true);
  recovery.append(txt('strong', '修复 QQ 连接'),
    txt('p', '重新输入 AppSecret 即可恢复，默认安全保存以供重启自动连接。', 'channel-users-hint'),
    recoveryField, persistOption, recoverButton,
    txt('p', '恢复凭证会保留已启用的私聊、群聊和媒体能力，无需再次授权群聊。', 'channel-users-hint'));
  panel.append(overview, state, message, recovery);
  const coreSlot = host.querySelector('[data-qq-core-settings-slot]');
  if (coreSlot) coreSlot.appendChild(panel);
  else host.prepend(panel);
  const advancedSlot = host.querySelector('[data-qq-advanced-slot]');
  if (advancedSlot) advancedSlot.appendChild(advanced);
  else panel.appendChild(advanced);

  let channelId = '';
  let snapshot = null;
  let busy = false;
  const say = value => { message.textContent = value; };
  const setBusy = value => {
    busy = !!value;
    for (const button of [advancedSave, recoverButton]) button.disabled = value;
  };
  function display(data) {
    snapshot = data;
    overviewFields.connection.textContent = data.connected ? '已连接' : '未连接';
    overviewFields.connection.classList.toggle('qq-value-success', !!data.connected);
    overviewFields.private.textContent = '仅聊天';
    overviewFields.group.textContent = data.enabled_scopes.includes('group') ? '已订阅' : '未订阅';
    state.textContent = '连接凭证：' +
      ({runtime:'仅本次使用',persistent:'已安全保存 · 自动重连',env:'环境变量',mounted_file:'只读挂载',legacy:'历史配置'}[data.credential_mode] || data.credential_mode);
    recovery.hidden = data.connected && data.credential_ready !== false;
    scopes.querySelectorAll('input').forEach(input => {
      input.checked = data.enabled_scopes.includes(input.value);
    });
    flags.querySelectorAll('input').forEach(input => {
      input.checked = !!data.flags[input.value];
    });
  }
  async function load(id) {
    channelId = id;
    const data = await jsonRequest(API + '/' + encodeURIComponent(id) + '/settings');
    if (channelId === id) {
      display(data);
      document.dispatchEvent(new CustomEvent('qq-settings-updated', {
        detail: {channelId:id, groupEnabled:data.enabled_scopes.includes('group')},
      }));
      say('');
    }
  }
  async function save(button, task, okMessage) {
    if (busy || !channelId) return;
    setBusy(true);
    button.textContent = '保存并验证中…';
    try {
      const result = await task();
      await load(channelId);
      say(result.connected === false && result.changed
        ? okMessage + '（渠道已停止，启动后生效）'
        : okMessage);
    } catch (error) {
      const failure = '操作未完成：' + (error.name === 'AbortError'
        ? '连接超时，请重新读取状态' : error.message);
      try { await load(channelId); } catch (_) { /* preserve the actionable error */ }
      say(failure);
    } finally {
      button.textContent = button === advancedSave ? '保存高级配置并验证连接' : '恢复 QQ 连接';
      setBusy(false);
    }
  }
  advancedSave.addEventListener('click', () => save(advancedSave, () => {
    const selectedScopes = [...scopes.querySelectorAll('input:checked')].map(x => x.value);
    const selectedFlags = Object.fromEntries(
      [...flags.querySelectorAll('input')].map(x => [x.value, x.checked]));
    if (!selectedScopes.length) throw Error('至少保留一个事件范围');
    return jsonRequest(API + '/' + encodeURIComponent(channelId) + '/settings', {
      method:'PUT', body:{mode:'advanced', enabled_scopes:selectedScopes,flags:selectedFlags},
    });
  }, '高级配置已保存并验证'));
  recoverButton.addEventListener('click', () => {
    const credential = recoveryField.value;
    recoveryField.value = '';
    if (!credential) { say('请先输入 QQ AppSecret。'); return; }
    return save(recoverButton, () => jsonRequest(
      API + '/' + encodeURIComponent(channelId) + '/credential',
      {method:'POST', body:{app_secret:credential,restore_basic:false,
                            persist:persistOption.querySelector('input').checked}},
    ), 'QQ 连接已恢复，原有群聊设置保持不变。');
  });

  document.addEventListener('qq-management-refresh', event => {
    if (event.detail?.channelId === channelId) {
      load(channelId).catch(error => say('刷新状态失败：' + error.message));
    }
  });
  document.addEventListener('qq-channel-manage', event => {
    const id = event.detail?.channelId;
    if (!id) return;
    load(id).then(() => {
      advanced.open = event.detail?.tab === 'advanced';
      document.getElementById('channel-details-panel')?.scrollIntoView({block:'start'});
    }).catch(error => say('读取 QQ 配置失败：' + error.message));
  });
}

/** Redirect only the built-in QQ Configure button; other channels are untouched.
 * Capture before the original delegated click handler on channel-list.
 */
export function interceptLegacyQQConfigure() {
  const list = document.getElementById('channel-list');
  if (!list || list.dataset.qqConfigureEnhanced) return;
  list.dataset.qqConfigureEnhanced = 'true';
  const labelQQSettings = () => {
    for (const button of list.querySelectorAll('[data-channel-action="qq-edit"]')) {
      if (button.textContent !== '管理') button.textContent = '管理';
    }
  };
  new MutationObserver(labelQQSettings).observe(list, {childList:true, subtree:true});
  labelQQSettings();
  list.addEventListener('click', event => {
    const button = event.target.closest?.('[data-channel-action="qq-edit"]');
    const card = button?.closest('.channel-item');
    if (!button || !card) return;
    event.stopImmediatePropagation();
    event.preventDefault();
    const channelId = card.querySelector('.channel-item-name')?.textContent?.trim();
    if (channelId) document.dispatchEvent(new CustomEvent('qq-channel-manage', {
      detail: {channelId, tab:'standard'},
    }));
  }, true);
}
