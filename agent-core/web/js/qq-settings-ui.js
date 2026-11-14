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
const CATEGORIES = [
  ['speech', 'TTS 与语音'],
  ['sensing', '传感器和感知'],
  ['motion', '机器人运动'],
  ['system', 'Shell 和文件操作'],
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
function elRadio(value, title, name, checked = false) {
  const label = txt('label', '', 'qq-settings-checkbox');
  const input = document.createElement('input');
  input.type = 'radio';
  input.name = name;
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
    ['connection', '连接状态'], ['owner', '管理员权限'],
    ['private', '私聊用户'], ['group', '群聊消息'],
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

  const owner = txt('section', '', 'qq-settings-section qq-owner-settings');
  owner.dataset.qqOwnerSection = '';
  owner.appendChild(txt('strong', '管理员'));
  const ownerHint = txt('p', '', 'qq-owner-status');
  owner.appendChild(ownerHint);
  const ownerEditor = document.createElement('details');
  ownerEditor.dataset.qqOwnerEditor = '';
  ownerEditor.className = 'qq-settings-subdetails';
  ownerEditor.appendChild(txt('summary', '编辑管理员权限'));
  const presets = txt('div', '', 'qq-settings-options');
  for (const [id, label] of [
    ['full', '全部 Agent Core 工具（管理员绑定后默认最高权限）'],
    ['daily', '日常使用'],
    ['chat', '仅聊天'],
    ['custom', '自定义工具类别'],
  ]) presets.appendChild(elRadio(id, label, 'qq-owner-permission'));
  const categories = txt('div', '', 'qq-settings-options');
  categories.dataset.qqOwnerCategories = '';
  for (const [id, label] of CATEGORIES) categories.appendChild(elCheckbox(id, label));
  categories.hidden = true;
  const ownerSave = txt('button', '保存管理员权限', 'btn-ghost btn-sm');
  ownerSave.type = 'button';
  ownerEditor.append(presets, categories, ownerSave);
  owner.appendChild(ownerEditor);
  const rebind = host.querySelector('[data-qq-rebind]');
  if (rebind) {
    rebind.classList.add('qq-settings-subdetails');
    owner.appendChild(rebind);
  }
  const recovery = txt('section', '', 'qq-settings-section');
  recovery.hidden = true;
  const recoveryField = document.createElement('input');
  recoveryField.type = 'password';
  recoveryField.autocomplete = 'new-password';
  recoveryField.placeholder = '请输入机器人连接密钥';
  const recoverButton = txt('button', '恢复私聊连接', 'btn-ghost btn-sm');
  recoverButton.type = 'button';
  const persistOption = elCheckbox('persist', '保存并自动重连（推荐）', true);
  recovery.append(txt('strong', '修复 QQ 连接'),
    txt('p', '重新输入 AppSecret 即可恢复，默认安全保存以供重启自动连接。', 'channel-users-hint'),
    recoveryField, persistOption, recoverButton,
    txt('p', '恢复时先使用私聊模式；群聊按平台授权重新开启。', 'channel-users-hint'));
  panel.append(overview, state, message, recovery, owner);
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
    for (const button of [advancedSave, ownerSave, recoverButton]) button.disabled = value;
  };
  function display(data) {
    snapshot = data;
    overviewFields.connection.textContent = data.connected ? '已连接' : '未连接';
    overviewFields.connection.classList.toggle('qq-value-success', !!data.connected);
    overviewFields.owner.textContent = !data.owner?.bound ? '未绑定' :
      data.owner.filter === '*' ? '全部工具' : '自定义权限';
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
    const filter = data.owner?.filter || '';
    const preset = !data.owner?.bound ? null
      : filter === '*' ? 'full'
      : filter === 'mcp__channel__channel_reply,finish' ? 'chat'
      : filter.includes('mcp__*__tts') && !filter.includes('mcp__*__loco') && !filter.includes('Bash') ? 'daily'
      : 'custom';
    presets.querySelectorAll('input').forEach(input => input.checked = input.value === preset);
    categories.hidden = preset !== 'custom';
    ownerHint.textContent = data.owner?.bound
      ? '已绑定 · ' + data.owner.suffix + ' · ' +
        ({full:'全部工具',daily:'日常使用',chat:'仅聊天',custom:'自定义工具'}[preset] || '当前权限')
      : '未绑定 · 请先完成身份验证';
    ownerSave.disabled = !data.owner?.bound;
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
  presets.addEventListener('change', () => {
    categories.hidden = presets.querySelector('input:checked')?.value !== 'custom';
  });
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
      button.textContent = button === advancedSave ? '保存高级配置并验证连接'
        : button === recoverButton ? '恢复私聊连接' : '保存管理员权限';
      setBusy(false);
      ownerSave.disabled = !snapshot?.owner?.bound;
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
  ownerSave.addEventListener('click', () => save(ownerSave, () => {
    const preset = presets.querySelector('input:checked')?.value;
    if (!preset) throw Error('请选择管理员权限预设');
    const selected = [...categories.querySelectorAll('input:checked')].map(x => x.value);
    return jsonRequest(API + '/' + encodeURIComponent(channelId) + '/owner', {
      method:'PUT', body:{preset, capabilities:preset === 'custom' ? selected : []},
    });
  }, '管理员权限已更新'));

  recoverButton.addEventListener('click', () => {
    const credential = recoveryField.value;
    recoveryField.value = '';
    if (!credential) { say('请先输入 QQ AppSecret。'); return; }
    return save(recoverButton, () => jsonRequest(
      API + '/' + encodeURIComponent(channelId) + '/credential',
      {method:'POST', body:{app_secret:credential,restore_basic:true,
                            persist:persistOption.querySelector('input').checked}},
    ), 'QQ 私聊连接已恢复。现在可按平台授权启用群聊。');
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
