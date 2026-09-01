/**
 * channels.js — Channel management panel (Telegram/Slack configuration + user management).
 */

let _overlay, _channelList, _addBtn;
let _channels = [];
let _usersList;

export function initChannels() {
  _overlay = document.getElementById('channel-overlay');
  if (!_overlay) return;

  _channelList = document.getElementById('channel-list');
  _addBtn = document.getElementById('channel-add-btn');
  _usersList = document.getElementById('channel-users-list');

  document.getElementById('btn-channels').addEventListener('click', _open);
  document.getElementById('channel-close').addEventListener('click', _close);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) _close(); });
  _addBtn.addEventListener('click', _showAddForm);
  _channelList.addEventListener('click', _handleChannelAction);
  _usersList.addEventListener('click', _handleUserAction);
  _usersList.addEventListener('change', _handleUserRoleChange);
  document.getElementById('channel-user-form-add').addEventListener('click', _submitUser);
}

function _open() {
  _overlay.classList.remove('hidden');
  _loadChannels();
  _loadUsers();
}

function _close() {
  _overlay.classList.add('hidden');
}

// ── Load channels ─────────────────────────────────────────────────────────────

async function _loadChannels() {
  try {
    const res = await fetch('/api/channel/list');
    const json = await res.json();
    const channels = json.channels || [];
    _renderChannels(channels);
  } catch (e) {
    _channelList.innerHTML = '<div class="channel-empty">Failed to load channels</div>';
  }
}

function _renderChannels(channels) {
  _channels = channels;
  if (!channels.length) {
    _channelList.innerHTML = `
      <div class="channel-empty">
        <p>No channels configured</p>
        <p style="font-size:12px;color:var(--text-dim);margin-top:4px">Add a Telegram, Slack, or Feishu bot to enable remote messaging control</p>
      </div>`;
    return;
  }

  _channelList.innerHTML = channels.map((ch, index) => `
    <div class="channel-item" data-channel-index="${index}">
      <div class="channel-item-header">
        <span class="channel-item-icon">${_platformIcon(ch.platform)}</span>
        <span class="channel-item-name">${_esc(ch.id)}</span>
        <span class="channel-item-platform">${_esc(ch.platform)}</span>
        <span class="channel-item-status ${ch.status === 'connected' ? 'online' : 'offline'}"
              title="${_escAttr(ch.health_error || '')}">${_esc(ch.status)}</span>
      </div>
      ${ch.health_error ? `<div class="channel-item-error">${_esc(ch.health_error)}</div>` : ''}
      <div class="channel-item-actions">
        ${ch.platform === 'feishu' ? `
          <button class="btn-ghost btn-sm" data-channel-action="bot-to-bot">
            Bot @ Bot: ${ch.bot_to_bot_enabled ? 'On' : 'Off'}
          </button>
          <button class="btn-ghost btn-sm" data-channel-action="trusted-bots">
            Trusted Bots: ${(ch.trusted_bots || []).length}
          </button>` : ''}
        <button class="btn-ghost btn-sm" data-channel-action="stop">Stop</button>
        <button class="btn-ghost btn-sm" data-channel-action="restart">Restart</button>
        <button class="btn-ghost btn-sm btn-danger" data-channel-action="delete">Delete</button>
      </div>
    </div>
  `).join('');
}

// ── Add channel form ──────────────────────────────────────────────────────────

function _showAddForm() {
  const formHtml = `
    <div class="channel-add-form" id="channel-add-form">
      <div class="channel-form-row">
        <label>Platform</label>
        <select id="channel-form-platform">
          <option value="telegram">Telegram</option>
          <option value="slack">Slack</option>
          <option value="feishu">Feishu (飞书)</option>
        </select>
      </div>
      <div class="channel-form-row">
        <label>ID</label>
        <input type="text" id="channel-form-id" placeholder="e.g. telegram_main" />
      </div>
      <div class="channel-form-row" id="channel-form-token-row">
        <label>Bot Token</label>
        <input type="password" id="channel-form-token" placeholder="Bot token" />
      </div>
      <div class="channel-form-row hidden" id="channel-form-app-token-row">
        <label>App Token</label>
        <input type="password" id="channel-form-app-token" placeholder="Slack App Token (xapp-...)" />
      </div>
      <div class="channel-form-row hidden" id="channel-form-app-id-row">
        <label>App ID</label>
        <input type="text" id="channel-form-app-id" placeholder="App ID" />
      </div>
      <div class="channel-form-row hidden" id="channel-form-app-secret-row">
        <label>App Secret</label>
        <input type="password" id="channel-form-app-secret" placeholder="App Secret" />
      </div>
      <div class="channel-form-row hidden" id="channel-form-bot-to-bot-row">
        <label title="Accept explicit @ messages from any bot in any group this app joins">
          <input type="checkbox" id="channel-form-bot-to-bot" /> Allow group Bot @ Bot
        </label>
      </div>
      <div class="channel-form-row">
        <label><input type="checkbox" id="channel-form-enabled" checked /> Enable immediately</label>
      </div>
      <div class="channel-form-actions">
        <button class="btn-primary" id="channel-form-submit">Add</button>
        <button class="btn-ghost" id="channel-form-cancel">Cancel</button>
      </div>
    </div>`;

  _channelList.insertAdjacentHTML('beforebegin', formHtml);
  const form = document.getElementById('channel-add-form');
  const platformSel = document.getElementById('channel-form-platform');

  function updateFormFields() {
    const p = platformSel.value;
    const isSlack = p === 'slack';
    const isFeishu = p === 'feishu';

    document.getElementById('channel-form-app-token-row').classList.toggle('hidden', !isSlack);
    document.getElementById('channel-form-app-id-row').classList.toggle('hidden', !isFeishu);
    document.getElementById('channel-form-app-secret-row').classList.toggle('hidden', !isFeishu);
    document.getElementById('channel-form-bot-to-bot-row').classList.toggle('hidden', !isFeishu);
    document.getElementById('channel-form-token-row').querySelector('label').textContent =
      isSlack ? 'Bot Token (xoxb-...)' : 'Bot Token';
    document.getElementById('channel-form-token-row').classList.toggle('hidden', isFeishu);
  }

  platformSel.addEventListener('change', updateFormFields);
  updateFormFields();

  document.getElementById('channel-form-submit').addEventListener('click', () => _submitAdd(form));
  document.getElementById('channel-form-cancel').addEventListener('click', () => form.remove());
}

async function _submitAdd(formEl) {
  const submitBtn = document.getElementById('channel-form-submit');
  if (submitBtn.disabled) return;  // 防止重复提交

  const platform = document.getElementById('channel-form-platform').value;
  const id = document.getElementById('channel-form-id').value.trim();
  const token = document.getElementById('channel-form-token').value.trim();
  const appToken = document.getElementById('channel-form-app-token')?.value.trim() || '';
  const appId = document.getElementById('channel-form-app-id')?.value.trim() || '';
  const appSecret = document.getElementById('channel-form-app-secret')?.value.trim() || '';
  const enabled = document.getElementById('channel-form-enabled').checked;
  const botToBotEnabled = platform === 'feishu'
    && document.getElementById('channel-form-bot-to-bot').checked;

  if (!id) { alert('ID is required'); return; }

  const config = {};

  if (platform === 'telegram') {
    if (!token) { alert('Bot Token is required'); return; }
    config.bot_token = token;
  } else if (platform === 'slack') {
    if (!token) { alert('Bot Token is required'); return; }
    config.bot_token = token;
    if (appToken) config.app_token = appToken;
  } else if (platform === 'feishu') {
    if (!appId || !appSecret) { alert('App ID and App Secret are required'); return; }
    config.app_id = appId;
    config.app_secret = appSecret;
  }

  // 进入 loading 状态
  submitBtn.disabled = true;
  submitBtn.classList.add('btn-loading');
  const originalText = submitBtn.textContent;
  submitBtn.textContent = '验证中…';

  try {
    const res = await fetch('/api/channel/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id,
        platform,
        config,
        enabled,
        bot_to_bot_enabled: botToBotEnabled,
      }),
    });
    const json = await res.json();
    if (!res.ok) {
      alert(json.detail || 'Failed to add channel');
      return;
    }
    formEl.remove();
    _loadChannels();
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('btn-loading');
    submitBtn.textContent = originalText;
  }
}

// ── Actions ───────────────────────────────────────────────────────────────────

function _handleChannelAction(e) {
  const btn = e.target.closest?.('button[data-channel-action]');
  if (!btn || !_channelList.contains(btn)) return;
  const item = btn.closest('.channel-item');
  const channel = _channels[Number(item?.dataset.channelIndex)];
  if (!channel) return;

  switch (btn.dataset.channelAction) {
    case 'bot-to-bot': _channelSetBotToBot(channel.id, !channel.bot_to_bot_enabled, btn); break;
    case 'trusted-bots': _showTrustedBots(channel, item); break;
    case 'stop':       _channelStop(channel.id, btn); break;
    case 'restart':    _channelRestart(channel.id, btn); break;
    case 'delete':     _channelDelete(channel.id); break;
  }
}

async function _channelSetBotToBot(id, enabled, btn) {
  if (enabled && !confirm(
    'Enable Bot @ Bot for this Feishu channel?\n\n'
    + 'Any bot in a shared group can explicitly @mention this Agent, but only Bots listed '
    + 'under Trusted Bots can activate Skills or invoke bound tools.'
  )) return;

  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const res = await fetch(`/api/channel/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot_to_bot_enabled: enabled }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Update failed (${res.status})`);
    }
  } catch (e) {
    alert('Update failed: ' + e.message);
  } finally {
    setTimeout(_loadChannels, 500);
  }
}

function _trustedBotRowHtml(bot = {}) {
  return `
    <div class="trusted-bot-card" data-trusted-bot-row>
      <button class="trusted-bot-remove" type="button" data-trusted-bot-remove title="Remove this Bot">×</button>
      <div class="trusted-bot-fields">
        <label class="tb-field">
          <span>Name</span>
          <input type="text" data-field="name" placeholder="Peer Bot" value="${_escAttr(bot.name || '')}" />
        </label>
        <label class="tb-field">
          <span>ID</span>
          <input type="text" data-field="id" placeholder="peer" value="${_escAttr(bot.id || '')}" />
        </label>
        <label class="tb-field tb-field-wide">
          <span>Group chat_id</span>
          <input type="text" class="mono" data-field="chat_id" placeholder="oc_..." value="${_escAttr(bot.chat_id || '')}" />
        </label>
        <label class="tb-field tb-field-wide">
          <span>Bot open_id</span>
          <input type="text" class="mono" data-field="open_id" placeholder="ou_..." value="${_escAttr(bot.open_id || '')}" />
        </label>
      </div>
    </div>`;
}

function _showTrustedBots(channel, item) {
  document.querySelectorAll('.channel-trusted-form').forEach((el) => el.remove());
  const bots = channel.trusted_bots || [];
  item.insertAdjacentHTML('beforeend', `
    <div class="channel-add-form channel-trusted-form">
      <div class="channel-form-row">
        <label>Trusted Bots</label>
        <div data-trusted-bots-rows>${bots.map(_trustedBotRowHtml).join('') || _trustedBotRowHtml()}</div>
        <button class="btn-ghost btn-sm" type="button" data-trusted-bots-add>+ Add Bot</button>
      </div>
      <div class="channel-item-error">Trusted Bots have full access to Skills and every Canvas-bound tool. Configure only exact Bot open_id + group chat_id pairs — find them in the group's Activity log after the peer bot first @s you, or in Feishu's own bot info API.</div>
      <div class="channel-form-actions">
        <button class="btn-primary" data-trusted-bots-save>Save</button>
        <button class="btn-ghost" data-trusted-bots-cancel>Cancel</button>
      </div>
    </div>`);
  const form = item.querySelector('.channel-trusted-form');
  const rows = form.querySelector('[data-trusted-bots-rows]');
  rows.addEventListener('click', (e) => {
    if (e.target.closest('[data-trusted-bot-remove]')) {
      e.target.closest('[data-trusted-bot-row]').remove();
    }
  });
  form.querySelector('[data-trusted-bots-add]').addEventListener('click', () => {
    rows.insertAdjacentHTML('beforeend', _trustedBotRowHtml());
    rows.lastElementChild.querySelector('[data-field="name"]').focus();
  });
  form.querySelector('[data-trusted-bots-save]').addEventListener('click', (e) => {
    _saveTrustedBots(channel.id, form, e.currentTarget);
  });
  form.querySelector('[data-trusted-bots-cancel]').addEventListener('click', () => form.remove());
}

function _readTrustedBotRows(form) {
  const rows = [...form.querySelectorAll('[data-trusted-bot-row]')];
  const bots = [];
  for (const row of rows) {
    const get = (field) => row.querySelector(`[data-field="${field}"]`).value.trim();
    const bot = { name: get('name'), id: get('id'), chat_id: get('chat_id'), open_id: get('open_id') };
    // 空行（用户加了但没填）直接跳过，不当成一条无效配置去校验
    if (!bot.id && !bot.chat_id && !bot.open_id && !bot.name) continue;
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(bot.id)) {
      throw new Error(`"${bot.id || '(empty)'}" 不是合法的 ID（1-64 位字母/数字/_/-，且首字符不能是 - 或 _）`);
    }
    if (!/^oc_[A-Za-z0-9]+$/.test(bot.chat_id)) {
      throw new Error(`Bot "${bot.id}" 的 chat_id 必须是飞书 oc_... 格式的群 ID`);
    }
    if (!/^ou_[A-Za-z0-9]+$/.test(bot.open_id)) {
      throw new Error(`Bot "${bot.id}" 的 open_id 必须是飞书 ou_... 格式的机器人 ID`);
    }
    bots.push(bot);
  }
  return bots;
}

async function _saveTrustedBots(id, form, btn) {
  let trustedBots;
  try {
    trustedBots = _readTrustedBotRows(form);
  } catch (e) {
    alert(e.message);
    return;
  }
  if (trustedBots.length && !confirm(
    'Trust these Bots with full Agent execution access?\n\nOnly exact open_id + group chat_id pairs will match.'
  )) return;

  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const res = await fetch(`/api/channel/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trusted_bots: trustedBots }),
    });
    if (!res.ok) {
      const json = await res.json().catch(() => ({}));
      throw new Error(json.detail || `Update failed (${res.status})`);
    }
    _loadChannels();
  } catch (e) {
    alert('Update failed: ' + e.message);
    btn.disabled = false;
    btn.textContent = 'Save';
  }
}

async function _channelStop(id, btn) {
  btn.disabled = true;
  btn.textContent = 'Stopping…';
  try {
    const res = await fetch(`/api/channel/${encodeURIComponent(id)}/stop`, { method: 'POST' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Stop failed (${res.status})`);
    }
  } catch (e) {
    alert('Stop failed: ' + e.message);
  } finally {
    setTimeout(_loadChannels, 300);
  }
}

async function _channelRestart(id, btn) {
  btn.disabled = true;
  btn.textContent = 'Restarting…';
  try {
    const res = await fetch(`/api/channel/${encodeURIComponent(id)}/restart`, { method: 'POST' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Restart failed (${res.status})`);
    }
  } catch (e) {
    alert('Restart failed: ' + e.message);
  } finally {
    setTimeout(_loadChannels, 500);
  }
}

async function _channelDelete(id) {
  if (!confirm(`Delete channel "${id}"?`)) return;
  try {
    const res = await fetch(`/api/channel/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Delete failed (${res.status})`);
    }
    _loadChannels();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

// ── Users (ACL role management) ─────────────────────────────────────────────
//
// Channel messages get auto-registered with channel_settings.default_role
// (viewer by default). Viewer-triggered turns are now hard-restricted to
// sensor/resource tools + channel_reply/finish (see event/llm.py
// _viewer_channel_restricted) — this panel is the only place to promote a
// specific user to operator/owner, since the backend never did more than
// carry the role as text before that enforcement existed.

const _ROLE_ORDER = ['owner', 'operator', 'viewer', 'blocked'];

async function _loadUsers() {
  _usersList.innerHTML = '<div class="channel-empty">Loading...</div>';
  try {
    const res = await fetch('/api/channel/users');
    const json = await res.json();
    _renderUsers(json.users || []);
  } catch (e) {
    _usersList.innerHTML = '<div class="channel-empty">Failed to load users</div>';
  }
}

function _renderUsers(users) {
  if (!users.length) {
    _usersList.innerHTML = '<div class="channel-empty">No users yet — they auto-register on first message, or add one below.</div>';
    return;
  }
  _usersList.innerHTML = users.map((u) => {
    const hasName = u.display_name && u.display_name !== u.user_id;
    const identity = hasName
      ? `<span class="user-name">${_esc(u.display_name)}</span><span class="user-id">${_esc(u.user_id)}</span>`
      : `<span class="user-name mono">${_esc(u.user_id)}</span>`;
    return `
      <div class="user-row" data-user-platform="${_escAttr(u.platform)}" data-user-id="${_escAttr(u.user_id)}" data-user-name="${_escAttr(u.display_name || '')}">
        <span class="user-role-dot" data-role="${_escAttr(u.role)}" title="${_escAttr(u.role)}"></span>
        <span class="user-platform-badge">${_esc(u.platform)}</span>
        <span class="user-identity" title="${_escAttr(u.user_id)}">${identity}</span>
        <select class="user-role-select" data-user-role>
          ${_ROLE_ORDER.map((r) => `<option value="${r}" ${r === u.role ? 'selected' : ''}>${r[0].toUpperCase()}${r.slice(1)}</option>`).join('')}
        </select>
        <button class="user-remove" data-user-delete title="Remove user">×</button>
      </div>`;
  }).join('');
}

async function _handleUserRoleChange(e) {
  const select = e.target.closest('[data-user-role]');
  if (!select) return;
  const row = select.closest('[data-user-platform]');
  const platform = row.dataset.userPlatform;
  const userId = row.dataset.userId;
  const displayName = row.dataset.userName || '';
  select.disabled = true;
  try {
    const res = await fetch('/api/channel/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ platform, user_id: userId, display_name: displayName, role: select.value }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Update failed (${res.status})`);
      _loadUsers();
      return;
    }
  } catch (err) {
    alert('Update failed: ' + err.message);
    _loadUsers();
    return;
  } finally {
    select.disabled = false;
  }
}

async function _handleUserAction(e) {
  const btn = e.target.closest?.('[data-user-delete]');
  if (!btn) return;
  const row = btn.closest('[data-user-platform]');
  const platform = row.dataset.userPlatform;
  const userId = row.dataset.userId;
  if (!confirm(`Remove user "${userId}" (${platform})? They will auto-register again with the default role on their next message.`)) return;
  try {
    const res = await fetch('/api/channel/users', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ platform, user_id: userId }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Delete failed (${res.status})`);
    }
  } catch (err) {
    alert('Delete failed: ' + err.message);
  } finally {
    _loadUsers();
  }
}

async function _submitUser() {
  const platform = document.getElementById('channel-user-form-platform').value;
  const userId = document.getElementById('channel-user-form-id').value.trim();
  const displayName = document.getElementById('channel-user-form-name').value.trim();
  const role = document.getElementById('channel-user-form-role').value;
  if (!userId) { alert('User ID is required'); return; }

  const btn = document.getElementById('channel-user-form-add');
  btn.disabled = true;
  try {
    const res = await fetch('/api/channel/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ platform, user_id: userId, display_name: displayName, role }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Failed to add user (${res.status})`);
      return;
    }
    document.getElementById('channel-user-form-id').value = '';
    document.getElementById('channel-user-form-name').value = '';
    _loadUsers();
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _platformIcon(platform) {
  switch (platform) {
    case 'telegram': return '✈';
    case 'slack':    return '◆';
    case 'feishu':   return '飞';
    case 'whatsapp': return '◉';
    default:         return '◇';
  }
}

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function _escAttr(s) {
  return _esc(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
