/**
 * channels.js — Channel management panel (Telegram/Slack configuration + user management).
 */

let _overlay, _channelList, _addBtn;

export function initChannels() {
  _overlay = document.getElementById('channel-overlay');
  if (!_overlay) return;

  _channelList = document.getElementById('channel-list');
  _addBtn = document.getElementById('channel-add-btn');

  document.getElementById('btn-channels').addEventListener('click', _open);
  document.getElementById('channel-close').addEventListener('click', _close);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) _close(); });
  _addBtn.addEventListener('click', _showAddForm);
}

function _open() {
  _overlay.classList.remove('hidden');
  _loadChannels();
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
  if (!channels.length) {
    _channelList.innerHTML = `
      <div class="channel-empty">
        <p>No channels configured</p>
        <p style="font-size:12px;color:var(--text-dim);margin-top:4px">Add a Telegram or Slack bot to enable remote messaging control</p>
      </div>`;
    return;
  }

  _channelList.innerHTML = channels.map(ch => `
    <div class="channel-item" data-id="${_esc(ch.id)}">
      <div class="channel-item-header">
        <span class="channel-item-icon">${_platformIcon(ch.platform)}</span>
        <span class="channel-item-name">${_esc(ch.id)}</span>
        <span class="channel-item-platform">${_esc(ch.platform)}</span>
        <span class="channel-item-status ${ch.status === 'connected' ? 'online' : 'offline'}"
              title="${_esc(ch.health_error || '')}">${_esc(ch.status)}</span>
      </div>
      ${ch.health_error ? `<div class="channel-item-error">${_esc(ch.health_error)}</div>` : ''}
      <div class="channel-item-actions">
        <button class="btn-ghost btn-sm" onclick="window._channelStop('${_esc(ch.id)}')">Stop</button>
        <button class="btn-ghost btn-sm" onclick="window._channelRestart('${_esc(ch.id)}')">Restart</button>
        <button class="btn-ghost btn-sm btn-danger" onclick="window._channelDelete('${_esc(ch.id)}')">Delete</button>
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
      body: JSON.stringify({ id, platform, config, enabled }),
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

window._channelStop = async function(id) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Stopping…';
  try {
    const res = await fetch(`/api/channel/${id}/stop`, { method: 'POST' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Stop failed (${res.status})`);
    }
  } catch (e) {
    alert('Stop failed: ' + e.message);
  } finally {
    setTimeout(_loadChannels, 300);
  }
};

window._channelRestart = async function(id) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Restarting…';
  try {
    const res = await fetch(`/api/channel/${id}/restart`, { method: 'POST' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Restart failed (${res.status})`);
    }
  } catch (e) {
    alert('Restart failed: ' + e.message);
  } finally {
    setTimeout(_loadChannels, 500);
  }
};

window._channelDelete = async function(id) {
  if (!confirm(`Delete channel "${id}"?`)) return;
  try {
    const res = await fetch(`/api/channel/${id}`, { method: 'DELETE' });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      alert(j.detail || `Delete failed (${res.status})`);
    }
    _loadChannels();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
};

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
