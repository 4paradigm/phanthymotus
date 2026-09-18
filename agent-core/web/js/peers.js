/**
 * peers.js — Peer collaboration panel (discovery, SAS pairing, per-peer roles).
 *
 * Mirrors channels.js deliberately: both are "list of connections + role ACL",
 * so the markup classes (user-row / user-role-select / user-role-dot) and the
 * load/render split are reused rather than reinvented.
 *
 * Auth is handled globally — auth.js patches window.fetch to attach the token,
 * so plain fetch('/api/peer/...') works here.
 *
 * Four sections, matching what an operator actually needs to answer:
 *   1. Is peering on, and who am I?      → /settings + /identity
 *   2. Is anyone asking to pair?         → /pair/active   (approve / reject)
 *   3. Who am I paired with, and can     → /paired        (role, unpair)
 *      they do anything?
 *   4. Who else is out there?            → /discovered + /providers
 */

import { showToast } from './toast.js';

let _overlay, _body;
let _pollTimer = null;

// Peers can never be `owner` — that role implies user management, which a
// remote machine has no business holding. Kept in sync with peer/store.py.
const _ROLE_ORDER = ['operator', 'viewer', 'blocked'];

const _ROLE_HINT = {
  operator: '可请求执行器工具（仍受画布绑定与本机 LLM 决策约束）',
  viewer: '只读：传感器与状态查询',
  blocked: '拒绝一切请求',
};

export function initPeers() {
  _overlay = document.getElementById('peer-overlay');
  if (!_overlay) return;
  _body = document.getElementById('peer-body');

  const btn = document.getElementById('btn-peers');
  if (btn) btn.addEventListener('click', _open);
  document.getElementById('peer-close').addEventListener('click', _close);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) _close(); });

  _body.addEventListener('click', _onClick);
  _body.addEventListener('change', _onChange);
}

/** Called by motus-stream when a peer asks to pair. */
export function onPairRequest(payload) {
  const who = payload?.display_name || (payload?.peer_id || '').slice(0, 12);
  showToast(`${who} 请求配对 — 验证码 ${payload?.code || '??????'}`);
  if (_isOpen()) _refresh();
}

function _isOpen() { return _overlay && !_overlay.classList.contains('hidden'); }

function _open() {
  _overlay.classList.remove('hidden');
  _refresh();
  // Discovery changes as mDNS comes and goes, so poll — but only while the
  // panel is visible. A background poll would hammer the API for nothing.
  _pollTimer = setInterval(_refresh, 5000);
}

function _close() {
  _overlay.classList.add('hidden');
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

// ── Load ─────────────────────────────────────────────────────────────────────

async function _refresh() {
  try {
    const [settings, identity, pending, paired, discovered, providers] = await Promise.all([
      _get('/api/peer/settings'),
      _get('/api/peer/identity'),
      _get('/api/peer/pair/active'),
      _get('/api/peer/paired'),
      _get('/api/peer/discovered'),
      _get('/api/peer/providers'),
    ]);
    _render({ settings, identity, pending, paired, discovered, providers });
  } catch (e) {
    _body.innerHTML = '<div class="channel-empty">加载失败，请检查 Agent Core 是否在运行</div>';
  }
}

async function _get(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

// ── Render ───────────────────────────────────────────────────────────────────

function _render({ settings, identity, pending, paired, discovered, providers }) {
  const enabled = !!settings.enabled;
  const sessions = pending.sessions || [];
  const peers = paired.peers || [];
  const found = discovered.peers || [];

  // The panel repaints on a 5s poll, and this is a full innerHTML replacement, so
  // anything half-typed in the name field was destroyed along with the element —
  // the field looked uneditable because a keystroke survived at most five seconds.
  // Carry the in-progress value, caret and focus across the repaint.
  const nameEl = document.getElementById('peer-display-name');
  const editing = nameEl && document.activeElement === nameEl;
  const draft = nameEl ? nameEl.value : null;
  const caret = editing ? [nameEl.selectionStart, nameEl.selectionEnd] : null;

  _body.innerHTML = [
    _renderSettings(settings, identity, providers.providers || []),
    enabled ? _renderPending(sessions) : '',
    enabled ? _renderPaired(peers) : '',
    enabled ? _renderStatic(settings) : '',
    enabled ? _renderDiscovered(found) : '',
  ].join('');

  const fresh = document.getElementById('peer-display-name');
  if (fresh && draft !== null && draft !== fresh.value) fresh.value = draft;
  if (fresh && editing) {
    fresh.focus();
    if (caret) fresh.setSelectionRange(caret[0], caret[1]);
  }
}

function _renderSettings(s, identity, providers) {
  const enabled = !!s.enabled;
  const provLine = providers.length
    ? providers.map((p) => {
        const state = p.running ? 'ok' : 'err';
        const detail = p.error ? ` — ${_esc(p.error)}` : '';
        return `<span class="peer-prov peer-prov--${state}">${_esc(p.name)}${detail}</span>`;
      }).join('')
    : '<span class="peer-prov peer-prov--err">未启动任何发现方式</span>';

  return `
    <section class="peer-section">
      <div class="channel-users-header">
        <span class="channel-users-title">本机</span>
        <span class="channel-users-hint">开启后，同局域网的其他机器人才能发现这台机器</span>
      </div>
      <div class="peer-settings-row">
        <label class="peer-toggle">
          <input type="checkbox" id="peer-enabled" ${enabled ? 'checked' : ''}>
          <span>启用多机协同</span>
        </label>
        <label class="peer-toggle">
          <input type="checkbox" id="peer-mdns" ${s.discovery?.mdns ? 'checked' : ''} ${enabled ? '' : 'disabled'}>
          <span>局域网自动发现 (mDNS)</span>
        </label>
        <label class="peer-toggle" title="蓝牙只用来交换公钥，不承载数据；配对握手仍走 HTTPS，两台之间必须还能通 IP。需要主机上 rfkill 解锁并运行 bluetoothd。">
          <input type="checkbox" id="peer-ble" ${s.discovery?.ble ? 'checked' : ''} ${enabled ? '' : 'disabled'}>
          <span>蓝牙发现 (BLE)</span>
        </label>
      </div>
      <div class="peer-settings-row">
        <input class="peer-input" id="peer-display-name" placeholder="${_escAttr(s.resolved_display_name || '本机名称')}"
               value="${_escAttr(s.display_name || '')}" ${enabled ? '' : 'disabled'}>
        <button class="btn-primary btn-sm" data-peer-save ${enabled ? '' : 'disabled'}>保存</button>
      </div>
      <div class="peer-identity">
        <span class="peer-identity-label">本机指纹</span>
        <code class="peer-fingerprint">${_esc(identity.peer_id || '')}</code>
      </div>
      ${enabled ? `<div class="peer-providers">${provLine}</div>` : ''}
      ${enabled ? '' : '<div class="channel-empty">多机协同已关闭。勾选上方开关即可开启，无需重启。</div>'}
    </section>`;
}

function _renderPending(sessions) {
  if (!sessions.length) return '';
  return `
    <section class="peer-section peer-section--alert">
      <div class="channel-users-header">
        <span class="channel-users-title">待确认的配对</span>
        <span class="channel-users-hint">两台设备上的验证码必须完全一致才可批准 —— 不一致说明链路被人插了中间人</span>
      </div>
      ${sessions.map((s) => `
        <div class="peer-pending" data-peer-id="${_escAttr(s.peer_id)}">
          <div class="peer-pending-who">
            <span class="user-name">${_esc(s.display_name || '(未提供名称)')}</span>
            <code class="peer-fingerprint-sm">${_esc((s.peer_id || '').slice(0, 12))}…</code>
          </div>
          <div class="peer-sas">${_esc(s.code)}</div>
          <div class="peer-pending-actions">
            <button class="btn-primary btn-sm" data-peer-approve>批准</button>
            <button class="btn-ghost btn-sm" data-peer-reject>拒绝</button>
          </div>
        </div>`).join('')}
    </section>`;
}

/**
 * Status of one paired peer, as three distinct facts rather than one dot.
 *
 * The row used to carry a single dot whose colour encoded the *role*, sitting
 * right next to a dropdown that already said "Operator" — so it read as a status
 * light, and an `operator` peer looked green whether it was running, idle or
 * switched off. Three states are worth telling apart:
 *
 *   online + agent on   — can be given work
 *   online + agent off  — reachable, answers tool calls and shares state, but
 *                         /delegate returns 503 (智能控制 is off over there)
 *   offline             — nothing has been heard from it
 */
function _peerStatus(p) {
  // Pairing is per-direction: each side writes its own record. Confirming on only
  // one machine leaves *this* side listing the peer as paired while the other
  // rejects every signed request — and the row looked perfectly healthy. The state
  // push runs every 5s, so a 403 on it is the signal that nobody approved over
  // there. Shown as its own state because the fix is a human action on the other
  // screen, not something to wait out.
  // Two kinds of 403, with different fixes:
  // 1. One-sided pairing: we confirmed, they haven't yet → wait for them
  // 2. They unpaired us: was mutual, now they deleted us → re-pair from scratch
  if ((p.last_push_error || '').includes('403')) {
    const wasMutual = p.mutual_at != null;  // null/undefined = never mutual; 0 is a timestamp
    return wasMutual
      ? { state: 'idle', text: '对方已解除配对',
          title: '对方删除了本机的配对记录 —— 需要重新配对' }
      : { state: 'idle', text: '对方尚未批准配对',
          title: '本机已批准，但对端没有本机的配对记录 —— 请在对方设备上批准同一个验证码' };
  }
  // No evidence yet that the far side has us. Shown as waiting rather than paired:
  // confirming on this screen writes only this side's record, so until the peer
  // accepts a signed request from us (or sends us one) the pairing is half-done —
  // and it used to render exactly like a finished one.
  if (!p.mutual) {
    return { state: 'idle', text: '等待对方确认',
             title: '本机已批准。对端也批准同一个验证码之后，这里会自动变为在线' };
  }
  if (!p.online) {
    const age = p.contact_age_s == null ? '从未联系过'
      : `最后联系 ${_describeAge(p.contact_age_s)}`;
    return { state: 'offline', text: '离线', title: age };
  }
  if (p.agent_running === false) {
    return { state: 'idle', text: '在线 · 智能控制未开',
             title: '可查状态、可被调用工具，但接不了委派的任务' };
  }
  if (p.agent_running === true) {
    return { state: 'online', text: '在线', title: '智能控制已开，可以协同' };
  }
  return { state: 'online', text: '在线', title: '对方未上报智能控制状态' };
}

function _describeAge(seconds) {
  if (seconds < 60) return `${Math.round(seconds)} 秒前`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} 分钟前`;
  return `${(seconds / 3600).toFixed(1)} 小时前`;
}

function _renderPaired(peers) {
  const rows = peers.length
    ? peers.map((p) => {
        const st = _peerStatus(p);
        return `
        <div class="user-row" data-peer-id="${_escAttr(p.peer_id)}">
          <span class="peer-status-dot" data-state="${_escAttr(st.state)}" title="${_escAttr(st.title)}"></span>
          <span class="user-identity" title="${_escAttr(p.peer_id)}">
            <span class="user-name">${_esc(p.label || p.display_name || p.peer_id.slice(0, 12))}</span>
            <span class="user-id" title="${_escAttr(st.title)}">${_esc(st.text)}</span>
          </span>
          <select class="user-role-select" data-peer-role>
            ${_ROLE_ORDER.map((r) => `<option value="${r}" ${r === p.role ? 'selected' : ''}>${r[0].toUpperCase()}${r.slice(1)}</option>`).join('')}
          </select>
          <button class="user-remove" data-peer-unpair title="解除配对">×</button>
        </div>`;
      }).join('')
    : '<div class="channel-empty">还没有配对的机器人。在下方「发现到的机器人」里发起配对。</div>';

  return `
    <section class="peer-section">
      <div class="channel-users-header">
        <span class="channel-users-title">已配对</span>
        <span class="channel-users-hint">角色决定对方能请求什么；执行器仍需本机画布绑定才可能真正执行</span>
      </div>
      ${rows}
    </section>`;
}

/**
 * Hand-entered peer addresses.
 *
 * The escape hatch for what mDNS cannot do: it uses link-local multicast, so a
 * peer one subnet away is invisible even when the two machines ping each other
 * fine (measured between 10.100.121.0/24 and 10.100.128.0/19 in this building).
 * Until now the only way to add one was editing SQLite by hand.
 *
 * An entry here is *discovery only* — it makes a peer appear in the list below;
 * trust still comes from the 6-digit code confirmed on both screens.
 */
function _renderStatic(s) {
  const entries = (s.discovery?.static || []).map((e) =>
    (typeof e === 'string' ? { url: e } : e));
  const rows = entries.length
    ? entries.map((e, i) => `
        <div class="user-row" data-static-index="${i}">
          <span class="user-identity">
            <span class="user-name">${_esc(e.display_name || e.url)}</span>
            ${e.display_name ? `<span class="user-id">${_esc(e.url)}</span>` : ''}
          </span>
          <button class="user-remove" data-static-remove title="移除">×</button>
        </div>`).join('')
    : '<div class="channel-empty">还没有手动地址。同一子网内用不到这里，mDNS 会自动发现。</div>';

  return `
    <section class="peer-section">
      <div class="channel-users-header">
        <span class="channel-users-title">手动地址</span>
        <span class="channel-users-hint">跨子网时用 —— mDNS 不跨路由器；填了仍需验证码配对</span>
      </div>
      ${rows}
      <div class="peer-settings-row">
        <input class="peer-input" id="peer-static-url"
               placeholder="10.100.121.14 或 https://10.100.121.14:15678">
        <button class="btn-primary btn-sm" data-static-add>添加</button>
      </div>
    </section>`;
}

function _renderDiscovered(found) {
  const unpaired = found.filter((p) => !p.paired);
  const rows = unpaired.length
    ? unpaired.map((p) => {
        // A hand-entered address has no fingerprint yet — the static provider files
        // it under `static:<url>`. Rendering that id as the name produced a row
        // reading "static:https  static:https… · static", which says nothing about
        // *which* machine it is. Show the address, and say the identity is still
        // unknown rather than implying an id exists.
        const provisional = p.peer_id.startsWith('static:');
        const address = (p.endpoints || [])[0] || p.peer_id.replace(/^static:/, '');
        const name = p.display_name || (provisional ? address : p.peer_id.slice(0, 12));
        // Signal strength is the only way to tell two identical robots in the
        // same room apart before pairing — the fingerprints are both opaque.
        const rssi = typeof p.rssi === 'number' ? ` · ${p.rssi}dBm` : '';
        const meta = provisional
          ? `指纹待配对确认 · ${_esc((p.sources || []).join(',') || 'static')}`
          : `${_esc(p.peer_id.slice(0, 12))}… · ${_esc((p.sources || []).join(',') || '?')}${rssi}`;
        return `
        <div class="user-row" data-peer-id="${_escAttr(p.peer_id)}">
          <span class="user-identity" title="${_escAttr(provisional ? address : p.peer_id)}">
            <span class="user-name">${_esc(name)}</span>
            <span class="user-id">${meta}</span>
          </span>
          <button class="btn-primary btn-sm" data-peer-pair>配对</button>
        </div>`;
      }).join('')
    : (found.length
        // 只剩已配对的时候，说"没发现"会让人以为发现坏了 —— 图里就是这样：
        // 唯一在附近的机器人正好已经配对，这一段却像在报故障。
        ? `<div class="channel-empty">附近的 ${found.length} 台机器人都已配对。</div>`
        // "同一局域网"这句话不够 —— 真实情况是天轶在 10.100.128.0/19、两台 Orin 在
        // 10.100.121.0/24，同一栋楼、互相 ping 得通，但 mDNS 用链路本地多播
        // （224.0.0.251，TTL=1），设计上不跨路由。读者会以为发现功能坏了，而它工作正常。
        : `<div class="channel-empty">附近没有发现其他机器人。
             <div class="peer-empty-hint">mDNS 只在<b>同一子网</b>内有效 —— 它走链路本地多播，
             不跨路由器。对方在别的子网时这里就会是空的，即使两台互相 ping 得通。
             那种情况需要用静态清单（<code>peer_settings.static.peers</code>）把地址写死，
             再走同样的验证码配对。</div>
           </div>`);

  return `
    <section class="peer-section">
      <div class="channel-users-header">
        <span class="channel-users-title">发现到的机器人</span>
        <span class="channel-users-hint">发现 ≠ 可信 —— 必须经过验证码配对才能协作</span>
      </div>
      ${rows}
    </section>`;
}

// ── Actions ──────────────────────────────────────────────────────────────────

async function _onClick(e) {
  const save = e.target.closest('[data-peer-save]');
  if (save) return _saveSettings();

  if (e.target.closest('[data-static-add]')) return _addStatic();
  const staticRow = e.target.closest('[data-static-index]');
  if (staticRow && e.target.closest('[data-static-remove]')) {
    return _removeStatic(Number(staticRow.dataset.staticIndex));
  }

  const row = e.target.closest('[data-peer-id]');
  const peerId = row?.dataset.peerId;

  if (e.target.closest('[data-peer-pair]')) return _startPairing(peerId);
  if (e.target.closest('[data-peer-approve]')) return _approve(peerId, row);
  if (e.target.closest('[data-peer-reject]')) return _reject(peerId);
  if (e.target.closest('[data-peer-unpair]')) return _unpair(peerId, row);
}

async function _onChange(e) {
  if (e.target.id === 'peer-enabled' || e.target.id === 'peer-mdns'
      || e.target.id === 'peer-ble') return _saveSettings();
  const sel = e.target.closest('[data-peer-role]');
  if (sel) return _setRole(sel);
}


async function _currentStatic() {
  const s = await _get('/api/peer/settings');
  return (s.discovery?.static || []).map((e) => (typeof e === 'string' ? { url: e } : e));
}

async function _putStatic(list, okText) {
  try {
    const res = await fetch('/api/peer/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ static: list }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || '保存失败');
    showToast(json.discovery_restarted ? okText : `已保存，但发现层重启失败：${json.error}`);
  } catch (err) {
    showToast(`失败：${err.message}`);
  }
  _refresh();
}

async function _addStatic() {
  const input = document.getElementById('peer-static-url');
  const url = (input?.value || '').trim();
  if (!url) return showToast('先填一个地址');
  const list = await _currentStatic();
  // Normalisation and duplicate-checking happen server-side, so a bare host or a
  // repeat is not an error here — it comes back cleaned in the next refresh.
  list.push({ url });
  if (input) input.value = '';
  return _putStatic(list, '已添加，正在尝试发现');
}

async function _removeStatic(index) {
  const list = await _currentStatic();
  if (index < 0 || index >= list.length) return _refresh();
  const [gone] = list.splice(index, 1);
  return _putStatic(list, `已移除 ${gone.display_name || gone.url}`);
}

async function _saveSettings() {
  const enabled = document.getElementById('peer-enabled')?.checked;
  const mdns = document.getElementById('peer-mdns')?.checked;
  const ble = document.getElementById('peer-ble')?.checked;
  const displayName = document.getElementById('peer-display-name')?.value ?? '';
  try {
    const res = await fetch('/api/peer/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled, mdns, ble, display_name: displayName }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || '保存失败');
    // Report a discovery restart that failed rather than implying success.
    showToast(json.discovery_restarted === false
      ? `已保存，但发现层重启失败：${json.error}`
      : '已保存并生效');
  } catch (err) {
    showToast(`保存失败：${err.message}`);
  }
  _refresh();
}

async function _startPairing(peerId) {
  try {
    const res = await fetch('/api/peer/pair/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ peer_id: peerId }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || '发起配对失败');
    // For a manual address the id we clicked was provisional; the response carries
    // the fingerprint the peer just proved, and that is what the row below is keyed
    // by from now on.
    const who = json.display_name || `${(json.peer_id || '').slice(0, 12)}…`;
    showToast(`验证码 ${json.code} — 对方是 ${who}，请与对方设备核对后双方批准`);
  } catch (err) {
    showToast(`发起配对失败：${err.message}`);
  }
  _refresh();
}

async function _approve(peerId, row) {
  const code = row?.querySelector('.peer-sas')?.textContent?.trim();
  if (!code) return;
  if (!confirm(
    `确认对方设备上显示的验证码也是 ${code} 吗？\n\n` +
    `对端指纹：${peerId.slice(0, 12)}…\n\n` +
    `两边不一致说明链路上有中间人，此时不要批准。`
  )) return;

  try {
    const res = await fetch('/api/peer/pair/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ peer_id: peerId, code }),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || '批准失败');
    showToast(json.code_verified === false
      ? '该 peer 此前已配对，本次未校验验证码'
      : '配对完成，默认权限为只读');
  } catch (err) {
    showToast(`批准失败：${err.message}`);
  }
  _refresh();
}

async function _reject(peerId) {
  try {
    await fetch('/api/peer/pair/reject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ peer_id: peerId }),
    });
    showToast('已拒绝该配对请求');
  } catch (err) {
    showToast(`拒绝失败：${err.message}`);
  }
  _refresh();
}

async function _setRole(select) {
  const row = select.closest('[data-peer-id]');
  const peerId = row?.dataset.peerId;
  const role = select.value;
  const name = row?.querySelector('.user-name')?.textContent || peerId.slice(0, 12);

  // Promotion is the one direction that widens what a remote machine may ask
  // for, so make the operator confirm — and say plainly that it is still a
  // request, not control, so this is not misread as handing over the robot.
  if (role === 'operator' && !confirm(
    `把「${name}」提升为 operator？\n\n` +
    `它将可以请求执行器工具。执行器双闸门仍然生效：\n` +
    `实际执行仍需本机 LLM 决策，且该工具必须已在画布上连到决策核心。\n\n` +
    `这是授予「可以请求」，不是交出控制权。`
  )) {
    _refresh();
    return;
  }

  try {
    const res = await fetch(`/api/peer/paired/${encodeURIComponent(peerId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || '修改失败');
    showToast(`${name} → ${role}：${_ROLE_HINT[role] || ''}`);
  } catch (err) {
    showToast(`修改失败：${err.message}`);
  }
  _refresh();
}

async function _unpair(peerId, row) {
  const name = row?.querySelector('.user-name')?.textContent || peerId.slice(0, 12);
  if (!confirm(`解除与「${name}」的配对？\n\n对方将无法再请求任何工具，重新协作需要再走一次验证码配对。`)) return;
  try {
    await fetch(`/api/peer/paired/${encodeURIComponent(peerId)}`, { method: 'DELETE' });
    showToast('已解除配对');
  } catch (err) {
    showToast(`解除失败：${err.message}`);
  }
  _refresh();
}

// ── Escaping ─────────────────────────────────────────────────────────────────
//
// display_name comes from the peer and is attacker-controlled; it must never
// reach innerHTML unescaped.

function _esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function _escAttr(s) { return _esc(s); }
