/**
 * network.js — 网络设置面板（WiFi 扫描/连接/断开/忘记）
 */

let _overlay, _wifiList, _savedList, _interfacesList, _scanBtn;

export function initNetwork() {
  _overlay        = document.getElementById('network-overlay');
  _wifiList       = document.getElementById('network-wifi-list');
  _savedList      = document.getElementById('network-saved-list');
  _interfacesList = document.getElementById('network-interfaces-list');
  _scanBtn        = document.getElementById('network-wifi-scan');

  document.getElementById('btn-network').addEventListener('click', _open);
  document.getElementById('network-close').addEventListener('click', _close);
  _overlay.addEventListener('click', (e) => { if (e.target === _overlay) _close(); });
  _scanBtn.addEventListener('click', _loadWifi);
}

function _open() {
  _overlay.classList.remove('hidden');
  _loadAll();
}

function _close() {
  _overlay.classList.add('hidden');
}

async function _loadAll() {
  await Promise.all([_loadInterfaces(), _loadWifi(), _loadSaved()]);
}

// ── Interfaces ───────────────────────────────────────────────────────────────

async function _loadInterfaces() {
  try {
    const res = await fetch('/api/network/interfaces');
    const json = await res.json();
    const ifaces = json.data?.interfaces || [];
    _interfacesList.innerHTML = ifaces.map(i => `
      <div class="network-iface-item">
        <div class="network-iface-header">
          <span class="network-iface-name">${_esc(i.device)}</span>
          <span class="network-iface-type">${i.type}</span>
          <span class="network-iface-state ${i.state === 'connected' ? 'connected' : ''}">${_stateLabel(i.state)}</span>
          ${i.connection ? `<span class="network-iface-conn">${_esc(i.connection)}</span>` : ''}
        </div>
        ${i.state === 'connected' ? `<div class="network-iface-grid">
          ${i.ip ? `<div class="network-iface-cell"><span class="network-iface-label">IP</span><span class="network-iface-value">${_esc(i.ip)}</span></div>` : ''}
          ${i.mask ? `<div class="network-iface-cell"><span class="network-iface-label">掩码</span><span class="network-iface-value">${_esc(i.mask)}</span></div>` : ''}
          ${i.gateway ? `<div class="network-iface-cell"><span class="network-iface-label">网关</span><span class="network-iface-value">${_esc(i.gateway)}</span></div>` : ''}
          ${i.mac ? `<div class="network-iface-cell"><span class="network-iface-label">MAC</span><span class="network-iface-value">${_esc(i.mac)}</span></div>` : ''}
        </div>` : ''}
        ${i.state === 'connected' && i.gateway ? `<div class="network-iface-policy-route" title="开启后，来自此网卡的连接会强制从此网卡回复，用于此网卡不是主上行网络时避免外部无法访问其 IP。不影响此网卡作为默认出口。旧版 NetworkManager 上切换时可能会短暂断开重连此网卡">
          <span class="network-iface-label">策略路由（防止回包走错网卡）</span>
          <label class="toggle-switch">
            <input type="checkbox" class="network-policy-route-toggle" data-device="${_attr(i.device)}" ${i.policy_route ? 'checked' : ''} />
            <span class="toggle-slider"></span>
          </label>
        </div>` : ''}
      </div>
    `).join('');

    _interfacesList.querySelectorAll('.network-policy-route-toggle').forEach(el => {
      el.addEventListener('change', () => _togglePolicyRoute(el));
    });
  } catch {
    _interfacesList.innerHTML = '<div class="network-empty">无法获取接口信息</div>';
  }
}

async function _togglePolicyRoute(el) {
  const device = el.dataset.device;
  const enable = el.checked;
  if (!confirm(`此操作会重新应用 ${device} 的网络配置，旧版 NetworkManager 上可能短暂断开重连，确认继续？`)) {
    el.checked = !enable;
    return;
  }
  el.disabled = true;
  try {
    const res = await fetch('/api/network/policy-route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device, enable }),
    });
    const json = await res.json();
    if (!json.data?.success) {
      el.checked = !enable;
      alert('设置失败: ' + (json.data?.error || '未知错误'));
    }
  } catch (e) {
    el.checked = !enable;
    alert('设置失败: ' + e.message);
  } finally {
    el.disabled = false;
  }
}

function _stateLabel(state) {
  const map = { connected: '已连接', disconnected: '已断开', unavailable: '不可用', unmanaged: '未托管' };
  return map[state] || state;
}

// ── WiFi Scan ────────────────────────────────────────────────────────────────

async function _loadWifi() {
  _wifiList.innerHTML = '<div class="network-loading">扫描中…</div>';
  _scanBtn.disabled = true;
  try {
    const res = await fetch('/api/network/wifi/scan');
    const json = await res.json();
    if (json.code !== 200) throw new Error(json.data?.error || 'scan failed');
    const networks = json.data?.networks || [];
    if (!networks.length) {
      _wifiList.innerHTML = '<div class="network-empty">未发现 WiFi 网络</div>';
      return;
    }
    _wifiList.innerHTML = networks.map(n => `
      <div class="network-wifi-item ${n.in_use ? 'active' : ''}" data-ssid="${_attr(n.ssid)}">
        <div class="network-wifi-row">
          <span class="network-wifi-signal">${_signalBars(n.signal)}</span>
          <span class="network-wifi-ssid">${_esc(n.ssid)}</span>
          ${n.security ? '<span class="network-wifi-lock">🔒</span>' : ''}
          ${n.in_use ? '<span class="network-wifi-badge">已连接</span>' : ''}
        </div>
        <div class="network-wifi-form hidden">
          <input type="password" class="network-wifi-pwd" placeholder="输入密码" autocomplete="off" />
          <label class="network-wifi-auto-label">
            <input type="checkbox" class="network-wifi-auto" checked /> 自动连接
          </label>
          <div class="network-wifi-actions">
            <button class="btn-primary btn-sm network-wifi-connect-btn">连接</button>
            ${n.in_use ? '<button class="btn-ghost btn-sm network-wifi-disconnect-btn">断开</button>' : ''}
          </div>
          <div class="network-wifi-msg"></div>
        </div>
      </div>
    `).join('');

    // Bind events
    _wifiList.querySelectorAll('.network-wifi-item').forEach(el => {
      const row = el.querySelector('.network-wifi-row');
      const form = el.querySelector('.network-wifi-form');
      row.addEventListener('click', () => {
        // Toggle this form, close others
        _wifiList.querySelectorAll('.network-wifi-form').forEach(f => {
          if (f !== form) f.classList.add('hidden');
        });
        form.classList.toggle('hidden');
      });
      const connectBtn = el.querySelector('.network-wifi-connect-btn');
      connectBtn?.addEventListener('click', () => _connectWifi(el));
      const disconnectBtn = el.querySelector('.network-wifi-disconnect-btn');
      disconnectBtn?.addEventListener('click', () => _disconnectWifi(el));
    });
  } catch (e) {
    _wifiList.innerHTML = `<div class="network-empty">扫描失败: ${_esc(e.message)}</div>`;
  } finally {
    _scanBtn.disabled = false;
  }
}

async function _connectWifi(el) {
  const ssid = el.dataset.ssid;
  const pwd = el.querySelector('.network-wifi-pwd').value;
  const auto = el.querySelector('.network-wifi-auto').checked;
  const msg = el.querySelector('.network-wifi-msg');
  const btn = el.querySelector('.network-wifi-connect-btn');

  btn.disabled = true;
  btn.textContent = '连接中…';
  msg.textContent = '';

  try {
    const res = await fetch('/api/network/wifi/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid, password: pwd, auto_connect: auto }),
    });
    const json = await res.json();
    if (json.data?.success) {
      msg.textContent = '✓ ' + (json.data.message || '已连接');
      msg.className = 'network-wifi-msg success';
      setTimeout(() => _loadAll(), 1000);
    } else {
      msg.textContent = '✗ ' + (json.data?.error || '连接失败');
      msg.className = 'network-wifi-msg error';
    }
  } catch (e) {
    msg.textContent = '✗ ' + e.message;
    msg.className = 'network-wifi-msg error';
  } finally {
    btn.disabled = false;
    btn.textContent = '连接';
  }
}

async function _disconnectWifi(el) {
  const msg = el.querySelector('.network-wifi-msg');
  try {
    const res = await fetch('/api/network/wifi/disconnect', { method: 'POST' });
    const json = await res.json();
    if (json.data?.success) {
      msg.textContent = '✓ 已断开';
      msg.className = 'network-wifi-msg success';
      setTimeout(() => _loadAll(), 1000);
    } else {
      msg.textContent = '✗ ' + (json.data?.error || '断开失败');
      msg.className = 'network-wifi-msg error';
    }
  } catch (e) {
    msg.textContent = '✗ ' + e.message;
    msg.className = 'network-wifi-msg error';
  }
}

// ── Saved Networks ───────────────────────────────────────────────────────────

async function _loadSaved() {
  try {
    const res = await fetch('/api/network/wifi/saved');
    const json = await res.json();
    const conns = json.data?.connections || [];
    if (!conns.length) {
      _savedList.innerHTML = '<div class="network-empty">无已保存的网络</div>';
      return;
    }
    _savedList.innerHTML = conns.map(c => `
      <div class="network-saved-item">
        <span class="network-saved-name">${_esc(c.name)}</span>
        ${c.db_only ? '<span class="network-saved-tag">仅数据库</span>' : ''}
        <button class="btn-ghost btn-sm network-forget-btn" data-name="${_attr(c.name)}">忘记</button>
      </div>
    `).join('');
    _savedList.querySelectorAll('.network-forget-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const name = btn.dataset.name;
        btn.disabled = true;
        await fetch(`/api/network/wifi/saved/${encodeURIComponent(name)}`, { method: 'DELETE' });
        _loadSaved();
      });
    });
  } catch {
    _savedList.innerHTML = '<div class="network-empty">无法获取已保存网络</div>';
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function _signalBars(signal) {
  const bars = signal >= 75 ? 4 : signal >= 50 ? 3 : signal >= 25 ? 2 : 1;
  return Array.from({ length: 4 }, (_, i) =>
    `<span class="signal-bar ${i < bars ? 'filled' : ''}"></span>`
  ).join('');
}

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function _attr(s) {
  return (s || '').replace(/"/g, '&quot;');
}
