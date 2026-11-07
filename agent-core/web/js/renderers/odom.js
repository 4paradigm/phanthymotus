/** Read-only motus.odom/1 snapshots and logs; original wire text is retained. */
import { decodeNestedJson } from './json-util.js';
import { ODOM_DETAILS, projectOdom, valueAtPath } from './odom-presentation.js';

export const OdomRenderer = {
  name: 'odom',
  canRender: hint => hint === 'state/odom',

  mount(container, options = {}) {
    this._options = typeof options === 'string' ? { stateKey: options } : options;
    this._mode = this._options.mode === 'log' ? 'log' : 'kv';
    this._records = [];
    this._el = document.createElement('div');
    this._el.className = 'renderer-odom';
    this._el.style.cssText = 'height:100%;min-height:0;display:flex;flex-direction:column;gap:6px;padding:6px';
    if (this._options.showModeToggle !== false) {
      const modes = document.createElement('div');
      modes.className = 'odom-modes monitor-card-modes';
      for (const [mode, label] of [['log', '日志'], ['kv', '最新']]) {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'monitor-card-mode-btn';
        button.textContent = label; button.dataset.mode = mode;
        button.onclick = () => this.setMode(mode);
        modes.appendChild(button);
      }
      this._el.appendChild(modes);
    }
    this._freshness = document.createElement('div');
    this._freshness.className = 'odom-freshness';
    this._freshness.style.cssText = 'font-size:11px;color:var(--text-dim)';
    this._el.appendChild(this._freshness);
    this._body = document.createElement('div');
    this._body.style.cssText = 'min-height:0;overflow:auto;flex:1';
    this._el.appendChild(this._body);
    container.appendChild(this._el);
    this._render();
    this._timer = setInterval(() => this._updateFreshness(), 1000);
  },

  onData(buffer, format, context = {}) {
    if (!this._el) return;
    let raw, value;
    try {
      raw = typeof buffer === 'string' ? buffer : new TextDecoder().decode(buffer);
      value = JSON.parse(raw);
    } catch { return; }
    if (value?.schema !== 'motus.odom/1') return;
    const record = { raw, value: decodeNestedJson(value), receivedAt: context.receivedAt ?? Date.now() };
    this._records.push(record);
    if (this._records.length > 50) this._records.shift();
    if (this._mode === 'log') {
      this._appendRecord(record);
      while (this._body.children.length > 50) this._body.removeChild(this._body.firstChild);
      this._body.scrollTop = this._body.scrollHeight;
    } else this._renderLatest(record);
    this._updateFreshness();
  },

  onDataSilent(buffer, format, context) { this.onData(buffer, format, context); },
  onEvent() {},

  setMode(mode) {
    if (mode !== 'log' && mode !== 'kv') return;
    this._mode = mode;
    this._render();
  },

  _render() {
    this._body.textContent = '';
    this._body.className = this._mode === 'log' ? 'odom-log' : 'odom-latest';
    for (const button of this._el.querySelectorAll('.monitor-card-mode-btn')) {
      button.classList.toggle('active', button.dataset.mode === this._mode);
    }
    if (this._mode === 'log') this._records.forEach(record => this._appendRecord(record));
    else if (this._records.length) this._appendRecord(this._records.at(-1));
    this._updateFreshness();
  },

  _renderLatest(record) {
    const open = [...this._body.querySelectorAll('details')].map(detail => detail.open);
    this._body.textContent = '';
    this._appendRecord(record);
    this._body.querySelectorAll('details').forEach((detail, index) => { detail.open = open[index] || false; });
  },

  _appendRecord(record) {
    const row = document.createElement('div');
    row.className = 'odom-record';
    if (this._mode === 'log') {
      const time = document.createElement('div');
      time.className = 'log-time'; time.textContent = new Date(record.receivedAt).toLocaleTimeString();
      row.appendChild(time);
    }
    const summary = document.createElement('div');
    summary.className = 'odom-summary';
    summary.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px;padding:4px';
    for (const field of projectOdom(record.value)) {
      const cell = document.createElement('div'); cell.className = 'kv-cell';
      const key = document.createElement('div'); key.className = 'kv-key';
      key.textContent = field.group ? field.group + ' · ' + field.label : field.label;
      const value = document.createElement('div'); value.className = 'kv-value'; value.textContent = field.text;
      cell.append(key, value); summary.appendChild(cell);
    }
    row.appendChild(summary);
    for (const group of ODOM_DETAILS) {
      const detail = document.createElement('details'); detail.className = 'odom-detail';
      const label = document.createElement('summary'); label.textContent = group.label || group.path;
      const value = document.createElement('pre');
      value.textContent = JSON.stringify(valueAtPath(record.value, group.path) ?? null, null, 2);
      value.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere';
      detail.append(label, value); row.appendChild(detail);
    }
    const raw = document.createElement('details'); raw.className = 'odom-raw';
    const label = document.createElement('summary'); label.textContent = '原始数据';
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'odom-copy monitor-card-mode-btn'; copy.textContent = '复制';
    copy.onclick = async () => {
      try { await copyRawText(record.raw, this._el); copy.textContent = '已复制'; }
      catch { copy.textContent = '复制失败'; }
    };
    const text = document.createElement('pre'); text.textContent = record.raw;
    text.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere';
    raw.append(label, copy, text); row.appendChild(raw);
    this._body.appendChild(row);
  },

  _updateFreshness() {
    const record = this._records.at(-1);
    if (!record) { this._freshness.textContent = '等待数据'; return; }
    const age = Math.max(0, Date.now() - record.receivedAt);
    const stale = 5000;
    this._freshness.textContent = `接收年龄 ${(age / 1000).toFixed(1)} s${age > stale ? ' · 陈旧' : ''}`;
  },

  unmount() {
    clearInterval(this._timer);
    this._el?.remove(); this._el = null; this._records = [];
  },
};

async function copyRawText(raw, container) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(raw); return; } catch { /* HTTP/permission fallback below */ }
  }
  const input = document.createElement('textarea');
  input.value = raw; input.style.cssText = 'position:fixed;left:-10000px;top:0';
  container.appendChild(input);
  try {
    input.select();
    if (!document.execCommand('copy')) throw new Error('Copy unavailable');
  } finally { input.remove(); }
}

/** Prefer an advertised odometry preview without modifying the producer's port order. */
export function odomPreview(skill) {
  const topic = (skill.topic_out || []).find(port => port.format === 'state/odom' && port.topic);
  return topic ? { path: '/ws/bus' + topic.topic, hint: topic.format, topic }
    : { path: skill.ws_path, hint: skill.render_hint || 'activity', topic: skill.topic_out?.[0] || null };
}
