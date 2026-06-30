/**
 * kv-latest.js — "最新模式" renderer for JSON topics.
 * Shows each top-level key as a cell with live-updating value.
 */
export const KvLatestRenderer = {
  name: 'kv-latest',
  canRender: (hint) => hint && (hint.startsWith('text/') || hint === 'data/json'),

  _el: null,
  _cells: {},  // { key: { el, valueEl, lastValue } }

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-kv-latest';
    this._cells = {};
    container.appendChild(this._el);
  },

  onData(buffer) {
    if (!this._el) return;
    try {
      const str = new TextDecoder().decode(buffer);
      const json = JSON.parse(str);
      if (json.type === 'ping' || json.type === 'meta') return;

      const flat = _flatten(json);
      for (const [key, value] of Object.entries(flat)) {
        this._updateCell(key, value);
      }
    } catch { /* non-JSON, ignore */ }
  },

  _updateCell(key, value) {
    let cell = this._cells[key];
    if (!cell) {
      const el = document.createElement('div');
      el.className = 'kv-cell';

      const keyEl = document.createElement('div');
      keyEl.className = 'kv-key';
      keyEl.textContent = key;
      el.appendChild(keyEl);

      const valueEl = document.createElement('div');
      valueEl.className = 'kv-value';
      el.appendChild(valueEl);

      this._el.appendChild(el);
      cell = { el, valueEl, lastValue: undefined };
      this._cells[key] = cell;
    }

    const formatted = _formatValue(value);
    if (formatted !== cell.lastValue) {
      cell.valueEl.textContent = formatted;
      cell.lastValue = formatted;
      // Flash animation
      cell.el.classList.remove('flash');
      void cell.el.offsetWidth; // force reflow
      cell.el.classList.add('flash');
    }
  },

  onEvent() {},

  unmount() {
    this._el?.remove();
    this._el = null;
    this._cells = {};
  },
};

/** Flatten object 1 level deep using dot notation */
function _flatten(obj) {
  const result = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      for (const [sk, sv] of Object.entries(v)) {
        result[`${k}.${sk}`] = sv;
      }
    } else {
      result[k] = v;
    }
  }
  return result;
}

/** Format a value for display */
function _formatValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) {
    return v.map(n => {
      if (typeof n === 'number') return Number.isInteger(n) ? String(n) : n.toFixed(2);
      if (typeof n === 'object' && n !== null) return JSON.stringify(n);
      return String(n);
    }).join(', ');
  }
  if (typeof v === 'object') return JSON.stringify(v);
  if (typeof v === 'string') return v.length > 60 ? v.slice(0, 57) + '…' : v;
  return JSON.stringify(v);
}
