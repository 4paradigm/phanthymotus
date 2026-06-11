/** text.js — Renders streaming text output */
export const TextRenderer = {
  name: 'text',
  canRender: (hint) => hint && (hint.startsWith('text/') || hint === 'data/json'),
  _el: null,
  _textEl: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-text';

    this._textEl = document.createElement('div');
    this._textEl.className = 'renderer-text-main';
    this._el.appendChild(this._textEl);

    container.appendChild(this._el);
  },

  onData(buffer) {
    if (!this._textEl) return;
    try {
      const str = new TextDecoder().decode(buffer);
      const json = JSON.parse(str);
      if (json.type === 'ping' || json.type === 'meta') return;
      if (json.text) {
        const p = document.createElement('p');
        p.className = 'asr-line';
        p.textContent = json.text;
        this._textEl.appendChild(p);
        this._textEl.scrollTop = this._textEl.scrollHeight;
        while (this._textEl.children.length > 50) this._textEl.removeChild(this._textEl.firstChild);
      } else if (_isTableData(json)) {
        // Array-of-objects data (e.g. joints) — render as live-updating table
        _renderTable(this._textEl, json);
      } else {
        // Sensor/generic JSON — show as log entry with timestamp
        const row = document.createElement('div');
        row.className = 'log-entry';
        const t = new Date().toLocaleTimeString();
        row.innerHTML = `<span class="log-time">${t}</span><span class="log-msg" style="color:var(--text-muted);font-size:12px;white-space:pre-wrap">${_formatJson(json)}</span>`;
        this._textEl.appendChild(row);
        this._textEl.scrollTop = this._textEl.scrollHeight;
        while (this._textEl.children.length > 50) this._textEl.removeChild(this._textEl.firstChild);
      }
    } catch {
      // non-JSON text (plain text/*)
      const p = document.createElement('p');
      p.textContent = new TextDecoder().decode(buffer);
      this._textEl.appendChild(p);
    }
  },

  onDataSilent(buffer) { this.onData(buffer); },

  onEvent(event) {
    if (!this._el) return;
    if (event.type === 'mcp_result') {
      const text = event.payload?.result;
      if (typeof text === 'string') {
        const p = document.createElement('p');
        p.style.cssText = 'color:var(--accent);margin-bottom:4px';
        p.textContent = text;
        this._textEl.appendChild(p);
        this._textEl.scrollTop = this._textEl.scrollHeight;
      }
    }
    if (event.type === 'agent_thought') {
      const p = document.createElement('p');
      p.style.cssText = 'color:var(--text-muted);font-size:12px;margin-bottom:12px';
      p.textContent = '💭 ' + (event.payload?.text || '');
      this._textEl.appendChild(p);
    }
  },

  unmount() { this._el?.remove(); this._el = null; this._textEl = null; },
};

function _formatJson(obj) {
  // Compact single-line for small objects, otherwise pretty-print
  const keys = Object.keys(obj);
  if (keys.length <= 6) {
    return keys.map(k => {
      const v = obj[k];
      const val = Array.isArray(v)
        ? `[${v.map(n => typeof n === 'number' ? n.toFixed(3) : JSON.stringify(n)).join(', ')}]`
        : JSON.stringify(v);
      return `<b>${k}</b>: ${val}`;
    }).join('  ');
  }
  // For larger objects, use compact key: value format (one per line)
  return keys.map(k => {
    const v = obj[k];
    const val = Array.isArray(v)
      ? `[${v.map(n => typeof n === 'number' ? n.toFixed(3) : JSON.stringify(n)).join(', ')}]`
      : JSON.stringify(v);
    return `<b>${k}</b>: ${val}`;
  }).join('\n');
}

/** Returns true if obj is {key: Array<object>} style state data suitable for table rendering */
function _isTableData(obj) {
  const keys = Object.keys(obj);
  // Single key whose value is an array of plain objects with consistent numeric fields
  if (keys.length === 1) {
    const arr = obj[keys[0]];
    return Array.isArray(arr) && arr.length > 0 && typeof arr[0] === 'object' && arr[0] !== null;
  }
  return false;
}

/**
 * Render or update a live table inside container.
 * Uses a keyed approach: first call creates the table, subsequent calls update cell values in-place.
 */
function _renderTable(container, json) {
  const key = Object.keys(json)[0];
  const rows = json[key];
  if (!rows.length) return;

  const cols = Object.keys(rows[0]);

  // Create table on first call
  let wrap = container.querySelector('.live-table-wrap');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.className = 'live-table-wrap';

    const label = document.createElement('div');
    label.className = 'live-table-label';
    label.textContent = key;
    wrap.appendChild(label);

    const tbl = document.createElement('table');
    tbl.className = 'live-table';

    // Header
    const thead = tbl.createTHead();
    const hrow = thead.insertRow();
    cols.forEach(c => {
      const th = document.createElement('th');
      th.textContent = c;
      hrow.appendChild(th);
    });

    // Body — pre-create all rows
    const tbody = tbl.createTBody();
    rows.forEach((r, i) => {
      const tr = tbody.insertRow();
      tr.dataset.idx = i;
      cols.forEach(c => {
        const td = tr.insertCell();
        td.className = 'live-cell';
        td.textContent = _fmtCell(r[c]);
      });
    });

    wrap.appendChild(tbl);
    container.appendChild(wrap);
    return;
  }

  // Update existing cells in-place (no DOM churn)
  const tbody = wrap.querySelector('tbody');
  const trows = tbody.rows;
  rows.forEach((r, i) => {
    if (!trows[i]) return;
    cols.forEach((c, ci) => {
      trows[i].cells[ci].textContent = _fmtCell(r[c]);
    });
  });
}

function _fmtCell(v) {
  if (typeof v === 'number') return v.toFixed(4);
  if (Array.isArray(v)) return v.map(n => typeof n === 'number' ? n.toFixed(1) : String(n)).join(' ');
  return String(v ?? '');
}
