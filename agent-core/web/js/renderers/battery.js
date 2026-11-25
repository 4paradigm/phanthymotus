/**
 * battery.js — gauge for the normalized `sensor/battery` topic.
 *
 * The format comes from perception's `battery_monitor` card, which folds the
 * vendor-specific fields (Bumi soc/soh, G1 BMS lists, Tianyi master_voltage)
 * into one reading. Fields this robot does not report are absent rather than
 * zero, so the gauge shows "—" and says why instead of drawing an empty battery
 * that looks like 0% left.
 */

export function parseBatteryPayload(input) {
  let data;
  try {
    data = typeof input === 'string' ? JSON.parse(input)
      : input instanceof ArrayBuffer || ArrayBuffer.isView(input)
        ? JSON.parse(new TextDecoder().decode(input)) : input;
  } catch { return null; }
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null;
  if (data.type === 'ping' || data.type === 'meta') return null;
  return data;
}

/** Level bucket for the fill colour. Unknown SOC is its own bucket. */
export function batteryLevel(reading) {
  const raw = reading?.soc;
  // `Number(null)` and `Number('')` are both 0 — a JSON field that is present
  // but empty must not be drawn as "0% left".
  if (raw === null || raw === undefined || raw === '' || typeof raw === 'boolean') return 'unknown';
  const soc = Number(raw);
  if (!Number.isFinite(soc)) return 'unknown';
  if (soc <= 20) return 'low';
  if (soc <= 50) return 'mid';
  return 'ok';
}

/**
 * Cells shown under the gauge: [label, value, unit]. Absent fields are skipped.
 *
 * Canonical field names carry their unit (`current_ma`, `cell_voltage_max_mv`),
 * so the display converts to the units a person reads (A, V) instead of
 * relabelling the raw number — 3140 mA is shown as 3.14 A, never as "3140 A".
 */
export function batteryDetails(reading) {
  const cells = [];
  const add = (label, value, unit = '') => {
    if (value === undefined || value === null || value === '') return;
    cells.push([label, value, unit]);
  };
  add('SOH', reading?.soh, '%');
  add('TEMP', reading?.temperature_c, '°C');
  add('VOLT', reading?.voltage_v, 'V');
  if (reading?.pack_voltage_mv !== undefined) {
    add('PACK', (Number(reading.pack_voltage_mv) / 1000).toFixed(1), 'V');
  }
  if (reading?.current_a !== undefined) add('CURR', reading.current_a, 'A');
  else if (reading?.current_ma !== undefined) add('CURR', (Number(reading.current_ma) / 1000).toFixed(2), 'A');
  add('PWR', reading?.power_w, 'W');
  if (reading?.cell_voltage_max_mv !== undefined) {
    add('CELL MAX', (Number(reading.cell_voltage_max_mv) / 1000).toFixed(2), 'V');
  }
  add('CYCLE', reading?.cycle, '');
  add('ALARM', reading?.alarm, '');
  if (reading?.charging === true) add('CHARGING', 'yes', '');
  if (reading?.charging === false) add('CHARGING', 'no', '');
  if (reading?.installed === false) add('BATTERY', 'not installed', '');
  if (reading?.working === false) add('BATTERY', 'not working', '');
  return cells;
}

const LEVEL_COLOR = {
  ok: 'var(--green)',
  mid: 'var(--yellow)',
  low: 'var(--red)',
  unknown: 'var(--text-dim)',
};

export const BatteryRenderer = {
  name: 'battery',
  channels: ['data'],   // frames consumed: 'data' topic frames, 'event' tool-call results
  canRender: (hint) => hint === 'sensor/battery',

  _el: null,
  _parts: null,

  mount(container) {
    const el = document.createElement('div');
    el.style.cssText =
      'display:flex;flex-direction:column;gap:10px;height:100%;padding:14px 16px;' +
      'background:var(--bg-card);font-family:var(--font-mono,monospace);box-sizing:border-box;overflow:hidden';

    const head = document.createElement('div');
    head.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px';
    const title = document.createElement('span');
    title.textContent = 'BATTERY';
    title.style.cssText = 'font-size:11px;letter-spacing:.14em;color:var(--text-muted)';
    const chip = document.createElement('span');
    chip.style.cssText = 'font-size:10px;padding:1px 7px;border-radius:999px;' +
      'border:1px solid var(--border-med);color:var(--text-dim);white-space:nowrap';
    head.append(title, chip);

    const gaugeRow = document.createElement('div');
    gaugeRow.style.cssText = 'display:flex;align-items:center;gap:12px';
    const shape = document.createElement('div');
    shape.style.cssText =
      'position:relative;flex:1;height:46px;border:2px solid var(--border-med);' +
      'border-radius:7px;padding:3px;box-sizing:border-box;overflow:hidden';
    const cap = document.createElement('div');
    cap.style.cssText = 'position:absolute;right:-7px;top:50%;transform:translateY(-50%);' +
      'width:5px;height:16px;border-radius:0 3px 3px 0;background:var(--border-med)';
    const fill = document.createElement('div');
    fill.style.cssText = 'height:100%;width:0;border-radius:4px;transition:width .4s ease,background .4s ease';
    shape.append(fill);
    const pct = document.createElement('span');
    pct.style.cssText = 'font-size:30px;line-height:1;color:var(--text);min-width:78px;text-align:right';
    gaugeRow.append(shape, cap, pct);

    const details = document.createElement('div');
    details.style.cssText =
      'display:grid;grid-template-columns:repeat(auto-fill,minmax(76px,1fr));gap:6px 10px';
    const note = document.createElement('div');
    note.style.cssText = 'font-size:11px;color:var(--text-muted);min-height:14px';
    const source = document.createElement('div');
    source.style.cssText = 'font-size:10px;color:var(--text-dim);white-space:nowrap;' +
      'overflow:hidden;text-overflow:ellipsis;margin-top:auto';

    el.append(head, gaugeRow, details, note, source);
    container.appendChild(el);

    this._el = el;
    this._parts = { chip, fill, pct, details, note, source };
    this._lastDetails = '';
  },

  onData(buffer) {
    const reading = parseBatteryPayload(buffer);
    const p = this._parts;
    if (!reading || !p) return;

    const level = batteryLevel(reading);
    const hasSoc = level !== 'unknown';
    const stale = reading.stale === true;

    p.fill.style.width = hasSoc ? Math.max(0, Math.min(100, Number(reading.soc))) + '%' : '0';
    p.fill.style.background = LEVEL_COLOR[level];
    p.fill.style.opacity = hasSoc || stale ? '1' : '.35';
    p.pct.textContent = hasSoc ? Math.round(Number(reading.soc)) + '%' : '—';

    if (stale) {
      p.chip.textContent = 'STALE' + (Number.isFinite(Number(reading.age_s)) ? ' ' + Math.round(Number(reading.age_s)) + 's' : '');
      p.chip.style.color = 'var(--red)';
      p.chip.style.borderColor = 'var(--red)';
      this._el.style.opacity = '.6';
    } else {
      p.chip.textContent = 'LIVE';
      p.chip.style.color = 'var(--green)';
      p.chip.style.borderColor = 'var(--green)';
      this._el.style.opacity = '1';
    }

    const cells = batteryDetails(reading);
    const key = JSON.stringify(cells);
    if (key !== this._lastDetails) {
      p.details.replaceChildren(...cells.map(([label, value, unit]) => {
        const cell = document.createElement('div');
        cell.style.cssText = 'display:flex;flex-direction:column;gap:1px;min-width:0';
        const k = document.createElement('span');
        k.textContent = label;
        k.style.cssText = 'font-size:9px;letter-spacing:.08em;color:var(--text-dim)';
        const v = document.createElement('span');
        v.textContent = value + unit;
        v.style.cssText = 'font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
        cell.append(k, v);
        return cell;
      }));
      this._lastDetails = key;
    }

    p.note.textContent = hasSoc
      ? ''
      : 'This robot does not report state of charge (SOC).';
    p.source.textContent = reading.source_topic ? 'from ' + reading.source_topic : '';
  },

  onEvent() {},

  unmount() {
    this._el?.remove();
    this._el = null;
    this._parts = null;
    this._lastDetails = '';
  },
};
