/** Joint feedback table for sensor/skeleton. q is radians; no target angles. */
const text = (value, fallback) => typeof value === 'string' && value.trim()
  ? value.trim().slice(0, 128) : fallback;

export function jointAngleRows(data) {
  if (data?.status?.available === false || !Array.isArray(data?.joints)) return [];
  return data.joints.slice(0, 256).filter(j => j && typeof j === 'object').map((j, i) => {
    const idx = Number.isInteger(j.idx) && j.idx >= 0 ? j.idx : i;
    const name = text(j.name, `joint_${idx}`);
    const q = typeof j.q === 'number' && Number.isFinite(j.q) ? j.q : null;
    const converted = q === null ? null : q * 180 / Math.PI;
    return { idx, name, label: text(j.label, name), group: text(j.group, '关节'),
      rad: q, deg: Number.isFinite(converted) ? converted : null };
  }).sort((a, b) => a.idx - b.idx);
}

export function angleText(value) {
  if (value === null || !Number.isFinite(value)) return '—';
  const formatted = value.toFixed(3);
  return formatted === '-0.000' ? '0.000' : formatted;
}

export function createJointAngleView(parent, { staleMs = 10000 } = {}) {
  const details = document.createElement('details');
  details.className = 'skeleton-joint-angles';
  details.style.cssText = 'flex:0 0 auto;max-height:50%;overflow:auto;color:#ddd;background:#202020;border-top:1px solid #444;font:12px/1.5 system-ui;';
  const summary = document.createElement('summary');
  summary.textContent = '关节角度（等待数据）';
  summary.style.cssText = 'cursor:pointer;padding:6px 10px;position:sticky;top:0;background:#202020;z-index:2;';
  details.appendChild(summary);
  const table = document.createElement('table');
  table.style.cssText = 'width:100%;border-collapse:collapse;table-layout:fixed;';
  table.setAttribute('aria-label', '实际关节角度：度和弧度');
  const head = document.createElement('thead');
  const header = document.createElement('tr');
  for (const [title, width] of [['索引', '12%'], ['关节', '40%'], ['角度（°）', '24%'], ['弧度（rad）', '24%']]) {
    const cell = document.createElement('th');
    cell.textContent = title;
    cell.style.cssText = `width:${width};padding:4px;text-align:left;color:#aaa;`;
    header.appendChild(cell);
  }
  head.appendChild(header);
  table.appendChild(head);
  const body = document.createElement('tbody');
  table.appendChild(body);
  details.appendChild(table);
  parent.appendChild(details);
  let layout = '', cells = [], timer = null, disposed = false;

  function render(rows, state) {
    summary.textContent = rows.length ? `关节角度（${rows.length}）` : `关节角度（${state}）`;
    const key = JSON.stringify(rows.map(r => [r.idx, r.name, r.group, r.label]));
    if (key !== layout) {
      layout = key;
      body.replaceChildren();
      cells = [];
      let group = null;
      for (const row of rows) {
        if (group !== row.group) {
          group = row.group;
          const tr = document.createElement('tr');
          const th = document.createElement('th');
          th.colSpan = 4;
          th.textContent = group;
          th.style.cssText = 'padding:5px 8px;text-align:left;background:#292929;color:#62cbe3;';
          tr.appendChild(th);
          body.appendChild(tr);
        }
        const tr = document.createElement('tr');
        const rowCells = [];
        for (const value of [String(row.idx), row.label, '', '']) {
          const td = document.createElement('td');
          td.textContent = value;
          td.style.cssText = 'padding:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border-bottom:1px solid #303030;';
          rowCells.push(td);
          tr.appendChild(td);
        }
        rowCells[1].title = row.name;
        rowCells[2].style.fontVariantNumeric = rowCells[3].style.fontVariantNumeric = 'tabular-nums';
        body.appendChild(tr);
        cells.push(rowCells);
      }
    }
    rows.forEach((row, index) => {
      cells[index][2].textContent = angleText(row.deg);
      cells[index][3].textContent = angleText(row.rad);
      cells[index][2].title = row.deg === null ? '' : String(row.deg);
      cells[index][3].title = row.rad === null ? '' : String(row.rad);
    });
  }

  return {
    element: details,
    update(data) {
      if (disposed) return;
      clearTimeout(timer);
      const rows = jointAngleRows(data);
      render(rows, data?.status?.available === false ? '数据不可用' : '暂无数据');
      if (rows.length) timer = setTimeout(() => render([], '数据已暂停'), staleMs);
    },
    dispose() { disposed = true; clearTimeout(timer); details.remove(); },
  };
}
