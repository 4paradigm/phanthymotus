/**
 * usage.js — Token 用量统计 Modal
 */

let _overlay, _rangeSelect;

export function initUsage() {
  _overlay = document.getElementById('usage-overlay');
  if (!_overlay) return;

  _rangeSelect = document.getElementById('usage-range');

  document.getElementById('btn-usage')?.addEventListener('click', _open);
  document.getElementById('usage-close')?.addEventListener('click', _close);
  _overlay.addEventListener('click', e => { if (e.target === _overlay) _close(); });
  _rangeSelect?.addEventListener('change', () => _load());

  document.querySelector('[data-action="usage"]')?.addEventListener('click', () => {
    document.getElementById('btn-usage')?.click();
  });
}

function _open() {
  _overlay.classList.remove('hidden');
  _load();
}

function _close() {
  _overlay.classList.add('hidden');
}

async function _load() {
  const range = _rangeSelect?.value || '7d';
  const cards = document.getElementById('usage-summary-cards');
  const chart = document.getElementById('usage-daily-chart');
  const table = document.getElementById('usage-daily-table');

  cards.innerHTML = '<div class="usage-loading">加载中…</div>';
  chart.innerHTML = '';
  table.innerHTML = '';

  try {
    const res = await fetch(`/api/performance/usage?range=${range}`);
    const data = await res.json();
    _renderCards(cards, data.summary);
    _renderChart(chart, data.breakdown, data.granularity);
    _renderTable(table, data.breakdown, data.granularity);
  } catch (e) {
    cards.innerHTML = '<div class="usage-loading" style="color:var(--red)">加载失败</div>';
  }
}

function _renderCards(el, summary) {
  const hasData = summary.total_tokens > 0;
  const items = [
    { label: '输入', value: summary.prompt_tokens, color: '#3b82f6' },
    { label: '输出', value: summary.completion_tokens, color: '#10b981' },
    { label: '缓存', value: summary.cached_tokens, color: '#8b5cf6' },
  ];

  el.innerHTML = `
    <div class="usage-cards-grid">
      ${items.map(it => `
        <div class="usage-card">
          <div class="usage-card-indicator" style="background:${it.color}"></div>
          <div class="usage-card-content">
            <div class="usage-card-value">${_fmt(it.value)}</div>
            <div class="usage-card-label">${it.label}</div>
          </div>
        </div>
      `).join('')}
    </div>
    <div class="usage-total">
      <span class="usage-total-value">${_fmt(summary.total_tokens)}</span>
      <span class="usage-total-label">总 tokens · ${summary.call_count} 次调用</span>
    </div>
  `;
}

function _renderChart(el, breakdown, granularity) {
  if (!breakdown || !breakdown.length) {
    el.innerHTML = `
      <div class="usage-empty">
        <div class="usage-empty-icon">📊</div>
        <div class="usage-empty-text">当前周期内暂无用量数据</div>
        <div class="usage-empty-hint">发送消息后将自动记录 token 消耗</div>
      </div>`;
    return;
  }

  const days = [...breakdown].reverse();
  const maxVal = Math.max(...days.map(d => d.prompt_tokens + d.completion_tokens), 1);

  // Generate Y-axis reference lines (3-4 lines)
  const yLines = _calcYLines(maxVal);

  const bars = days.map(d => {
    const total = d.prompt_tokens + d.completion_tokens;
    const h = Math.max((total / maxVal) * 100, 3);
    const uncached = Math.max(d.prompt_tokens - (d.cached_tokens || 0), 0);
    const cached = d.cached_tokens || 0;
    const comp = d.completion_tokens || 0;
    // Format label based on granularity
    let dateLabel;
    if (granularity === 'hourly') {
      // d.date is "2026-07-29 14" → show "14:00" or "07-29 14h"
      const parts = d.date.split(' ');
      dateLabel = parts.length > 1 ? parts[1] + ':00' : d.date;
    } else {
      dateLabel = d.date.slice(5); // MM-DD
    }
    return `
      <div class="usage-bar" title="${d.date}\n非缓存输入: ${_fmt(uncached)}\n缓存输入: ${_fmt(cached)}\n输出: ${_fmt(comp)}">
        <div class="usage-bar-track">
          <div class="usage-bar-fill" style="height:${h}%">
            ${comp ? `<div class="usage-bar-segment completion" style="flex-grow:${comp}"></div>` : ''}
            ${cached ? `<div class="usage-bar-segment cached" style="flex-grow:${cached}"></div>` : ''}
            ${uncached ? `<div class="usage-bar-segment prompt" style="flex-grow:${uncached}"></div>` : ''}
          </div>
        </div>
        <span class="usage-bar-date">${dateLabel}</span>
      </div>`;
  }).join('');

  const yLinesHtml = yLines.map(line => `
    <div class="usage-y-line" style="bottom:${(line.value / maxVal) * 100}%">
      <span class="usage-y-label">${line.label}</span>
    </div>`).join('');

  const titleText = granularity === 'hourly' ? '每小时用量' : '每日用量';
  el.innerHTML = `
    <div class="usage-chart-header">
      <span class="usage-chart-title">${titleText}</span>
      <div class="usage-chart-legend">
        <span class="usage-legend-item"><i style="background:#3b82f6"></i>非缓存输入</span>
        <span class="usage-legend-item"><i style="background:#93c5fd"></i>缓存输入</span>
        <span class="usage-legend-item"><i style="background:#10b981"></i>输出</span>
      </div>
    </div>
    <div class="usage-chart-wrap">
      <div class="usage-y-lines">${yLinesHtml}</div>
      <div class="usage-chart">${bars}</div>
    </div>`;
}

/** Calculate nice Y-axis reference values. */
function _calcYLines(maxVal) {
  if (maxVal <= 0) return [];
  // Find a nice step (1, 2, 5 × 10^n)
  const rough = maxVal / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step;
  if (rough / mag < 1.5) step = mag;
  else if (rough / mag < 3.5) step = mag * 2;
  else if (rough / mag < 7.5) step = mag * 5;
  else step = mag * 10;

  const lines = [];
  for (let v = step; v < maxVal; v += step) {
    lines.push({ value: v, label: _fmt(v) });
  }
  return lines;
}

function _renderTable(el, breakdown, granularity) {
  if (!breakdown || !breakdown.length) { el.innerHTML = ''; return; }

  const PAGE_SIZE = 20;
  let page = 0;
  const totalPages = Math.ceil(breakdown.length / PAGE_SIZE);

  function render() {
    const start = page * PAGE_SIZE;
    const slice = breakdown.slice(start, start + PAGE_SIZE);

    const rows = slice.map(d => {
      let label;
      if (granularity === 'hourly') {
        const parts = d.date.split(' ');
        label = parts.length > 1 ? `${parts[0].slice(5)} ${parts[1]}:00` : d.date;
      } else {
        label = d.date;
      }
      return `
      <div class="usage-row">
        <span class="usage-row-date">${label}</span>
        <div class="usage-row-bars">
          <span class="usage-row-tag prompt">${_fmt(d.prompt_tokens)}</span>
          <span class="usage-row-tag completion">${_fmt(d.completion_tokens)}</span>
          <span class="usage-row-tag cached">${_fmt(d.cached_tokens)}</span>
        </div>
      </div>`;
    }).join('');

    let paginationHtml = '';
    if (totalPages > 1) {
      paginationHtml = `
        <div class="usage-pagination">
          <button class="usage-page-btn" data-dir="prev" ${page === 0 ? 'disabled' : ''}>‹</button>
          <span class="usage-page-info">${page + 1} / ${totalPages}</span>
          <button class="usage-page-btn" data-dir="next" ${page >= totalPages - 1 ? 'disabled' : ''}>›</button>
        </div>`;
    }

    el.innerHTML = `<div class="usage-table">${rows}</div>${paginationHtml}`;

    // Bind pagination buttons
    el.querySelectorAll('.usage-page-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.dir === 'prev' && page > 0) { page--; render(); }
        if (btn.dataset.dir === 'next' && page < totalPages - 1) { page++; render(); }
      });
    });
  }

  render();
}

function _fmt(n) {
  if (n == null || n === 0) return '0';
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'G';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 10_000) return (n / 1_000).toFixed(1) + 'K';
  if (n >= 1_000) return (n / 1_000).toFixed(2) + 'K';
  return String(n);
}
