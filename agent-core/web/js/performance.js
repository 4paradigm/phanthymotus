/**
 * performance.js — 性能分析 Dashboard（开放 Span 式）
 */

// 按 component 定义色系，同 component 内自动分配
const PALETTE = {
  perception: ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#06b6d4', '#0891b2', '#67e8f9', '#22d3ee'],
  core:       ['#f59e0b', '#d97706', '#b45309', '#92400e', '#64748b', '#475569', '#334155', '#94a3b8'],
  driver:     ['#10b981', '#059669', '#047857', '#065f46', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5'],
};
const FALLBACK_PALETTE = ['#ec4899', '#f97316', '#84cc16', '#14b8a6', '#e879f9', '#fb7185', '#4ade80', '#facc15'];

// 运行时缓存：span name → color
const _colorCache = {};
const _componentCounters = {};

function _spanColor(span, component) {
  if (_colorCache[span]) return _colorCache[span];
  const comp = component || _guessComponent(span);
  const palette = PALETTE[comp] || FALLBACK_PALETTE;
  if (!_componentCounters[comp]) _componentCounters[comp] = 0;
  const idx = _componentCounters[comp]++ % palette.length;
  _colorCache[span] = palette[idx];
  return _colorCache[span];
}

function _guessComponent(span) {
  if (/^(vad|asr|kws|tts)/.test(span)) return 'perception';
  if (/^(llm|event_queue|tool:|finish|turn)/.test(span)) return 'core';
  return 'driver';
}

let _refreshTimer = null;
let _currentRange = '24h';

export function initPerformance() {
  const overlay = document.getElementById('performance-overlay');
  if (!overlay) return;

  document.getElementById('performance-close')?.addEventListener('click', _close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) _close(); });
  document.getElementById('perf-range')?.addEventListener('change', (e) => {
    _currentRange = e.target.value;
    _load();
  });
  document.getElementById('perf-refresh')?.addEventListener('click', _load);
  document.getElementById('perf-clear')?.addEventListener('click', async () => {
    if (!confirm('确定清空所有性能数据？')) return;
    await fetch('/api/performance/clear', { method: 'DELETE' });
    _load();
  });
  document.getElementById('btn-performance')?.addEventListener('click', _open);
}

function _open() {
  document.getElementById('performance-overlay')?.classList.remove('hidden');
  _load();
}

function _close() {
  document.getElementById('performance-overlay')?.classList.add('hidden');
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

function _rangeToTs() {
  const now = Date.now() / 1000;
  const map = { '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800 };
  return { start: now - (map[_currentRange] || 86400), end: 0 };
}

async function _load() {
  const { start, end } = _rangeToTs();
  // 保存展开状态
  const expandedSet = new Set();
  document.querySelectorAll('.perf-row-group.expanded').forEach(el => {
    expandedSet.add(el.dataset.idx);
  });
  try {
    const [latestRes, aggRes] = await Promise.all([
      fetch('/api/performance/latest?n=50'),
      fetch(`/api/performance/aggregate?start=${start}&end=${end}`),
    ]);
    const latest = await latestRes.json();
    const agg = await aggRes.json();
    _renderSummary(agg);
    _renderWaterfall(Array.isArray(latest) ? latest : []);
    // 恢复展开状态
    expandedSet.forEach(idx => {
      document.querySelector(`.perf-row-group[data-idx="${idx}"]`)?.classList.add('expanded');
    });
  } catch (e) {
    console.error('[performance] load error:', e);
  }
}

function _renderSummary(agg) {
  const el = document.getElementById('perf-summary-cards');
  if (!el) return;

  if (!agg || agg.count === 0) {
    el.innerHTML = '<div class="perf-empty">暂无数据</div>';
    return;
  }

  const bySpan = agg.by_span || {};
  const spanNames = Object.keys(bySpan);

  // 计算总体甘特图的时间范围（取所有 span 的 max(offset + duration)）
  const totalRange = Math.max(...spanNames.filter(n => n !== 'turn_total').map(n => (bySpan[n].avg_offset_ms || 0) + bySpan[n].avg_ms), 1);

  // 按 avg_offset_ms 排序（排除 turn_total）
  const sortedSpans = spanNames.filter(n => n !== 'turn_total')
    .sort((a, b) => (bySpan[a].avg_offset_ms || 0) - (bySpan[b].avg_offset_ms || 0));

  const summaryGantt = sortedSpans.map(name => {
    const s = bySpan[name];
    const color = _spanColor(name);
    const left = ((s.avg_offset_ms || 0) / totalRange * 100).toFixed(1);
    const width = Math.max(0.5, (s.avg_ms / totalRange * 100)).toFixed(1);
    return `<div class="perf-gantt-row">
      <span class="perf-gantt-label">${name}</span>
      <div class="perf-gantt-track">
        <div class="perf-gantt-bar" style="left:${left}%;width:${width}%;background:${color}" title="${name}: avg ${_fmtMs(s.avg_ms)} @ offset ${_fmtMs(s.avg_offset_ms)}"></div>
      </div>
      <span class="perf-gantt-dur">${_fmtMs(s.avg_ms)}</span>
    </div>`;
  }).join('');

  el.innerHTML = `
    <div class="perf-cards">
      <div class="perf-card">
        <div class="perf-card-value">${agg.count}</div>
        <div class="perf-card-label">总轮次</div>
      </div>
      ${bySpan['turn_total'] ? `<div class="perf-card">
        <div class="perf-card-value">${_fmtMs(bySpan['turn_total'].avg_ms)}</div>
        <div class="perf-card-label">平均总耗时</div>
      </div>
      <div class="perf-card">
        <div class="perf-card-value">${_fmtMs(bySpan['turn_total'].p95_ms)}</div>
        <div class="perf-card-label">P95 总耗时</div>
      </div>` : ''}
    </div>
    <div class="perf-gantt">${summaryGantt}</div>
    <div class="perf-avg-detail">
      <table class="perf-detail-table">
        <thead><tr><th>阶段</th><th>平均</th><th>P95</th><th>次数</th></tr></thead>
        <tbody>
          ${sortedSpans.map(name => {
            const s = bySpan[name];
            const color = _spanColor(name);
            return `<tr>
              <td><span class="perf-legend-dot" style="background:${color}"></span> ${name}</td>
              <td class="perf-detail-val">${_fmtMs(s.avg_ms)}</td>
              <td class="perf-detail-val">${_fmtMs(s.p95_ms)}</td>
              <td class="perf-detail-val">${s.count}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function _renderWaterfall(turns) {
  const el = document.getElementById('perf-waterfall');
  if (!el) return;

  if (!turns.length) {
    el.innerHTML = '<div class="perf-empty">暂无记录</div>';
    return;
  }

  const rows = turns.map((t, idx) => {
    const spans = (t.spans || []).filter(s => s.span !== 'turn_total');
    const timeStr = new Date(t.created_at * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const text = (t.trigger_text || '').replace(/<[^>]*>/g, '').replace(/\{[^}]*\}/g, '').trim().slice(0, 25);
    const totalMs = t.total_duration_ms || 0;

    // 紧凑摘要：取 top 3 耗时 span
    const topSpans = [...spans].sort((a, b) => (b.duration_ms || 0) - (a.duration_ms || 0)).slice(0, 3);
    const summaryChips = topSpans.filter(s => s.duration_ms > 0).map(s => {
      const color = _spanColor(s.span, s.component);
      return `<span class="perf-chip" style="background:${color}">${s.span.replace(/^tool:/, '')} ${_fmtMs(s.duration_ms)}</span>`;
    }).join('');

    // Gantt chart — each span gets its own row (展开时显示)
    let ganttHtml = '';
    const validSpans = spans.filter(s => s.start_ts && s.start_ts > 1e9 && s.duration_ms > 0);
    if (validSpans.length) {
      const minStart = Math.min(...validSpans.map(s => s.start_ts));
      // 用所有 span 结束时间的最大值作为时间轴范围
      const maxEnd = Math.max(...validSpans.map(s => s.end_ts || (s.start_ts + s.duration_ms / 1000)));
      const rangeSec = maxEnd - minStart;
      if (rangeSec > 0) {
        ganttHtml = validSpans.map(s => {
          const left = Math.min(99, ((s.start_ts - minStart) / rangeSec * 100)).toFixed(1);
          const width = Math.min(100 - parseFloat(left), Math.max(0.5, (s.duration_ms / 1000) / rangeSec * 100)).toFixed(1);
          const color = _spanColor(s.span, s.component);
          return `<div class="perf-gantt-row">
            <span class="perf-gantt-label">${s.span}</span>
            <div class="perf-gantt-track">
              <div class="perf-gantt-bar" style="left:${left}%;width:${width}%;background:${color}" title="${s.span}: ${_fmtMs(s.duration_ms)}"></div>
            </div>
            <span class="perf-gantt-dur">${_fmtMs(s.duration_ms)}</span>
          </div>`;
        }).join('');
      }
    }

    return `
      <div class="perf-row-group" data-idx="${idx}">
        <div class="perf-row-header" onclick="this.parentElement.classList.toggle('expanded')">
          <span class="perf-row-time">${timeStr}</span>
          <span class="perf-row-text" title="${t.trigger_text || ''}">${text}</span>
          <span class="perf-row-chips">${summaryChips}</span>
          <span class="perf-row-total">${_fmtMs(totalMs)}</span>
          <span class="perf-row-expand">▸</span>
        </div>
        <div class="perf-row-detail">
          <div class="perf-gantt">${ganttHtml}</div>
        </div>
      </div>
    `;
  }).join('');

  el.innerHTML = `
    <h3 class="perf-section-title">最近请求 (${turns.length})</h3>
    <div class="perf-waterfall-list">${rows}</div>
  `;
}

function _fmtMs(ms) {
  if (ms == null) return '-';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
