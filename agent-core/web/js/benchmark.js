/**
 * benchmark.js — 浏览场景、勾选批次、起跑、看分数、看本机历史。
 *
 * 这一块是「单次跑的过程可视化」之外的另一个需求：已有的面板（画布、活动流、
 * performance、history）能看一趟跑得怎么样，但看不了「有哪些场景能跑、跑哪些、
 * 每次多少分、分数趋势」。
 *
 * 两条在 UI 上必须守住的规矩：
 *
 *  1. **n 永远和分数一起显示。** LLM 是随机的，一次运行的分数是分布里的一个样本。
 *     只显示数字不显示样本量，n=1 会被当成结论 —— 这是 benchmark 和演示的分界线。
 *  2. **分数旁边必须有被测配置。** 「换了模型 / 改了镜像之后分数动没动」才是真正
 *     有人问的问题；不显示 llm_model 和 tag，两行分数之间没有可比性。
 *
 * 面板只在检测到仿真器时出现 —— 出厂的机器人不该看到一个 Benchmark 标签。
 */

import { showToast } from './toast.js';

// `auth.js` patches window.fetch to attach the Bearer token, so plain fetch is
// already authenticated — there is no shared api() helper in this codebase.
async function api(path, opts) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' }, ...(opts || {}),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return response.json();
}

let _pollTimer = null;
let _runId = null;
let _scenarios = [];

export function initBenchmark() {
  document.getElementById('benchmark-close')?.addEventListener('click', _close);
  document.getElementById('btn-benchmark')?.addEventListener('click', _open);
  document.getElementById('bm-run')?.addEventListener('click', _run);
  document.getElementById('bm-abort')?.addEventListener('click', _abort);
  document.getElementById('bm-refresh')?.addEventListener('click', _load);
  _detect();
}

async function _detect() {
  try {
    const info = await api('/api/benchmark/available');
    // Absent simulator, the entry stays hidden rather than showing a tab that
    // can only ever say "nothing here".
    document.querySelectorAll('[data-target="btn-benchmark"]').forEach((el) => {
      el.classList.toggle('hidden', !info.available);
    });
  } catch { /* older backend: leave it hidden */ }
}

function _open() {
  document.getElementById('benchmark-overlay')?.classList.remove('hidden');
  _load();
}

function _close() {
  document.getElementById('benchmark-overlay')?.classList.add('hidden');
  _stopPolling();
}

async function _load() {
  await Promise.all([_loadScenarios(), _loadRuns()]);
  await _poll();
}

// ── 场景 ─────────────────────────────────────────────────────────────────────

async function _loadScenarios() {
  const el = document.getElementById('bm-scenarios');
  if (!el) return;
  try {
    const data = await api('/api/benchmark/scenarios');
    _scenarios = data.scenarios || [];
  } catch {
    _scenarios = [];
  }
  if (!_scenarios.length) {
    el.innerHTML = '<div class="bm-empty">没有发现可跑的场景</div>';
    return;
  }
  el.innerHTML = _scenarios.map((s) => `
    <label class="bm-scenario">
      <input type="checkbox" value="${_esc(s.slug)}" checked>
      <span class="bm-scenario-name">${_esc(s.name || s.slug)}</span>
      <span class="bm-scenario-meta">${(s.waypoints || []).length} 个航点${
        (s.injections || []).length ? ' · 含打断' : ''}</span>
    </label>`).join('');
}

function _selected() {
  return Array.from(document.querySelectorAll('#bm-scenarios input:checked'))
    .map((el) => el.value);
}

// ── 起一批 ───────────────────────────────────────────────────────────────────

async function _run() {
  const scenarios = _selected();
  if (!scenarios.length) { showToast('先选至少一个场景'); return; }
  const repeats = Math.max(1, parseInt(document.getElementById('bm-repeats')?.value || '1', 10));
  try {
    const result = await api('/api/benchmark/run', {
      method: 'POST',
      body: JSON.stringify({ scenarios, repeats, seed: 0 }),
    });
    _runId = result.run_id;
    showToast(`已开始：${scenarios.length} 个场景 × ${repeats} 次`);
    _startPolling();
  } catch (e) {
    showToast(`起批失败：${e.message || e}`);
  }
}

async function _abort() {
  try {
    await api('/api/benchmark/abort', { method: 'POST' });
    await _collect();
    showToast('已中止，已完成的结果保留');
  } catch (e) {
    showToast(`中止失败：${e.message || e}`);
  }
}

function _startPolling() {
  _stopPolling();
  // The suite runs on the driver, so progress is a poll rather than a stream.
  // sim_report is a `resource`, which is what makes it answerable while a
  // navigation is pending.
  _pollTimer = setInterval(_poll, 3000);
}

function _stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

async function _poll() {
  const el = document.getElementById('bm-progress');
  if (!el) return;
  let data;
  try {
    data = await api('/api/benchmark/progress');
  } catch { return; }

  if (!data || data.error || !data.cases) {
    el.innerHTML = '<div class="bm-empty">没有正在进行的批次</div>';
    return;
  }
  const done = data.cases.filter((c) => c.outcome !== 'pending').length;
  el.innerHTML = `
    <div class="bm-progress-head">
      <span class="bm-state bm-state--${_esc(data.state)}">${_esc(data.state)}</span>
      <span>${done} / ${data.cases.length}</span>
      ${_scoreLine(data.mean, data.stdev, data.n)}
    </div>
    <table class="bm-table">
      <thead><tr><th>场景</th><th>#</th><th>结果</th><th>分数</th><th>用时</th><th>失败项</th></tr></thead>
      <tbody>${data.cases.map((c) => `
        <tr>
          <td>${_esc(c.scenario)}</td>
          <td>${c.repeat}</td>
          <td><span class="bm-outcome bm-outcome--${_esc(c.outcome)}">${_esc(c.outcome)}</span></td>
          <td>${c.score?.total ?? '—'}</td>
          <td>${c.elapsed != null ? `${c.elapsed}s` : '—'}</td>
          <td class="bm-failures">${(c.failures || []).map(_esc).join('、') || '—'}</td>
        </tr>`).join('')}</tbody>
    </table>`;

  if (data.state === 'done') {
    _stopPolling();
    await _collect();
  }
}

async function _collect() {
  if (!_runId) return;
  try {
    await api(`/api/benchmark/runs/${_runId}/collect`, { method: 'POST' });
    _runId = null;
    await _loadRuns();
  } catch { /* the run stays 'running' and can be collected on the next open */ }
}

// ── 历史 ─────────────────────────────────────────────────────────────────────

async function _loadRuns() {
  const el = document.getElementById('bm-runs');
  if (!el) return;
  let runs = [];
  try {
    runs = (await api('/api/benchmark/runs?limit=30')).runs || [];
  } catch { /* leave empty */ }

  if (!runs.length) {
    el.innerHTML = '<div class="bm-empty">还没有跑过</div>';
    return;
  }
  el.innerHTML = `
    <table class="bm-table">
      <thead><tr><th>时间</th><th>套件</th><th>层</th><th>分数</th><th>模型</th><th>镜像</th></tr></thead>
      <tbody>${runs.map((r) => `
        <tr>
          <td>${_time(r.started_at)}</td>
          <td>${_esc(r.suite)}</td>
          <td>${_esc(r.tier)}</td>
          <td>${_scoreLine(r.score_total, r.score_stdev, r.n_repeats)}</td>
          <td class="bm-config">${_esc(r.llm_model) || '—'}</td>
          <td class="bm-config">${_esc(Object.values(r.image_tags || {}).join(' ')) || '—'}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

// ── helpers ──────────────────────────────────────────────────────────────────

function _scoreLine(mean, stdev, n) {
  if (mean == null) return '<span class="bm-score bm-score--none">—</span>';
  // n travels with the number, always. A score without its sample size reads as
  // a conclusion when it is one sample of a distribution.
  const spread = stdev != null ? ` ±${stdev}` : '';
  return `<span class="bm-score">${mean}${spread}</span>` +
         `<span class="bm-n" title="重复次数">n=${n ?? 1}</span>`;
}

function _time(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ` +
         `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function _esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
