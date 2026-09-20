/**
 * benchmark.js — 浏览场景、勾选批次、起跑、看分数、看本机历史。
 *
 * 这块只回答一个问题：**分数动没动，为什么。** 所以历史里的分数不是表格里的数字，
 * 而是共享 0-100 轴上的一个刻度：标准差是须线，样本量印在刻度旁，模型与镜像 tag
 * 作为这一行的身份。「动没动」由形状回答，不用在四个列之间心算。
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
    try { detail = (await response.json()).detail ?? detail; } catch { /* keep */ }
    // 拒绝的理由有时是结构化的（缺什么、哪张卡不安全）。`new Error(obj)` 会把它
    // 变成 "[object Object]"，正好丢掉的就是要说给人听的那部分。
    const error = new Error(typeof detail === 'string' ? detail : response.statusText);
    error.detail = detail;
    throw error;
  }
  return response.json();
}

let _pollTimer = null;
let _runId = null;
let _mode = 'suite';         // 'case' 的进度在另一个端点上
let _scenarios = [];

export function initBenchmark() {
  document.getElementById('bm-case-file')?.addEventListener('change', _pickCaseFile);
  document.getElementById('benchmark-close')?.addEventListener('click', _close);
  document.getElementById('btn-benchmark')?.addEventListener('click', _open);
  document.getElementById('bm-run')?.addEventListener('click', _run);
  document.getElementById('bm-abort')?.addEventListener('click', _abort);
  document.getElementById('bm-refresh')?.addEventListener('click', _load);
  document.getElementById('bm-repeats')?.addEventListener('input', _repeatsNote);
  _repeatsNote();
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
    // 被测配置常驻标题栏：分数属于某一次具体的模型与镜像，不该藏在某一行里。
    const env = info.environment || {};
    const meta = document.getElementById('bm-env');
    if (meta) {
      meta.innerHTML = [
        env.llm_model && `模型 <b>${_esc(env.llm_model)}</b>`,
        env.tier && `层 <b>${_esc(env.tier)}</b>`,
        env.host && `机器 <b>${_esc(env.host)}</b>`,
      ].filter(Boolean).join('');
    }
  } catch { /* older backend: leave it hidden */ }
}

function _open() {
  document.getElementById('benchmark-overlay')?.classList.remove('hidden');
  _load();
}

function _repeatsNote() {
  const el = document.getElementById('bm-repeats-note');
  if (!el) return;
  const n = Math.max(1, parseInt(document.getElementById('bm-repeats')?.value || '1', 10));
  // LLM 是随机的，n=1 是一次抛硬币。只在真的是 1 的时候提醒，别每次都念。
  el.textContent = n === 1 ? '跑一次得到的是一个样本，不是结论。' : '';
}

function _close() {
  document.getElementById('benchmark-overlay')?.classList.add('hidden');
  _stopPolling();
}

async function _load() {
  await Promise.all([_loadCase(), _loadScenarios(), _loadRuns()]);
  await _poll();
}

// ── 用例 ─────────────────────────────────────────────────────────────────────
//
// 用例是一个带 `test` 段的解决方案，所以它自带画布 —— 载入会覆盖用户现在这张。
// 这一段的三件事全都围绕那一次覆盖：先给退路（存成文件），再要明确的勾选，最后
// 把「为什么还跑不了」按层说清楚，而不是给一个灰掉的按钮。

let _pending = null;      // 已读入但还没载入的用例包体

async function _loadCase() {
  try {
    const data = await api('/api/benchmark/case');
    _renderCase(data.case ? { loaded: data } : null);
  } catch { _renderCase(null); }
}

function _renderCase(view) {
  const el = document.getElementById('bm-case');
  if (!el) return;

  if (_pending) { el.innerHTML = _pendingHtml(); _bindPending(); return; }
  if (!view?.loaded?.case) {
    el.innerHTML = `<div class="bm-empty">还没有载入用例。<br>
      用例是一个带 <code>test</code> 段的解决方案，打开一个用例文件就能跑。</div>`;
    return;
  }

  const { case: block, problems = [], readiness = {} } = view.loaded;
  el.innerHTML = `
    <div class="bm-case-loaded">
      <div class="bm-case-title">${_esc(block.name || '当前用例')}</div>
      <div class="bm-case-prompt">“${_esc((block.run || {}).prompt || '')}”</div>
      ${_blockers(problems, readiness)}
    </div>`;
}

/**
 * 按层报出还差什么，顺序即修复顺序：驱动 → 资产 → 用例本身。
 *
 * 每一条的下一步动作都不一样 —— 缺驱动去装驱动，缺地图去放地图，用例写坏了去改
 * 用例。合成一句「不能跑」，这些全丢了，剩下一个灰掉的按钮。
 */
export function blockerRows(problems = [], readiness = {}) {
  const rows = [];
  (readiness.missing_drivers || []).forEach((d) => rows.push(
    `缺驱动 <b>${_esc(d)}</b>　先装上并启动它`));
  (readiness.missing_assets || []).forEach((a) => rows.push(
    `缺地图 <b>${_esc(a)}</b>　放进仿真器的 maps 目录`));
  problems.forEach((p) => rows.push(`用例本身：${_esc(p)}`));
  // 问不到 ≠ 缺。说成「缺」会把一次临时故障说成装错了东西。
  if (readiness.assets_error) rows.push(`问不到仿真器的地图列表（${_esc(readiness.assets_error)}）`);
  return rows;
}

function _blockers(problems, readiness) {
  const rows = blockerRows(problems, readiness);
  if (!rows.length) return '<div class="bm-case-ok">依赖齐了，可以跑。</div>';
  return `<ul class="bm-blockers">${rows.map((r) => `<li>${r}</li>`).join('')}</ul>`;
}

function _pendingHtml() {
  const test = _pending.payload.test || {};
  const cards = ((_pending.payload.canvas || {}).cards || []).length;
  const missing = _pending.preflight?.devices?.missing || [];
  const overwrite = _pending.preflight?.overwrite?.canvas || {};
  return `
    <div class="bm-case-loaded">
      <div class="bm-case-title">${_esc(_pending.name)}</div>
      <div class="bm-case-prompt">“${_esc((test.run || {}).prompt || '')}”</div>
      ${missing.length ? `<ul class="bm-blockers">${missing.map((d) => `
        <li>缺驱动 <b>${_esc(d.serverName || d.name || '')}</b>　先装上并启动它</li>`).join('')}</ul>`
        : `<p class="bm-note">这个用例自带 ${cards} 张卡片，载入会替换掉画布上现在的
             ${overwrite.cards ?? 0} 张。</p>`}
      <button class="bm-linkbtn" id="bm-save-canvas">先把当前画布存成解决方案</button>
      <label class="bm-confirm"><input type="checkbox" id="bm-confirm-overwrite">
        我知道会覆盖当前画布</label>
      <div class="bm-case-actions">
        <button class="btn-primary" id="bm-case-apply" disabled>载入用例</button>
        <button class="bm-linkbtn" id="bm-case-cancel">取消</button>
      </div>
    </div>`;
}

function _bindPending() {
  const confirm = document.getElementById('bm-confirm-overwrite');
  const apply = document.getElementById('bm-case-apply');
  // 覆盖是默认不做的事：勾了才亮。
  confirm?.addEventListener('change', () => { apply.disabled = !confirm.checked; });
  document.getElementById('bm-save-canvas')?.addEventListener('click', _saveCanvas);
  document.getElementById('bm-case-apply')?.addEventListener('click', _applyCase);
  document.getElementById('bm-case-cancel')?.addEventListener('click', () => {
    _pending = null; _loadCase();
  });
}

async function _pickCaseFile(event) {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  let payload;
  try {
    payload = JSON.parse(await file.text());
  } catch { showToast('这个文件不是解决方案包体'); return; }
  if (!payload.test) { showToast('这个解决方案没有 test 段，不是一个测试用例'); return; }

  try {
    const result = await api('/api/solutions/preflight', {
      method: 'POST', body: JSON.stringify({ payload, includes: ['canvas', 'test'] }),
    });
    _pending = { name: file.name.replace(/\.json$/i, ''), payload, preflight: result.data };
    _renderCase(null);
  } catch (e) {
    showToast(`读不了这个用例：${e.message || e}`);
  }
}

async function _saveCanvas() {
  try {
    const result = await api('/api/benchmark/snapshot', { method: 'POST' });
    // 存两份：一份留在机器上（还原用），一份下载（能在别处载入）。
    const blob = new Blob([JSON.stringify(result.payload, null, 1)],
                          { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `canvas-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    showToast(`已存下 ${result.cards} 张卡片，跑完可以还原`);
  } catch (e) {
    showToast(`存不下来：${e.message || e}`);
  }
}

async function _applyCase() {
  try {
    await api('/api/solutions/apply', {
      method: 'POST',
      body: JSON.stringify({ payload: _pending.payload, includes: ['canvas', 'test'],
                             confirm: true }),
    });
    _pending = null;
    await _loadCase();
    showToast('用例已载入，画布已换成它自带的那张');
  } catch (e) {
    showToast(`载入失败：${e.message || e}`);
  }
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
    // 空状态是行动邀请，不是句号：说清楚去哪儿放，人就知道下一步做什么。
    el.innerHTML = `<div class="bm-empty">还没有场景。<br>
      把一个 yaml 放进 <code>/opt/phanthy-motus/data/sim/scenarios</code>，
      再点「重新扫描」。</div>`;
    return;
  }
  el.innerHTML = _scenarios.map((s) => {
    const stops = (s.waypoints || []).length;
    const bits = [stops ? `${stops} 站` : '点位随地图', (s.injections || []).length ? '含打断' : ''];
    return `
    <label class="bm-scenario">
      <input type="checkbox" value="${_esc(s.slug)}" checked>
      <span class="bm-scenario-name">${_esc(s.name || s.slug)}</span>
      <span class="bm-scenario-meta">${bits.filter(Boolean).join('　')}</span>
    </label>`;
  }).join('');
}

function _selected() {
  return Array.from(document.querySelectorAll('#bm-scenarios input:checked'))
    .map((el) => el.value);
}

// ── 起一批 ───────────────────────────────────────────────────────────────────

async function _run() {
  const repeats = Math.max(1, parseInt(document.getElementById('bm-repeats')?.value || '1', 10));
  // 载入了用例就跑用例 —— 它自带画布、指令和判定，比勾几个场景说得更全。
  if (document.querySelector('.bm-case-ok')) { await _runCase(repeats); return; }

  const scenarios = _selected();
  if (!scenarios.length) { showToast('先选至少一个场景'); return; }
  try {
    const result = await api('/api/benchmark/run', {
      method: 'POST',
      body: JSON.stringify({ scenarios, repeats, seed: 0 }),
    });
    _runId = result.run_id;
    _mode = 'suite';
    showToast(`已开始：${scenarios.length} 个场景 × ${repeats} 次`);
    _startPolling();
  } catch (e) {
    showToast(`起批失败：${e.message || e}`);
  }
}

async function _runCase(repeats) {
  try {
    const result = await api('/api/benchmark/case/run', {
      method: 'POST', body: JSON.stringify({ repeats, seed: 0 }),
    });
    _runId = result.run_id;
    _mode = 'case';
    showToast(`用例已开始 × ${repeats} 次`);
    _startPolling();
  } catch (e) {
    showToast(runRefusal(e));
  }
}

/** 拒绝跑的理由要说成人话，尤其是安全那条 —— 它不是故障，是它该拦下来。 */
export function runRefusal(error) {
  const detail = error?.detail || error?.message || error;
  if (typeof detail === 'string') return `跑不了：${detail}`;
  if (detail?.unsafe?.length) {
    const names = detail.unsafe.map((u) => `${u.device} 的 ${u.tool}`).join('、');
    return `画布上有真设备会跟着动（${names}）。用例的指令和真指令分不出来 —— ` +
           `先把它们从画布上拿掉。`;
  }
  if (detail?.readiness) return '用例的依赖还不齐，看上面那几条。';
  return `跑不了：${JSON.stringify(detail)}`;
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
    data = await api(_mode === 'case' ? '/api/benchmark/case/progress'
                                      : '/api/benchmark/progress');
  } catch { return; }

  if (!data || data.error || !data.cases || !data.cases.length) {
    // 空闲时不报「0 / 0 个 case」——那是噪音，不是信息。
    el.innerHTML = '<div class="bm-empty">没有正在进行的批次。</div>';
    return;
  }
  const done = data.cases.filter((c) => c.outcome !== 'pending').length;
  el.innerHTML = `
    <div class="bm-progress-head">
      <span class="bm-state bm-state--${_esc(data.state)}">${_stateLabel(data.state)}</span>
      <span><b>${done}</b> / ${data.cases.length} 个 case</span>
      ${data.mean != null ? `<span>均分 <b>${data.mean}</b>${
        data.stdev != null ? ` ±${data.stdev}` : ''}<span class="bm-n">n=${data.n}</span></span>` : ''}
    </div>
    ${data.cases.map((c) => {
      const score = c.score || {};
      const dims = score.by_dimension || {};
      const order = ['orchestration', 'interruption', 'long_horizon', 'latency', 'safety'];
      const bar = order.map((k) => {
        const v = dims[k];
        const cls = v == null ? '' : (v >= 100 ? ' bm-dim--ok' : ' bm-dim--bad');
        return `<span class="bm-dim${cls}" title="${_esc(_dimLabel(k))}${
          v == null ? '：未测' : `：${v}`}"></span>`;
      }).join('');
      return `
      <div class="bm-case">
        <div class="bm-case-name">
          ${_esc(c.scenario)} <i>#${c.repeat + 1}</i>
          ${(c.failures || []).length ? `<i>· ${(c.failures || []).map(_dimOrName).join('、')}</i>` : ''}
          <div class="bm-dims">${bar}</div>
        </div>
        <div class="bm-case-score">${score.total ?? '—'}</div>
        <div class="bm-case-outcome bm-case-outcome--${_esc(c.outcome)}">${_outcome(c.outcome)}</div>
      </div>`;
    }).join('')}`;

  if (['done', 'aborted', 'error'].includes(data.state)) {
    _stopPolling();
    if (_mode === 'case') { _runId = null; await _loadRuns(); await _offerRestore(); }
    else await _collect();
  }
}

async function _offerRestore() {
  const el = document.getElementById('bm-progress');
  try {
    const info = await api('/api/benchmark/snapshot');
    if (!info.saved || !el) return;
    // 提示，不自动还原：跑完把画布换回去，会在用户正看着结果的时候把画布抽走。
    const row = document.createElement('div');
    row.className = 'bm-restore';
    row.innerHTML = `画布还是用例自带的那张。
      <button class="bm-linkbtn" id="bm-restore">还原成我存下的 ${info.cards} 张卡片</button>`;
    el.appendChild(row);
    row.querySelector('#bm-restore')?.addEventListener('click', async () => {
      try {
        const done = await api('/api/benchmark/snapshot/restore', { method: 'POST' });
        showToast(`画布已还原（${done.cards} 张卡片）`);
        row.remove();
      } catch (e) { showToast(`还原失败：${e.message || e}`); }
    });
  } catch { /* 没存过快照就没有这一步 */ }
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
    el.innerHTML = '<div class="bm-empty">还没有跑过。选好场景，点「开始」。</div>';
    return;
  }
  el.innerHTML = runs.map((r) => {
    const mean = r.score_total;
    const sd = r.score_stdev;
    const at = mean == null ? 0 : Math.max(0, Math.min(100, mean));
    const lo = Math.max(0, at - (sd || 0));
    const hi = Math.min(100, at + (sd || 0));
    const config = [r.llm_model, Object.values(r.image_tags || {}).filter(Boolean).join(' ')]
      .filter(Boolean).join('  ');
    return `
    <div class="bm-run-row">
      <div class="bm-run-id">
        <div class="bm-run-when">${_time(r.started_at)}　${_esc(r.suite)}</div>
        <span class="bm-run-config">${_esc(config) || '未记录配置'}</span>
      </div>
      <div class="bm-axis" title="${mean == null ? '未评分' : `${mean}${sd != null ? ` ±${sd}` : ''}，n=${r.n_repeats}`}">
        <span class="bm-axis-mark" style="left:0"></span>
        <span class="bm-axis-mark" style="left:50%"></span>
        <span class="bm-axis-mark" style="left:calc(100% - 1px)"></span>
        ${sd != null ? `<span class="bm-whisker" style="left:${lo}%;width:${hi - lo}%"></span>` : ''}
        <span class="bm-tick${mean == null ? ' bm-tick--none' : ''}" style="left:${at}%"></span>
        ${mean == null ? '' : `<span class="bm-score${
          at >= 50 ? ' bm-score--end' : ' bm-score--start'
        }" style="left:${at}%">${mean}<span class="bm-n">n=${r.n_repeats}</span></span>`}
      </div>
    </div>`;
  }).join('');
}

const _DIMS = {
  orchestration: '编排', interruption: '打断', long_horizon: '长程',
  latency: '时效', safety: '安全',
};
const _CHECK_DIM = {
  waypoint_order: '编排', announce_after_arrive: '编排', exactly_one_terminal_post: '编排',
  interrupted_leg: '打断', resume_correctness: '长程',
  max_wall_seconds: '时效', never_occupied: '安全',
};
const _OUTCOMES = { ok: '通过', stalled: '卡住', timeout: '超时', error: '出错', pending: '排队中' };
const _STATES = { idle: '空闲', loading: '准备中', running: '进行中', done: '已完成' };

function _dimLabel(k) { return _DIMS[k] || k; }
function _dimOrName(name) { return _CHECK_DIM[name] || name; }
function _outcome(o) { return _OUTCOMES[o] || o; }
function _stateLabel(s) { return _STATES[s] || s; }

// ── helpers ──────────────────────────────────────────────────────────────────

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
