/**
 * benchmark.js — 用例库、起跑、看分数、看本机历史。
 *
 * 主体是**看现有的方案**：本机一份可编辑的用例库，加上市场里带 `test` 段的方案。
 * 打开一个用例文件是导入路径，不是入口 —— 它在标签旁边，是个小链接。
 *
 * 这块只回答一个问题：**分数动没动，为什么。** 所以历史里的分数不是表格里的数字，
 * 而是共享 0-100 轴上的一个刻度：标准差是须线，样本量印在刻度旁，模型与镜像 tag
 * 作为这一行的身份。「动没动」由形状回答，不用在四个列之间心算。
 *
 * 这一块是「单次跑的过程可视化」之外的另一个需求：已有的面板（画布、活动流、
 * performance、history）能看一趟跑得怎么样，但看不了「有哪些用例、跑哪个、
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
import { initEditor, openEditor } from './benchmark-editor.js';
import { initTimeline, openTimeline } from './benchmark-timeline.js';

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

export function initBenchmark() {
  initEditor(_loadLibrary);
  initTimeline();
  document.getElementById('bm-case-file')?.addEventListener('change', _pickCaseFile);
  document.getElementById('bm-case-new')?.addEventListener('click', _newCase);
  document.querySelectorAll('.bm-lib-tab').forEach((tab) => {
    tab.addEventListener('click', () => _switchTab(tab.dataset.lib));
  });
  document.getElementById('benchmark-close')?.addEventListener('click', _close);
  document.getElementById('btn-benchmark')?.addEventListener('click', _open);
  document.getElementById('bm-abort')?.addEventListener('click', _abort);
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
  await Promise.all([_loadLibrary(), _loadRuns()]);
  await _poll();
}

// ── 用例库 ───────────────────────────────────────────────────────────────────
//
// 主体是**看现有的方案**：本机一份可编辑的库，加上市场里带 `test` 段的方案。
// 「导入文件」还在，但它是导入路径，不是入口。
//
// 用例自带画布，所以跑一个还没载入的用例会覆盖当前这张 —— 那条路照旧：先给退路
// （把当前画布存下来），再要明确的勾选。

let _cases = [];
let _tab = 'local';

async function _loadLibrary() {
  if (_tab === 'market') return _loadMarketCases();
  const el = document.getElementById('bm-lib-local');
  if (!el) return;
  try {
    _cases = (await api('/api/benchmark/cases')).cases || [];
  } catch { _cases = []; }

  if (!_cases.length) {
    // 空状态是行动邀请，不是句号。
    el.innerHTML = `<div class="bm-empty">本机还没有用例。<br>
      从「市场」收一个，或者「新建」一个从头写。</div>`;
    return;
  }
  el.innerHTML = _cases.map(_caseCard).join('');
  _bindCaseCards(el);
}

function _caseCard(c) {
  // 地图名不进这一行 —— 扫列表时它不影响选哪个用例，而它一挤，站数和插话数就被
  // 省略号吃掉了。地图在编辑器里看。
  const bits = [
    c.waypoints ? `${c.waypoints} 站` : '',
    c.injections ? `${c.injections} 处插话` : '',
  ].filter(Boolean).join(' · ');
  // 分数贴着用例本身 ——「这个用例现在多少分」是打开面板的第一个问题，
  // 让它只活在右栏历史里，等于要人自己把两边对起来。
  const last = c.last
    ? `<span class="bm-card-score">${c.last.score_total ?? '—'}${
        c.last.score_stdev != null ? ` ±${c.last.score_stdev}` : ''
      }<span class="bm-n">n=${c.last.n_repeats}</span></span>`
    : '<span class="bm-card-score bm-card-score--none">还没运行过</span>';
  return `
    <div class="bm-card${c.isLoaded ? ' bm-card--loaded' : ''}" data-id="${_esc(c.id)}">
      <div class="bm-card-head">
        <span class="bm-card-name">${_esc(c.name)}</span>
        ${last}
      </div>
      <div class="bm-card-prompt">“${_esc(c.prompt || '还没写初始指令')}”</div>
      <div class="bm-card-foot">
        <span class="bm-card-meta">${
          c.isLoaded ? '<i class="bm-card-flag">已在画布上</i>' : ''}${_esc(bits)}</span>
        <span class="bm-card-actions">
          <button class="bm-linkbtn" data-edit="${_esc(c.id)}">编辑</button>
          <button class="bm-linkbtn" data-del="${_esc(c.id)}">删除</button>
          <button class="bm-cardrun" data-run="${_esc(c.id)}">运行</button>
        </span>
      </div>
      ${(c.problems || []).length ? `<ul class="bm-blockers">${
        c.problems.map((p) => `<li>${_esc(p)}</li>`).join('')}</ul>` : ''}
    </div>`;
}

function _bindCaseCards(el) {
  el.querySelectorAll('[data-edit]').forEach((b) => b.addEventListener(
    'click', () => openEditor(b.dataset.edit)));
  el.querySelectorAll('[data-run]').forEach((b) => b.addEventListener(
    'click', () => _startCase(b.dataset.run)));
  el.querySelectorAll('[data-del]').forEach((b) => b.addEventListener('click', async () => {
    const card = _cases.find((c) => c.id === b.dataset.del);
    if (!window.confirm(`删掉用例「${card?.name || ''}」？运行过的分数会留在历史里。`)) return;
    await api(`/api/benchmark/cases/${b.dataset.del}`, { method: 'DELETE' });
    await _loadLibrary();
  }));
}

async function _loadMarketCases() {
  const el = document.getElementById('bm-lib-market');
  if (!el) return;
  el.innerHTML = '<div class="bm-empty">加载中…</div>';
  let data;
  try {
    data = await api('/api/benchmark/cases/market');
  } catch (e) {
    el.innerHTML = `<div class="bm-empty">连不上方案市场：${_esc(e.message || e)}</div>`;
    return;
  }
  const items = data.cases || [];
  if (!items.length) {
    el.innerHTML = data.error
      ? `<div class="bm-empty">连不上方案市场：${_esc(data.error)}</div>`
      : '<div class="bm-empty">市场上还没有带测试用例的方案。</div>';
    return;
  }
  el.innerHTML = items.map((s) => `
    <div class="bm-card">
      <div class="bm-card-head">
        <span class="bm-card-name">${_esc(s.name)}</span>
        <span class="bm-card-meta">v${_esc(s.version || '')} ↓${s.downloads || 0}</span>
      </div>
      <div class="bm-card-prompt">${_esc(s.oneLiner || '')}</div>
      <div class="bm-card-foot">
        <span class="bm-card-meta">${(s.requiredDrivers || [])
          .map((d) => _esc(d.name || d.serverName || '')).join('、')}</span>
        <button class="bm-cardrun" data-grab="${_esc(s.slug)}">收进本机</button>
      </div>
    </div>`).join('');
  el.querySelectorAll('[data-grab]').forEach((b) => b.addEventListener(
    'click', () => _grabFromMarket(b.dataset.grab)));
}

async function _grabFromMarket(slug) {
  try {
    // 市场列表不带包体（几十 KB），要收进本机得单取一次详情。
    const detail = await api(`/api/solutions/market/${slug}`);
    const solution = detail.data || {};
    await api('/api/benchmark/cases', {
      method: 'POST',
      body: JSON.stringify({ payload: solution.payload, name: solution.name,
                             origin: `market:${slug}` }),
    });
    _switchTab('local');
    showToast(`已收进本机：${solution.name || slug}`);
  } catch (e) {
    showToast(`收不进来：${e.message || e}`);
  }
}

function _switchTab(which) {
  _tab = which;
  document.querySelectorAll('.bm-lib-tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.lib === which);
  });
  document.getElementById('bm-lib-local')?.classList.toggle('hidden', which !== 'local');
  document.getElementById('bm-lib-market')?.classList.toggle('hidden', which !== 'market');
  _loadLibrary();
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

// ── 跑一个用例：没载入就先过覆盖确认 ─────────────────────────────────────────

async function _startCase(caseId) {
  const card = _cases.find((c) => c.id === caseId);
  const repeats = Math.max(1, parseInt(document.getElementById('bm-repeats')?.value || '1', 10));

  let pre;
  try {
    pre = (await api(`/api/benchmark/cases/${caseId}/preflight`, {
      method: 'POST', body: JSON.stringify({}),
    })).data;
  } catch (e) {
    showToast(`检查失败：${e.message || e}`); return;
  }

  const rows = blockerRows(pre?.test?.problems || [], pre?.test?.readiness || {});
  (pre?.devices?.missing || []).forEach((d) => rows.unshift(
    `缺驱动 <b>${_esc(d.serverName || d.name || '')}</b>　先装上并启动它`));
  if (rows.length) { _showBlockers(card?.name || '这个用例', rows); return; }

  if (card?.isLoaded) { await _runCase(repeats); return; }
  _confirmOverwrite(caseId, card, pre, repeats);
}

function _showBlockers(name, rows) {
  const el = document.getElementById('bm-progress');
  if (!el) return;
  el.innerHTML = `<div class="bm-blocked">
    <b>${_esc(name)}</b> 暂时无法运行：
    <ul class="bm-blockers">${rows.map((r) => `<li>${r}</li>`).join('')}</ul>
  </div>`;
}

function _confirmOverwrite(caseId, card, pre, repeats) {
  const el = document.getElementById('bm-progress');
  if (!el) return;
  const mine = pre?.overwrite?.canvas?.cards ?? 0;
  el.innerHTML = `
    <div class="bm-overwrite">
      <p class="bm-note">运行「${_esc(card?.name || '')}」要先把它自带的画布载入进来，
        会替换掉画布上现在的 ${mine} 张卡片。</p>
      <button class="bm-linkbtn" id="bm-save-canvas">先把当前画布存成解决方案</button>
      <label class="bm-confirm"><input type="checkbox" id="bm-confirm-overwrite">
        我知道会覆盖当前画布</label>
      <div class="bm-case-actions">
        <button class="btn-primary" id="bm-case-apply" disabled>载入并运行</button>
        <button class="bm-linkbtn" id="bm-case-cancel">取消</button>
      </div>
    </div>`;
  const confirm = document.getElementById('bm-confirm-overwrite');
  const apply = document.getElementById('bm-case-apply');
  // 覆盖是默认不做的事：勾了才亮。
  confirm?.addEventListener('change', () => { apply.disabled = !confirm.checked; });
  document.getElementById('bm-save-canvas')?.addEventListener('click', saveCanvasSnapshot);
  document.getElementById('bm-case-cancel')?.addEventListener('click', () => { _poll(); });
  apply?.addEventListener('click', async () => {
    try {
      await api(`/api/benchmark/cases/${caseId}/apply`, {
        method: 'POST', body: JSON.stringify({}),
      });
    } catch (e) { showToast(`载入失败：${e.message || e}`); return; }
    await _loadLibrary();
    await _runCase(repeats);
  });
}

export async function saveCanvasSnapshot() {
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
    showToast(`已存下 ${result.cards} 张卡片，运行结束后可以还原`);
  } catch (e) {
    showToast(`存不下来：${e.message || e}`);
  }
}

async function _pickCaseFile(event) {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  let payload;
  try {
    payload = JSON.parse(await file.text());
  } catch { showToast('这个文件不是解决方案包体'); return; }

  try {
    await api('/api/benchmark/cases', {
      method: 'POST',
      body: JSON.stringify({ payload, name: file.name.replace(/\.json$/i, ''),
                             origin: 'file' }),
    });
    _switchTab('local');
    showToast('已导入本机用例库');
  } catch (e) {
    showToast(`导入失败：${e.message || e}`);
  }
}

async function _newCase() {
  try {
    const created = await api('/api/benchmark/cases', {
      method: 'POST', body: JSON.stringify({ name: '新用例', origin: 'blank' }),
    });
    await _loadLibrary();
    openEditor(created.id);
  } catch (e) {
    showToast(`新建失败：${e.message || e}`);
  }
}

async function _runCase(repeats) {
  try {
    const result = await api('/api/benchmark/case/run', {
      method: 'POST', body: JSON.stringify({ repeats, seed: 0 }),
    });
    _runId = result.run_id;
    showToast(`用例开始运行 × ${repeats} 次`);
    _startPolling();
  } catch (e) {
    showToast(runRefusal(e));
  }
}

/** 拒绝跑的理由要说成人话，尤其是安全那条 —— 它不是故障，是它该拦下来。 */
export function runRefusal(error) {
  const detail = error?.detail || error?.message || error;
  if (typeof detail === 'string') return `无法运行：${detail}`;
  if (detail?.unsafe?.length) {
    const names = detail.unsafe.map((u) => `${u.device} 的 ${u.tool}`).join('、');
    return `画布上有真设备会跟着动（${names}）。用例的指令和真指令分不出来 —— ` +
           `先把它们从画布上拿掉。`;
  }
  if (detail?.readiness) return '用例的依赖还不齐，看上面那几条。';
  return `无法运行：${JSON.stringify(detail)}`;
}

async function _abort() {
  try {
    await api('/api/benchmark/case/abort', { method: 'POST' });
    showToast('已中止，已运行结束的那几次留在历史里');
  } catch (e) {
    showToast(`中止失败：${e.message || e}`);
  }
}

function _startPolling() {
  if (_pollTimer) return;      // 已经在轮询了，别把自己重置掉
  _stopPolling();
  // 运行在后台的 asyncio task 上推进，所以进度是轮询而不是推流。
  _pollTimer = setInterval(_poll, 3000);
}

function _stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

/**
 * 进度该显示什么。
 *
 * 跑着但一次 repeat 都还没运行结束，是长程用例的常态 —— 第一趟导览就是好几分钟。只按
 * 「有没有已完成的 case」判断，这段时间里刷新一下页面，面板会说「没有正在进行的
 * 运行」，而机器人正在走。真机上就是这么露出来的。
 */
export function progressView(data) {
  const live = ['starting', 'running'].includes(data?.state);
  const cases = data?.cases || [];
  return {
    live,
    show: !!data && !data.error && (live || cases.length > 0),
    done: cases.filter((c) => c.outcome !== 'pending').length,
    total: data?.repeats ?? cases.length,
  };
}

async function _poll() {
  const el = document.getElementById('bm-progress');
  if (!el) return;
  let data;
  try {
    data = await api('/api/benchmark/case/progress');
  } catch { return; }

  const view = progressView(data);
  if (view.live) _startPolling();

  if (!view.show) {
    // 空闲时不报「0 / 0 个 case」——那是噪音，不是信息。
    el.innerHTML = '<div class="bm-empty">当前没有运行中的测试。</div>';
    return;
  }
  const cases = data.cases || [];
  const done = view.done;
  el.innerHTML = `
    <div class="bm-progress-head">
      <span class="bm-state bm-state--${_esc(data.state)}">${_stateLabel(data.state)}</span>
      <span><b>${done}</b> / ${view.total} 次</span>
      ${data.mean != null ? `<span>均分 <b>${data.mean}</b>${
        data.stdev != null ? ` ±${data.stdev}` : ''}<span class="bm-n">n=${data.n}</span></span>` : ''}
    </div>
    ${cases.map((c) => {
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
    _runId = null;
    await Promise.all([_loadRuns(), _loadLibrary()]);
    await _offerRestore();
  }
}

async function _offerRestore() {
  const el = document.getElementById('bm-progress');
  try {
    const info = await api('/api/benchmark/snapshot');
    if (!info.saved || !el) return;
    // 提示，不自动还原：运行结束把画布换回去，会在用户正看着结果的时候把画布抽走。
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
    <div class="bm-run-row" data-run="${_esc(r.id)}" title="点开查看这次运行的详情">
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

  el.querySelectorAll('[data-run]').forEach((row) => {
    row.addEventListener('click', () => openTimeline(row.dataset.run));
  });
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
