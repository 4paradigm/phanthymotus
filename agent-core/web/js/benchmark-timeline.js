/**
 * benchmark-timeline.js — 一次运行的详情。
 *
 * 分数只回答「好不好」。要回答「哪儿坏了」，得看这一趟里说了什么、调了什么、世界
 * 怎么回应的 —— 而且要能把它们对上：那句插话到底落在哪一段路上，讲解是在到达之前
 * 还是之后。
 *
 * ## 一条共享时间轴，两条泳道
 *
 * 一开始做成了左右两栏各自排版，理由是「两个时钟不该合成一条」。真机上核对下来，
 * 这个顾虑是多余的而代价是实在的：调用时刻本来就是对的（`navigate` 与 `出发` 差
 * 0.0–0.1 秒），但同一秒在两栏的**高度完全不同**，读的人得拿眼睛找数字 —— 等于
 * 没对齐。
 *
 * 所以改成一条按时间排的流：左泳道是 agent 请求做什么，右泳道是世界真的做了什么，
 * 同一秒必然在同一行。两边都锚在本次运行开始，速率一致（仿真器在真机上按实时走）。
 * 哪天用例开了加速倍率，这个前提就不成立，那时要在界面上说清楚，而不是让人以为是
 * 延迟。
 *
 * **左右不总是成对的，那不是缺陷。** agent 在 +69.1s 请求讲解，世界在 +86.3s 才
 * 开讲 —— 中间那 17 秒是 barrier 在等导航结束。这个差本身就是要看的东西。
 *
 * ## 为什么这些数据存得下来
 *
 * 仿真器的世界下一次运行一开始就被重置，会话历史会被压缩、还会被继续改写。所以
 * 记录在运行结束时就定格进了 `benchmark_case.facts` 与 `benchmark_run.agent_track`。
 * 会话没了就说没了，不编。
 */

// 没过的那几条现在本来就是人话 —— 要么是用户写的要求，要么是默认目标的标签
// （「到达之前不开讲」「平均懵逼时长不超过 10 秒」）。原先这里有一张把
// `waypoint_order` 翻译成「站序不对」的表，那是断言名还是代码标识符时的事。

const DIMS = {
  world_timing: '物理世界时序性', concurrency: '同步执行效率',
  llm_latency: 'LLM 延时', cache_hit: 'cache 命中',
  answer_quality: '回答效果', ux: '用户体验', physical_safety: '安全',
};

/** 「分数为什么是这个」—— 每一项的判定、理由，以及算出来的那些数。
 *
 * 这一段原先不存在：跑完只留下失败项的名字，理由、裁判的逐步比对、七条原则的指标
 * 全都只活在内存里。于是这个弹窗能回答「好不好」，回答不了「哪儿坏了」，而后者才是
 * 打开它的理由。
 */
export function scorecard(c) {
  const items = c.results || [];
  if (!items.length) return (c.assertions || []).length
    ? `<div class="bm-tl-fail">没过：${(c.assertions || []).map(_esc).join('、')}</div>` : '';

  const byDim = {};
  items.forEach((i) => (byDim[i.dimension || 'answer_quality'] ||= []).push(i));
  const seen = c.observations || {};

  return `<div class="bm-sc">${Object.keys(DIMS).filter((k) => byDim[k] || seen[k])
    .map((k) => {
      const group = byDim[k] || [];
      const graded = group.filter((i) => i.measurable !== false);
      const score = graded.length
        ? Math.round(100 * graded.filter((i) => i.ok).reduce((a, i) => a + (i.weight || 0), 0)
            / (graded.reduce((a, i) => a + (i.weight || 0), 0) || 1))
        : null;
      return `<div class="bm-sc-dim">
        <div class="bm-sc-head">
          <b>${_esc(DIMS[k])}</b>
          <span class="bm-sc-num${score == null ? ' bm-sc-num--none' : ''}">${
            score == null ? '判不了' : score}</span>
          <span class="bm-sc-metrics">${_esc(metricLine(seen[k]))}</span>
        </div>
        ${group.map(scoreItem).join('')}
      </div>`;
    }).join('')}</div>`;
}

/** 一条判定：结论、它是算出来的还是判出来的、以及理由。 */
export function scoreItem(i) {
  const done = i.measurable !== false;
  const ratio = i.kind === 'ratio';
  const mark = !done ? '—' : (ratio ? `${Math.round((i.credit ?? 0) * 100)}` : (i.ok ? '✓' : '✗'));
  const cls = !done ? 'none' : (ratio ? 'ratio' : (i.ok ? 'ok' : 'bad'));
  // **标签要说实际发生了什么，不是它属于哪一类。** 原先按 kind 打，于是一条没算成的
  // 目标也写着「算出来的」—— 而它旁边正写着「判不了」的理由，两句直接打架。
  const from = !done ? '没算成'
    : ratio ? '按比例计分'
    : i.kind === 'target' ? '算出来的' : '裁判判的';
  return `<div class="bm-sc-item bm-sc-item--${cls}">
    <span class="bm-sc-mark">${mark}</span>
    <span class="bm-sc-text">${_esc(i.text || '')}
      <i class="bm-sc-from">${from}${i.weight ? ` · 权重 ${i.weight}` : ''}</i></span>
    ${i.detail ? `<span class="bm-clip bm-sc-why">${_esc(i.detail)}</span>` : ''}
    ${(i.steps || []).length ? `<div class="bm-sc-steps">${(i.steps || []).map((s) => `
      <div class="bm-sc-step${s.match ? '' : ' bm-sc-step--off'}">
        <span>${s.match ? '符合' : '偏离'}</span>
        <span>${_esc(s.step || '')}</span>
        <span class="bm-clip">${_esc(s.note || '')}</span>
      </div>`).join('')}</div>` : ''}
  </div>`;
}

/** 那条原则算出来的几个数。判分会抖，这些不会 —— 所以它们要一直在。 */
export function metricLine(block) {
  if (!block || typeof block !== 'object') return '';
  return Object.entries(block)
    .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
    .filter(([, v]) => v !== '')
    .slice(0, 5)
    .map(([k, v]) => `${k} ${typeof v === 'number' ? Math.round(v * 100) / 100 : v}`)
    .join('　');
}

// 一次运行里最值钱的线索往往是「这里什么都没发生」—— 机器人卡住、LLM 空转、
// barrier 等超时，长得都一样：一段静默。
//
// 门槛压到 1.5 秒：排查的时候要能看见**时间去哪儿了**。每行虽然有 +Xs，但那要人
// 自己做减法，一屏十几行减下来，小停顿根本不会被注意到。1.5 秒以下当作连续，
// 再往下标注本身就比它描述的停顿还长。
const QUIET_SECONDS = 1.5;
// 超过这个长度的静默换一种显眼的样子：短停顿是节奏，长静默是事故。
const QUIET_LOUD = 10;

const KINDS = {
  scenario_load: '世界', scenario_start: '世界', scenario_stop: '世界',
  note: '用例', nav_start: '出发', arrive: '到达', nav_cancelled: '放弃',
  nav_failed: '失败', speak_start: '开讲', speak_end: '讲完',
  speech_interrupt: '打断',
};

export function initTimeline() {
  document.getElementById('bm-timeline-close')?.addEventListener('click', close);
  // 省略号里的东西常常正是要看的（完整的讲解词、整串参数）。点一下展开，
  // 不用跳去翻 docker logs。
  document.getElementById('bm-timeline-body')?.addEventListener('click', (event) => {
    const tab = event.target.closest('.bm-tl-tab');
    if (tab) return _switchPane(tab.dataset.pane);
    const clip = event.target.closest('.bm-clip');
    if (clip) clip.classList.toggle('bm-clip--open');
  });
}

export async function openTimeline(runId) {
  const overlay = document.getElementById('bm-timeline-overlay');
  const body = document.getElementById('bm-timeline-body');
  if (!overlay || !body) return;
  overlay.classList.remove('hidden');
  body.innerHTML = '<div class="bm-empty">读取中…</div>';

  let data;
  try {
    const response = await fetch(`/api/benchmark/runs/${runId}/timeline`);
    if (!response.ok) throw new Error((await response.json()).detail || '读不到');
    data = await response.json();
  } catch (e) {
    body.innerHTML = `<div class="bm-empty">读不到这次运行：${_esc(e.message || e)}</div>`;
    return;
  }
  body.innerHTML = _render(data);
}

function _switchPane(which) {
  document.querySelectorAll('.bm-tl-tab').forEach((t) => t.classList.toggle(
    'active', t.dataset.pane === which));
  document.querySelectorAll('.bm-tl-pane').forEach((p) => p.classList.toggle(
    'hidden', p.dataset.pane !== which));
}

function close() {
  document.getElementById('bm-timeline-overlay')?.classList.add('hidden');
}

// ── 合流 ─────────────────────────────────────────────────────────────────────

/**
 * 把两侧摊平成一条按时间排的流。
 *
 * 时间不详的条目只在**它自己那一侧**往前继承：一轮配不上 spans 时，它离哪条 agent
 * 记录最近是知道的，离哪条世界事件最近则不知道。跨侧去借时刻，等于在两条流之间编
 * 一个并不存在的对应关系 —— 而那正是这个视图最容易被当真的假象。
 *
 * 一侧的第一条就没有时间，就让它保持没有：排在最前面、不显示秒数，比塞一个 0 诚实。
 */
export function merge(agent = [], world = []) {
  const agentRows = [];
  agent.forEach((turn) => {
    agentRows.push({ side: 'agent', kind: 'turn', at: turn.at, turn });
    (turn.says || []).forEach((text) => agentRows.push(
      { side: 'agent', kind: 'say', at: turn.at, text }));
    (turn.calls || []).forEach((call) => agentRows.push(
      { side: 'agent', kind: 'call', at: call.at == null ? turn.at : call.at, call }));
  });
  const worldRows = world.map((event) => (
    { side: 'world', kind: 'event', at: event.at, event }));

  [agentRows, worldRows].forEach((stream) => {
    let last = null;
    stream.forEach((row) => {
      if (row.at == null) row.at = last;    // 只跟自己这一侧的上一条
      else last = row.at;
    });
  });

  const rows = [...agentRows, ...worldRows];
  rows.forEach((row, index) => { row.order = index; });   // 同一秒内保持原有先后
  return rows.slice().sort((a, b) => {
    if (a.at == null) return b.at == null ? a.order - b.order : -1;
    if (b.at == null) return 1;
    return (a.at - b.at) || (a.order - b.order);
  });
}

/** 相邻两条之间的静默单独占一行 —— 它通常就是答案。 */
export function withQuiet(rows, quiet = QUIET_SECONDS) {
  const out = [];
  let previous = null;
  rows.forEach((row) => {
    if (row.at != null && previous != null) {
      const gap = row.at - previous;
      if (gap >= quiet) {
        out.push({ quiet: gap < 10 ? gap.toFixed(1) : Math.round(gap),
                   loud: gap >= QUIET_LOUD });
      }
    }
    if (row.at != null) previous = row.at;
    out.push(row);
  });
  return out;
}

// ── 渲染 ─────────────────────────────────────────────────────────────────────

function _render(data) {
  const run = data.run || {};
  const head = `
    <div class="bm-tl-head">
      <span class="bm-tl-score">${run.score_total ?? '—'}</span>
      <span class="bm-tl-meta">${_esc(run.suite || '')}　${_esc(run.llm_model || '')}
        ${_esc(Object.values(run.image_tags || {}).filter(Boolean).join(' '))}</span>
    </div>`;

  // **打分细节和日志是两回事，分两个 tab。**
  //
  // 「这次多少分、为什么」和「它一步步做了什么」是两种查法：前者按原则读，后者按时间
  // 读。叠在一条纵向流里，读任何一个都要滚过另一个 —— 而日志动辄几十行，打分细节就被
  // 推到看不见的地方。默认停在打分，因为那是打开这个弹窗的第一个问题。
  const cards = (data.cases || []).map(scorecard).join('');
  const agent = data.agent || [];
  const world = data.world || [];

  const log = (!agent.length && !world.length)
    ? `<div class="bm-empty">这次运行没有留下记录。<br>
        记录在运行结束时才落盘，更早的运行（或清过的会话）没有这一份。</div>`
    : `${agent.length ? '' : '<div class="bm-tl-fail">这次运行的对话记录已经没有了。</div>'}
       <div class="bm-tl-r bm-tl-r--headrow">
         <span class="bm-tl-left">Agent 请求做什么</span>
         <span class="bm-tl-t"></span>
         <span class="bm-tl-right">世界真的做了什么</span>
       </div>
       <div class="bm-tl-flow">${withQuiet(merge(agent, world)).map(_row).join('')}</div>`;

  return `${head}
    <div class="bm-tl-tabs">
      <button class="bm-tl-tab active" data-pane="score">打分细节</button>
      <button class="bm-tl-tab" data-pane="log">运行日志</button>
    </div>
    <div class="bm-tl-pane" data-pane="score">${cards ||
      '<div class="bm-empty">这次运行没有留下判定明细。<br>' +
      '更早的运行只存了没过的项目名，理由和指标是后来才开始落盘的。</div>'}</div>
    <div class="bm-tl-pane hidden" data-pane="log">${log}</div>`;
}

function _row(row) {
  if (row.quiet) {
    return `<div class="bm-tl-r bm-tl-r--quiet${row.loud ? ' bm-tl-r--loud' : ''}">
      <span class="bm-tl-left"></span><span class="bm-tl-t">静了 ${row.quiet} 秒</span>
      <span class="bm-tl-right"></span></div>`;
  }
  const time = row.at == null ? '' : `+${Number(row.at).toFixed(1)}s`;
  return `<div class="bm-tl-r bm-tl-r--${row.side}">
    <span class="bm-tl-left">${row.side === 'agent' ? _agentCell(row) : ''}</span>
    <span class="bm-tl-t">${time}</span>
    <span class="bm-tl-right">${row.side === 'world' ? _worldCell(row.event) : ''}</span>
  </div>`;
}

function _agentCell(row) {
  if (row.kind === 'turn') {
    const t = row.turn;
    const marks = [
      t.sessionTurn != null && t.sessionTurn !== t.turn
        ? `会话内第 ${t.sessionTurn + 1} 轮` : '',
      t.timing === 'written' ? '写完时' : '',
      t.timing === 'outside' ? '时间不在本次运行内' : '',
    ].filter(Boolean).map((m) => `<i class="bm-tl-session">${_esc(m)}</i>`).join('');
    return `<span class="bm-tl-turnhead">第 ${t.turn + 1} 轮${marks}</span>${
      t.trigger ? `<span class="bm-clip bm-tl-trigger">“${_esc(t.trigger)}”</span>` : ''}`;
  }
  if (row.kind === 'say') return `<span class="bm-clip bm-tl-says">${_esc(row.text)}</span>`;
  return `<b class="bm-tl-callname">${_esc(row.call.name)}</b>` +
    `<span class="bm-clip bm-tl-args">${_esc(row.call.args)}</span>`;
}

function _worldCell(e) {
  const label = KINDS[e.kind] || e.kind;
  const known = KINDS[e.kind] !== undefined;
  // 真机上没有 label（驱动的 ACP result 不带它），目标只在派发参数里 —— 退到参数，
  // 否则「出发」就是一行没有目的地的字。
  const detail = e.label || e.text || _args(e.args);
  return `<span class="bm-tl-kind${known ? '' : ' bm-tl-kind--minor'}">${_esc(label)}</span>` +
    `<span class="bm-clip bm-tl-detail">${_esc(detail)}${
      e.status ? ` <i>${_esc(e.status)}</i>` : ''}</span>`;
}

function _args(args) {
  const entries = Object.entries(args || {});
  return entries.length ? entries.map(([k, v]) => `${k}=${v}`).join(' ') : '';
}

function _esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
