/**
 * benchmark-timeline.js — 一次跑动的现场。
 *
 * 分数只回答「好不好」。要回答「哪儿坏了」，得看这一趟里说了什么、想了什么、调了
 * 什么工具、世界怎么回应的 —— 而且要能把它们对上：那句插话到底落在哪一段路上，
 * 讲解是在到达之前还是之后。
 *
 * ## 两条轨道并排，不合成一条
 *
 * agent 那侧的时间是墙钟，世界那侧是仿真时钟。两个钟相减没有意义，所以各自归一到
 * 「从本轮开始起算的秒数」并排放。对齐基准两边都是跑动开始：世界的第一条事件是
 * reset 那一刻，agent 的第一轮是被初始指令唤醒的那一轮。编一个共同时间轴会更好看，
 * 也会在读的人最需要精确的时候骗他。
 *
 * ## 为什么这些数据存得下来
 *
 * 仿真器的世界下一次跑动一开始就被重置，会话历史会被压缩。所以事实在写入结果时就
 * 一起存进了 `benchmark_case.facts`。会话没了就说没了，不编。
 */

// 断言名是给代码看的。「没过：waypoint_order」要人自己翻译，而这一行正是打开这个
// 弹窗的第一眼。
const CHECKS = {
  waypoint_order: '站序不对', announce_after_arrive: '到达后没讲解',
  never_occupied: '进了占用格', interrupted_leg: '被打断那段上报不实',
  resume_correctness: '绕行后没回到原来那站',
  exactly_one_terminal_post: 'ACP 重复上报', max_wall_seconds: '超时',
};

// 一次跑动里最值钱的线索往往是「这里什么都没发生」—— 机器人卡住、LLM 空转、
// barrier 等超时，长得都一样：一段静默。相邻两条事件挨着排，这段静默就看不见了。
const QUIET_SECONDS = 25;

// 这几类事件是读的时候真正在找的东西，其余的（led、note 之外的杂项）压成一行灰字。
const KINDS = {
  scenario_load: ['世界', '载入'],
  scenario_start: ['世界', '开始'],
  scenario_stop: ['世界', '结束'],
  note: ['用例', ''],
  nav_start: ['出发', ''],
  arrive: ['到达', ''],
  nav_cancelled: ['放弃', ''],
  nav_failed: ['失败', ''],
  speak_start: ['开讲', ''],
  speak_end: ['讲完', ''],
  speech_interrupt: ['打断', ''],
};

export function initTimeline() {
  document.getElementById('bm-timeline-close')?.addEventListener('click', close);
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
    body.innerHTML = `<div class="bm-empty">读不到这次跑动：${_esc(e.message || e)}</div>`;
    return;
  }
  body.innerHTML = _render(data);
}

function close() {
  document.getElementById('bm-timeline-overlay')?.classList.add('hidden');
}

function _render(data) {
  const run = data.run || {};
  const head = `
    <div class="bm-tl-head">
      <span class="bm-tl-score">${run.score_total ?? '—'}</span>
      <span class="bm-tl-meta">${_esc(run.suite || '')}　${_esc(run.llm_model || '')}
        ${_esc(Object.values(run.image_tags || {}).filter(Boolean).join(' '))}</span>
    </div>
    ${(data.cases || []).map((c) => (c.assertions || []).length
      ? `<div class="bm-tl-fail">没过：${(c.assertions || [])
          .map((a) => _esc(CHECKS[a] || a)).join('、')}</div>`
      : '').join('')}`;

  const world = data.world || [];
  const agent = data.agent || [];
  if (!world.length && !agent.length) {
    return head + `<div class="bm-empty">这次跑动的现场没有留下来。<br>
      事实在跑完时才落盘，更早的跑动（或清过的会话）没有这份记录。</div>`;
  }

  return `${head}
    <div class="bm-tl-cols">
      <section class="bm-tl-col">
        <h5 class="bm-tl-h">Agent 做了什么</h5>
        ${agent.length ? agent.map(_turn).join('')
          : '<div class="bm-empty">这次跑动的对话记录已经没有了。</div>'}
      </section>
      <section class="bm-tl-col">
        <h5 class="bm-tl-h">世界发生了什么</h5>
        ${_withQuiet(world)}
      </section>
    </div>`;
}

function _turn(t) {
  const calls = (t.calls || []).map((c) => `
    <div class="bm-tl-call"><b>${_esc(c.name)}</b>
      <span class="bm-tl-args">${_esc(c.args)}</span></div>`).join('');
  return `
    <div class="bm-tl-turn">
      <div class="bm-tl-when">${t.at == null ? '' : `+${t.at}s`}　第 ${t.turn + 1} 轮</div>
      ${t.trigger ? `<div class="bm-tl-trigger">“${_esc(t.trigger)}”</div>` : ''}
      ${(t.says || []).map((s) => `<div class="bm-tl-says">${_esc(s)}</div>`).join('')}
      ${calls}
    </div>`;
}

/** 事件之间的长静默单独占一行 —— 它通常就是答案。 */
export function withQuiet(world, quiet = QUIET_SECONDS) {
  const out = [];
  world.forEach((e, i) => {
    const gap = i ? e.at - world[i - 1].at : 0;
    if (gap >= quiet) out.push({ quiet: Math.round(gap) });
    out.push(e);
  });
  return out;
}

function _withQuiet(world) {
  return withQuiet(world).map((row) => (row.quiet
    ? `<div class="bm-tl-quiet">静了 ${row.quiet} 秒</div>`
    : _event(row))).join('');
}

function _event(e) {
  const [label, extra] = KINDS[e.kind] || [e.kind, ''];
  const known = KINDS[e.kind] !== undefined;
  const detail = e.label || e.text || extra || '';
  return `
    <div class="bm-tl-event${known ? '' : ' bm-tl-event--minor'}">
      <span class="bm-tl-when">+${e.at}s</span>
      <span class="bm-tl-kind">${_esc(label)}</span>
      <span class="bm-tl-detail">${_esc(detail)}${
        e.status ? ` <i>${_esc(e.status)}</i>` : ''}</span>
    </div>`;
}

function _esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
