/**
 * benchmark-editor.js — 改一个用例。
 *
 * 用例里能改的东西，绝大部分是**自然语言**：初始指令是用户会对机器人说的那句话，
 * 插话是他中途会插的那句话。只有触发条件（到达哪一站 / 第几秒）和几个数字是结构化
 * 的。所以这个编辑器长得像一张写字的表单，不像一个配置面板。
 *
 * ## 地图是一个声明，不是一个下拉框
 *
 * 这里**不**从仿真器的 `list_maps` 拉选项，站名也不从地图的 POI 里选。理由不只是
 * 解耦：判定层本来就设计成「同样形状的事件流，跑在真机器人上也一样能判」。一个必须
 * 先有仿真器、先有那张地图才能编辑的用例，把那条路堵死了。地图名和站名都是字符串，
 * 对不对得上由跑的时候说 —— 那时候本来就会按层报出来。
 *
 * ## 为什么保存不拦
 *
 * 编辑是分几次做完的，存一半不该被拒。校验的结果就贴在保存按钮旁边，随时看得见，
 * 而不是等点「跑」的时候才说。真正拦人的是跑，不是存。
 */

import { showToast } from './toast.js';

const DIMENSIONS = [
  ['orchestration', '编排'], ['interruption', '打断'], ['long_horizon', '长程'],
  ['safety', '安全'], ['latency', '时效'],
];

let _editing = null;     // { id, name, payload }
let _onSaved = null;

export function initEditor(onSaved) {
  _onSaved = onSaved;
  document.getElementById('bm-editor-close')?.addEventListener('click', closeEditor);
  document.getElementById('bm-editor-save')?.addEventListener('click', _save);
  document.getElementById('bm-editor-canvas')?.addEventListener('click', _takeCanvas);
}

export async function openEditor(caseId) {
  const response = await fetch(`/api/benchmark/cases/${caseId}`);
  if (!response.ok) { showToast('打不开这个用例'); return; }
  const record = await response.json();
  if (!record?.payload?.test) {
    // 没有 test 段就没有可编辑的东西。不拦的话会开出一个空白弹窗 —— 那比报错更难查。
    showToast('这个用例里没有 test 段，打不开编辑器');
    return;
  }
  _editing = { id: record.id, name: record.name, payload: record.payload };
  document.getElementById('bm-editor-overlay')?.classList.remove('hidden');
  _render();
}

export function closeEditor() {
  document.getElementById('bm-editor-overlay')?.classList.add('hidden');
  _editing = null;
}

// ── 纯函数：解析与提示（可单测，不碰 DOM） ───────────────────────────────────

/**
 * 「P3, P4, P5」「每行一个」都接受。
 *
 * 站序是最常被手写的一项，而它跨行粘贴过来的机会和逗号分隔一样多。只认一种分隔符
 * 的话，另一种会安静地变成**一个**名字很长的站点 —— 跑起来一站都对不上，看着却像
 * 是 agent 走错了。
 */
export function parseList(text) {
  return String(text || '')
    .split(/[,，\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/** 权重合计不是 100 时说一句。不阻止 —— 评分是按比例算的，105 也能跑。 */
export function weightNote(weights) {
  const total = DIMENSIONS.reduce((sum, [key]) => sum + (Number(weights[key]) || 0), 0);
  if (!total) return '所有权重都是 0，总分算不出来';
  return total === 100 ? '' : `合计 ${total}，不是 100（按比例折算，不影响跑）`;
}

/**
 * 一条插话只能有一个触发方式。
 *
 * 两个都填，跑的时候按 `after_arrival` 走 —— 而写的人以为按秒走。所以这里把没选中
 * 的那个清掉，让文件里存的就是实际会发生的事。
 */
export function normalizeInjection(row) {
  const out = { text: String(row.text || ''), delay: Number(row.delay) || 0 };
  if (row.mode === 'at') out.at = Number(row.at) || 0;
  else out.after_arrival = String(row.after_arrival || '');
  return out;
}

// ── 渲染 ─────────────────────────────────────────────────────────────────────

function _test() { return _editing.payload.test || {}; }

function _render() {
  const body = document.getElementById('bm-editor-body');
  if (!body || !_editing) return;
  const test = _test();
  const run = test.run || {};
  const world = run.world || {};
  const expect = (test.evaluate || {}).expect || {};
  const weights = (test.evaluate || {}).weights || {};
  const leg = expect.interrupted_leg || {};

  body.innerHTML = `
    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-name">名字</label>
      <input class="bm-ed-input" id="bm-ed-name" value="${_esc(_editing.name)}">
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-prompt">初始指令</label>
      <textarea class="bm-ed-input bm-ed-prose" id="bm-ed-prompt" rows="2"
        placeholder="用户会对机器人说的那句话">${_esc(run.prompt || '')}</textarea>
      <p class="bm-ed-hint">这句话会当作用户消息送进去，和真人说的走同一条路。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">世界</label>
      <div class="bm-ed-inline">
        <input class="bm-ed-input bm-ed-sm" id="bm-ed-map" placeholder="地图名"
          value="${_esc(world.map || '')}">
        <span class="bm-ed-unit">出生点</span>
        ${['x', 'y', 'yaw'].map((axis) => `
          <label class="bm-ed-weight">${axis}
            <input class="bm-ed-input bm-ed-xs" id="bm-ed-${axis}"
              value="${_esc((world.spawn || {})[axis] ?? '')}"></label>`).join('')}
      </div>
      <p class="bm-ed-hint">地图名交给仿真器解析。留空表示用它当前的世界 ——
        用例跑在真机器人上时就留空。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">插话
        <button class="bm-linkbtn" id="bm-ed-add">+ 添加一条</button>
      </label>
      <div id="bm-ed-injections">${(run.injections || []).map(_injectionHtml).join('')}</div>
      <p class="bm-ed-hint">按「到达某站」触发，而不是按绝对秒数：真机上 LLM 一轮
        3-48 秒，写死的偏移会落到完全不同的一段路上。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-order">期望站序</label>
      <textarea class="bm-ed-input bm-ed-prose" id="bm-ed-order" rows="2"
        placeholder="P3, P4, P5……">${_esc((expect.waypoint_order || []).join(', '))}</textarea>
    </div>

    <div class="bm-ed-row bm-ed-checks">
      <label><input type="checkbox" id="bm-ed-announce"
        ${expect.announce_after_arrive === false ? '' : 'checked'}> 每站到达后必须讲解</label>
      <label><input type="checkbox" id="bm-ed-occupied"
        ${expect.never_occupied === false ? '' : 'checked'}> 从不进入占用格</label>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">被打断的那一段</label>
      <div class="bm-ed-inline">
        <label class="bm-ed-weight">去
          <input class="bm-ed-input bm-ed-sm" id="bm-ed-leg" placeholder="哪一站"
            value="${_esc(leg.target || '')}"></label>
        <label class="bm-ed-weight">应报
          <select class="bm-ed-input bm-ed-sm" id="bm-ed-legstatus">
            ${['cancelled', 'completed', 'failed'].map((status) => `
              <option value="${status}"${
                (leg.acp_status || 'cancelled') === status ? ' selected' : ''
              }>${status}</option>`).join('')}
          </select></label>
        <label class="bm-ed-weight">最小进度
          <input class="bm-ed-input bm-ed-xs" id="bm-ed-legmin"
            value="${_esc(leg.min_progress ?? '')}"></label>
      </div>
      <div class="bm-ed-inline">
        <label class="bm-ed-weight">绕行后回到
          <input class="bm-ed-input bm-ed-sm" id="bm-ed-resume" placeholder="哪一站"
            value="${_esc(expect.resume_target || '')}"></label>
        <label class="bm-ed-weight">时间预算
          <input class="bm-ed-input bm-ed-xs" id="bm-ed-budget"
            value="${_esc(expect.max_wall_seconds ?? '')}"></label>
        <span class="bm-ed-unit">秒</span>
      </div>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">权重 <span class="bm-ed-hint" id="bm-ed-weightnote"></span></label>
      <div class="bm-ed-inline">
        ${DIMENSIONS.map(([key, label]) => `
          <label class="bm-ed-weight">${label}
            <input class="bm-ed-input bm-ed-xs" data-weight="${key}"
              value="${_esc(weights[key] ?? 0)}"></label>`).join('')}
      </div>
    </div>`;

  document.getElementById('bm-ed-add')?.addEventListener('click', _addInjection);
  _bindInjectionRows();
  body.querySelectorAll('[data-weight]').forEach((input) => {
    input.addEventListener('input', _updateWeightNote);
  });
  _updateWeightNote();
  _showProblems([]);
}

function _injectionHtml(injection, index) {
  const byArrival = injection.at == null;
  return `
    <div class="bm-ed-injection" data-idx="${index}">
      <div class="bm-ed-inline">
        <label><input type="radio" name="trig${index}" value="arrive"
          ${byArrival ? 'checked' : ''}> 到达</label>
        <input class="bm-ed-input bm-ed-sm" data-field="after_arrival" placeholder="站名"
          value="${_esc(injection.after_arrival || '')}">
        <label><input type="radio" name="trig${index}" value="at"
          ${byArrival ? '' : 'checked'}> 第</label>
        <input class="bm-ed-input bm-ed-xs" data-field="at" placeholder="秒"
          value="${_esc(injection.at ?? '')}">
        <label class="bm-ed-weight">延时
          <input class="bm-ed-input bm-ed-xs" data-field="delay"
            value="${_esc(injection.delay ?? 0)}"></label>
        <span class="bm-ed-unit">秒</span>
        <button class="bm-linkbtn" data-drop="${index}">删除</button>
      </div>
      <textarea class="bm-ed-input bm-ed-prose" data-field="text" rows="2"
        placeholder="用户插的那句话">${_esc(injection.text || '')}</textarea>
    </div>`;
}

function _bindInjectionRows() {
  document.querySelectorAll('[data-drop]').forEach((button) => {
    button.addEventListener('click', () => {
      const run = _test().run || {};
      run.injections = (run.injections || []).filter(
        (_, i) => i !== Number(button.dataset.drop));
      _collectInto(false);
      _render();
    });
  });
}

function _addInjection() {
  _collectInto(false);
  const run = _test().run || {};
  run.injections = [...(run.injections || []), { after_arrival: '', delay: 0, text: '' }];
  _render();
}

function _updateWeightNote() {
  const note = document.getElementById('bm-ed-weightnote');
  if (note) note.textContent = weightNote(_readWeights());
}

// ── 读回表单 ─────────────────────────────────────────────────────────────────

function _value(id) { return document.getElementById(id)?.value?.trim() ?? ''; }
function _number(id) {
  const raw = _value(id);
  return raw === '' ? null : Number(raw);
}

function _readWeights() {
  const weights = {};
  document.querySelectorAll('[data-weight]').forEach((input) => {
    weights[input.dataset.weight] = Number(input.value) || 0;
  });
  return weights;
}

function _readInjections() {
  return Array.from(document.querySelectorAll('.bm-ed-injection')).map((row) => {
    const field = (name) => row.querySelector(`[data-field="${name}"]`)?.value?.trim() ?? '';
    const mode = row.querySelector('input[type="radio"]:checked')?.value || 'arrive';
    return normalizeInjection({
      mode, after_arrival: field('after_arrival'), at: field('at'),
      delay: field('delay'), text: field('text'),
    });
  });
}

/** 把表单读回 `_editing.payload`。`spawn` 三个数都空就整块不写。 */
function _collectInto() {
  const test = _test();
  const spawn = { x: _number('bm-ed-x'), y: _number('bm-ed-y'), yaw: _number('bm-ed-yaw') };
  const hasSpawn = Object.values(spawn).some((v) => v != null);

  _editing.name = _value('bm-ed-name') || _editing.name;
  test.name = _editing.name;
  test.run = {
    prompt: document.getElementById('bm-ed-prompt')?.value ?? '',
    world: { ...(_value('bm-ed-map') ? { map: _value('bm-ed-map') } : {}),
             ...(hasSpawn ? { spawn } : {}) },
    injections: _readInjections(),
  };

  const expect = {
    waypoint_order: parseList(document.getElementById('bm-ed-order')?.value),
    announce_after_arrive: !!document.getElementById('bm-ed-announce')?.checked,
    never_occupied: !!document.getElementById('bm-ed-occupied')?.checked,
  };
  if (_value('bm-ed-leg')) {
    expect.interrupted_leg = {
      target: _value('bm-ed-leg'),
      acp_status: _value('bm-ed-legstatus') || 'cancelled',
      ...(_number('bm-ed-legmin') == null ? {} : { min_progress: _number('bm-ed-legmin') }),
    };
  }
  if (_value('bm-ed-resume')) expect.resume_target = _value('bm-ed-resume');
  if (_number('bm-ed-budget') != null) expect.max_wall_seconds = _number('bm-ed-budget');
  // 站序空着就别写一个空数组 —— validate 把空 expect 当作「没有断言」，
  // 而一个 `waypoint_order: []` 会让它看起来像是断言过了。
  if (!expect.waypoint_order.length) delete expect.waypoint_order;

  test.evaluate = { expect, weights: _readWeights() };
  _editing.payload.test = test;
}

// ── 保存 ─────────────────────────────────────────────────────────────────────

async function _save() {
  _collectInto();
  try {
    const response = await fetch(`/api/benchmark/cases/${_editing.id}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ payload: _editing.payload, name: _editing.name,
                             origin: 'edited' }),
    });
    const saved = await response.json();
    if (!response.ok) throw new Error(saved.detail || '保存失败');
    _showProblems(saved.problems || []);
    showToast((saved.problems || []).length ? '已保存，但暂时无法运行' : '已保存');
    _onSaved?.();
  } catch (e) {
    showToast(`保存失败：${e.message || e}`);
  }
}

function _showProblems(problems) {
  const el = document.getElementById('bm-editor-problems');
  if (!el) return;
  el.innerHTML = problems.length
    ? `<ul class="bm-blockers">${problems.map((p) => `<li>${_esc(p)}</li>`).join('')}</ul>`
    : '';
}

async function _takeCanvas() {
  // 画布那一半在画布上编，它*就是*画布。这里只是把它的当前样子收进这个用例。
  try {
    const snapshot = await (await fetch('/api/benchmark/snapshot', { method: 'POST' })).json();
    if (!snapshot.payload) throw new Error(snapshot.detail || '打包失败');
    _collectInto();
    _editing.payload = { ...snapshot.payload, test: _editing.payload.test };
    showToast(`已把当前画布的 ${snapshot.cards} 张卡片收进这个用例，记得保存`);
  } catch (e) {
    showToast(`收不进来：${e.message || e}`);
  }
}

function _esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
