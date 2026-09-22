/**
 * benchmark-editor.js — 改一个用例。
 *
 * 用例里能改的东西，绝大部分是**自然语言**：初始指令是用户会对机器人说的那句话，
 * 插话是他中途会插的那句话，要求是他对结果的期待。所以这个编辑器长得像一张写字的
 * 表单，不像一个配置面板。
 *
 * ## 要求是人话，挂在七条原则之一下面
 *
 * 原先这里是一组结构化字段：期望站序、每站到达后必须讲解、被打断的那一段……那是
 * 展区导览一个具体场景的词汇。现在写的是「到了再讲，不要没到就开讲」这样的句子，
 * 由裁判去判 —— 而它判的时候手里拿着算好的指标，所以是有据可依的判断。
 *
 * ## 参考流程有自己的框，但走普通的计分
 *
 * 「先开地图，再导航」是一串步骤，而且**偏离不等于失败** —— 好的 agent 可能找到更优
 * 的顺序。所以它有独立的输入框、裁判对它做逐步比对，但分数仍由一条要求的权重算。
 *
 * ## 地图不在这儿了
 *
 * 用例跑在**当前画布、当前世界**上。地图曾经是一个字段，现在不是了 —— 要换世界，
 * 在画布上换。
 *
 * ## 为什么保存不拦
 *
 * 编辑是分几次做完的，存一半不该被拒。校验的结果就贴在保存按钮旁边，随时看得见，
 * 而不是等点「跑」的时候才说。真正拦人的是跑，不是存。
 */

import { showToast } from './toast.js';

// 七条原则。和 `benchmark_case.DIMENSIONS` 必须一致 —— 对不上的那条会被后端归到
// 「回答效果」里，而用户看到的是自己选的那个原则下面空空如也。
const DIMENSIONS = [
  ['world_timing', '物理世界时序性'],
  ['concurrency', '同步执行效率'],
  ['llm_latency', 'LLM 延时'],
  ['cache_hit', 'cache 命中'],
  ['answer_quality', '回答效果'],
  ['ux', '用户体验'],
  ['physical_safety', '安全'],
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
 * 一条插话只能有一个触发方式。
 *
 * 两个都填，跑的时候按 `after_arrival` 走 —— 而写的人以为按秒走。所以这里把没选中
 * 的那个清掉，让文件里存的就是实际会发生的事。
 */
export function normalizeInjection(row) {
  const out = { text: String(row.text || ''), delay: Number(row.delay) || 0 };
  if (row.mode === 'at') out.after_action = undefined, out.at = Number(row.at) || 0;
  else out.after_action = Math.max(1, Number(row.after_action) || 1);
  // 两个触发条件**互斥**：两个都写进去，后端按 `after_action` 优先，而用户看到的是
  // 自己填的秒数被无视了。只留选中的那一个。
  if (out.at == null) delete out.at;
  if (out.after_action == null) delete out.after_action;
  return out;
}

// ── 渲染 ─────────────────────────────────────────────────────────────────────

function _test() { return _editing.payload.test || {}; }

function _render() {
  const body = document.getElementById('bm-editor-body');
  if (!body || !_editing) return;
  const test = _test();
  const run = test.run || {};
  const reqs = (test.requirements || []).filter((r) => r.id !== '__procedure__');

  body.innerHTML = `
    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-name">名字</label>
      <input class="bm-ed-input" id="bm-ed-name" value="${_esc(_editing.name)}">
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-prompt">初始指令</label>
      <textarea class="bm-ed-input bm-ed-prose" id="bm-ed-prompt" rows="2"
        placeholder="用户会对机器人说的那句话">${_esc(run.prompt || '')}</textarea>
      <p class="bm-ed-hint">这句话会当作用户消息送进去，和真人说的走同一条路。
        它跑在**当前画布**上 —— 运行不会改动画布。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label" for="bm-ed-procedure">参考流程<span class="bm-ed-opt">可选</span></label>
      <textarea class="bm-ed-input bm-ed-prose" id="bm-ed-procedure" rows="4"
        placeholder="一行一步，例如：&#10;1. 打开地图&#10;2. 导航到第一站&#10;3. 到达后再讲解"
        >${_esc(test.procedure || '')}</textarea>
      <p class="bm-ed-hint">写了就多一条「按参考流程执行」的要求。裁判会**逐步比对**
        并说明每一处偏离是否合理 —— 更优的顺序也是偏离，偏离不等于失败。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">要求
        <button class="bm-linkbtn" id="bm-ed-addreq">+ 添加一条</button>
      </label>
      <div id="bm-ed-reqs">${reqs.map(_requirementHtml).join('')}</div>
      <p class="bm-ed-hint">用人话写你对结果的期待，挂到它属于的那条原则下面。
        没写要求的原则照样按默认目标判 —— 那一半是算出来的，不经裁判。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">插话
        <button class="bm-linkbtn" id="bm-ed-add">+ 添加一条</button>
      </label>
      <div id="bm-ed-injections">${(run.injections || []).map(_injectionHtml).join('')}</div>
      <p class="bm-ed-hint">按「第 N 个动作完成后」触发，而不是按绝对秒数：真机上 LLM
        一轮 3-48 秒，写死的偏移会落到完全不同的一段过程上。</p>
    </div>

    <div class="bm-ed-row">
      <label class="bm-ed-label">收尾</label>
      <div class="bm-ed-inline">
        <label class="bm-ed-weight">时间预算
          <input class="bm-ed-input bm-ed-xs" id="bm-ed-budget"
            value="${_esc(run.budget_seconds ?? 900)}"></label>
        <span class="bm-ed-unit">秒</span>
        <label class="bm-ed-weight">安静多久算结束
          <input class="bm-ed-input bm-ed-xs" id="bm-ed-idle"
            value="${_esc(run.idle_seconds ?? 60)}"></label>
        <span class="bm-ed-unit">秒</span>
      </div>
      <p class="bm-ed-hint">安静 = 没有新事实、而且没有还没完成的动作。
        只看「没有新事实」会在机器人走在半路上时把运行判结束。</p>
    </div>`;

  document.getElementById('bm-ed-addreq')?.addEventListener('click', _addRequirement);
  _bindRequirementRows();
  document.getElementById('bm-ed-add')?.addEventListener('click', _addInjection);
  _bindInjectionRows();
  _showProblems([]);
}

function _requirementHtml(req, index) {
  return `
    <div class="bm-ed-req" data-idx="${index}">
      <div class="bm-ed-inline">
        <select class="bm-ed-input bm-ed-sm" data-rfield="dimension">
          ${DIMENSIONS.map(([key, label]) => `
            <option value="${key}"${
              (req.dimension || 'answer_quality') === key ? ' selected' : ''
            }>${label}</option>`).join('')}
        </select>
        <label class="bm-ed-weight">权重
          <input class="bm-ed-input bm-ed-xs" data-rfield="weight"
            value="${_esc(req.weight ?? 10)}"></label>
        <button class="bm-linkbtn" data-rdrop="${index}">删除</button>
      </div>
      <textarea class="bm-ed-input bm-ed-prose" data-rfield="text" rows="2"
        placeholder="例如：到了再讲，不要没到就开讲">${_esc(req.text || '')}</textarea>
    </div>`;
}


function _bindRequirementRows() {
  document.querySelectorAll('[data-rdrop]').forEach((button) => {
    button.addEventListener('click', () => _dropRow(
      (test) => (test.requirements ||= []), button.dataset.rdrop));
  });
}


function _addRequirement() {
  _collectInto();
  const test = _test();
  test.requirements = [...(test.requirements || []),
                       { text: '', weight: 10, dimension: 'answer_quality' }];
  _render();
}


function _readRequirements() {
  return Array.from(document.querySelectorAll('.bm-ed-req')).map((row, index) => {
    const field = (name) => row.querySelector(`[data-rfield="${name}"]`)?.value?.trim() ?? '';
    return { id: `r${index}`, text: field('text'),
             weight: Number(field('weight')) || 0,
             dimension: field('dimension') || 'answer_quality' };
  });
}


function _injectionHtml(injection, index) {
  // 「第 N 个动作完成后」是 `after_arrival`（到达某一站）的通用化：到站是展区导览的
  // 说法，而「第 N 个动作做完」在任何用例里都成立，判据也一样在事实流里。
  const byAction = injection.at == null;
  return `
    <div class="bm-ed-injection" data-idx="${index}">
      <div class="bm-ed-inline">
        <label><input type="radio" name="trig${index}" value="action"
          ${byAction ? 'checked' : ''}> 第</label>
        <input class="bm-ed-input bm-ed-xs" data-field="after_action" placeholder="N"
          value="${_esc(injection.after_action ?? '')}">
        <span class="bm-ed-unit">个动作完成后</span>
        <label><input type="radio" name="trig${index}" value="at"
          ${byAction ? '' : 'checked'}> 第</label>
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

/** 删掉列表里的第 index 条。
 *
 * **顺序是这个函数存在的理由：先把表单读回来，再删。** 反过来的话，`_collectInto()`
 * 会从 DOM 重建整个列表 —— 而 DOM 里那一行还在，刚删掉的那条又被写了回去。表现就是
 * 「点删除没反应」，而且控制台干干净净。插话那一段原先就是反的。
 *
 * 顺带：先 collect 也保住了用户在别的框里刚敲的字，那些还没进 payload。
 */
function _dropRow(pick, index) {
  _collectInto();
  const list = pick(_test());
  list.splice(Number(index), 1);
  _render();
}

function _bindInjectionRows() {
  document.querySelectorAll('[data-drop]').forEach((button) => {
    button.addEventListener('click', () => _dropRow(
      (test) => (test.run ||= {}).injections ||= [], button.dataset.drop));
  });
}

function _addInjection() {
  _collectInto();
  const run = _test().run || {};
  run.injections = [...(run.injections || []), { after_action: 1, delay: 0, text: '' }];
  _render();
}

// ── 读回表单 ─────────────────────────────────────────────────────────────────

function _value(id) { return document.getElementById(id)?.value?.trim() ?? ''; }
function _number(id) {
  const raw = _value(id);
  return raw === '' ? null : Number(raw);
}

function _readInjections() {
  return Array.from(document.querySelectorAll('.bm-ed-injection')).map((row) => {
    const field = (name) => row.querySelector(`[data-field="${name}"]`)?.value?.trim() ?? '';
    const mode = row.querySelector('input[type="radio"]:checked')?.value || 'action';
    return normalizeInjection({
      mode, after_action: field('after_action'), at: field('at'),
      delay: field('delay'), text: field('text'),
    });
  });
}

/** 把表单读回 `_editing.payload`。 */
function _collectInto() {
  const test = _test();
  _editing.name = _value('bm-ed-name') || _editing.name;
  test.name = _editing.name;
  test.procedure = document.getElementById('bm-ed-procedure')?.value ?? '';
  // 参考流程那条要求（`__procedure__`）不在渲染的行里 —— 它的文本由后端按 procedure
  // 生成，只有权重和归属可能被改过。`_readRequirements()` 整个替换数组，不把它带上
  // 的话，那份改动每保存一次就被抹掉一次，而且不报错。
  //
  // 放在**末尾**：上面那些行的下标就是渲染顺序，删除按下标走（见 `_dropRow`），
  // 插在前面会让每一次删除都删错一条。
  const procedure = (test.requirements || []).find((r) => r.id === '__procedure__');
  test.requirements = [..._readRequirements(), ...(procedure ? [procedure] : [])];
  test.run = {
    ...(test.run || {}),
    prompt: document.getElementById('bm-ed-prompt')?.value ?? '',
    injections: _readInjections(),
    budget_seconds: _number('bm-ed-budget') ?? 900,
    idle_seconds: _number('bm-ed-idle') ?? 60,
  };
  // 地图和出生点不再是用例的字段 —— 用例跑在当前世界上。旧包体里带着的原样留着，
  // 后端还认；这里只是不再产生新的。
  delete test.evaluate;
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
