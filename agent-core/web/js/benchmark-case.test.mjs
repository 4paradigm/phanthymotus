/**
 * 用例跑不了的时候，面板说什么。
 *
 * 这是把用例做成解决方案包体的一段、而不是一张卡片，换来的那件事：卡片住在驱动里，
 * 驱动没装的时候卡片本身就不存在 —— 界面上只是少了点什么，没有地方说「你缺这个
 * 驱动」。依赖写在包体里，载入之前就能逐层报出来。
 *
 * 所以这里钉的是**分层**：顺序即修复顺序，每一条对应一个不同的下一步动作。
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { blockerRows, cardKey, compareNote, hardwareNotice, plannedUtterances,
         runRefusal } from './benchmark.js';
import { normalizeInjection } from './benchmark-editor.js';

test('依赖齐了就没有任何一条拦路的', () => {
  assert.deepEqual(blockerRows([], { ok: true, missing_drivers: [], missing_assets: [] }), []);
});

test('缺驱动排在缺地图前面 —— 顺序就是修复顺序', () => {
  const rows = blockerRows([], {
    missing_drivers: ['simulator-generic'], missing_assets: ['bj-2f'],
  });

  assert.equal(rows.length, 2);
  assert.match(rows[0], /缺驱动/);
  assert.match(rows[1], /缺地图/);
});

test('缺什么就说是什么名字，不是说「依赖不全」', () => {
  const [row] = blockerRows([], { missing_drivers: ['simulator-generic'] });

  assert.match(row, /simulator-generic/);
  assert.match(row, /先装上并启动它/);
});

test('用例本身写坏了和缺东西是两回事，分开说', () => {
  const rows = blockerRows(['test.run.prompt 为空：没有初始指令，机器人不会动'], {});

  assert.equal(rows.length, 1);
  assert.match(rows[0], /用例本身/);
});

test('问不到仿真器不说成「缺地图」', () => {
  // 说成缺，一次临时故障就会看起来像装错了东西。
  const rows = blockerRows([], { missing_assets: [], assets_error: 'timeout' });

  assert.equal(rows.length, 1);
  assert.match(rows[0], /问不到/);
  assert.doesNotMatch(rows[0], /缺地图/);
});

test('名字里的尖括号不会变成标签', () => {
  const [row] = blockerRows([], { missing_drivers: ['<img src=x>'] });

  assert.match(row, /&lt;img/);
});

// ── 拒绝跑 ────────────────────────────────────────────────────────────────────

test('画布上有真设备时，说清楚是哪台的哪张卡', () => {
  // 这条不是故障提示，是它该拦下来：用例的指令和一句真指令在 collector 眼里一样，
  // 真机器人会跟着走起来。
  const message = runRefusal({ detail: { error: '...', unsafe: [
    { device: '天轶', tool: 'controlled_spatial' },
  ] } });

  assert.match(message, /天轶/);
  assert.match(message, /controlled_spatial/);
});

test('多张真卡片都点出来，不是只说第一张', () => {
  const message = runRefusal({ detail: { unsafe: [
    { device: '天轶', tool: 'loco' }, { device: 'Go2', tool: 'switch_mode' },
  ] } });

  assert.match(message, /天轶 的 loco/);
  assert.match(message, /Go2 的 switch_mode/);
});

// ── 真机确认 ──────────────────────────────────────────────────────────────────
//
// 确认弹窗要说清楚两件事：**哪些东西会动**，以及**会对它们说什么**。少了后者，人只
// 知道机器人要动，不知道它要被支使去做什么 —— 那个勾就没有意义。

test('画布上有真设备时，挑用例之前就提示', () => {
  // 原先这份清单只在被拒绝的 409 里出现 —— 人做决定之前面板一个字都没提。
  const notice = hardwareNotice([{ device: '天轶', tool: 'loco' },
                                 { mcpId: 'mcp-x', tool: 'speaker' }]);

  assert.match(notice, /天轶 的 loco/);
  assert.match(notice, /mcp-x 的 speaker/);   // 没有设备名就退回 mcpId
});

test('纯仿真的画布不挂这条提示', () => {
  assert.equal(hardwareNotice([]), '');
});

test('确认清单里的身份用 mcpId，不用设备名', () => {
  // 设备名改个昵称就变，`mcpId` 不会。服务端拿这个串比对，两边必须一字不差。
  assert.equal(cardKey({ mcpId: 'mcp-real', tool: 'loco', device: '天轶' }),
               'mcp-real:loco');
});

test('开场指令和每一条插话都念出来', () => {
  const lines = plannedUtterances({ run: {
    prompt: '带我转一下展区',
    injections: [{ after_arrival: 'P5', delay: 6, text: '先等一下' },
                 { at: 30, text: '走吧' }],
  } });

  assert.deepEqual(lines.map((l) => l.text), ['带我转一下展区', '先等一下', '走吧']);
  assert.match(lines[1].label, /到达 P5 后 6 秒/);
  assert.match(lines[2].label, /第 30 秒/);
});

test('没有插话的用例只念开场那一句', () => {
  const lines = plannedUtterances({ run: { prompt: '过来' } });

  assert.deepEqual(lines.map((l) => l.text), ['过来']);
});

test('读不到用例就给空清单，不编一句出来', () => {
  // 弹窗据此显示「读不到用例内容 —— 不知道会发出什么，先别跑」。编一句比不说更糟。
  assert.deepEqual(plannedUtterances(undefined), []);
  assert.deepEqual(plannedUtterances({}), []);
});

test('画布在确认之后变了，说的是重新确认而不是报错', () => {
  const message = runRefusal({ detail: { needs_confirmation: true, moving_cards: [] } });

  assert.match(message, /重新确认/);
  assert.doesNotMatch(message, /\[object Object\]/);
});

test('纯文本的拒绝理由原样说出来', () => {
  assert.match(runRefusal({ detail: '已经有一次基准测试在跑' }), /已经有一次基准测试在跑/);
});

test('结构化的理由不会退化成 [object Object]', () => {
  const message = runRefusal({ detail: { readiness: { missing_drivers: ['x'] } } });

  assert.doesNotMatch(message, /\[object Object\]/);
});

// ── 编辑器：几个必须自己算对的地方 ───────────────────────────────────────────

test('一条插话只留一个触发方式', () => {
  // 两个都填，跑的时候按 after_action 走，而写的人以为按秒走。
  const byAction = normalizeInjection({
    mode: 'action', after_action: '3', at: '90', delay: '6', text: '先等一下' });
  const bySecond = normalizeInjection({
    mode: 'at', after_action: '3', at: '90', delay: '0', text: '先等一下' });

  assert.deepEqual(byAction, { text: '先等一下', delay: 6, after_action: 3 });
  assert.deepEqual(bySecond, { text: '先等一下', delay: 0, at: 90 });
});

test('第 0 个动作不存在，按第 1 个算', () => {
  // 「第 0 个动作完成后」永远不会触发 —— 而它看起来像是「一开始就发」。
  const row = normalizeInjection({ mode: 'action', after_action: '0', text: '喂' });

  assert.equal(row.after_action, 1);
});

test('插话的延时空着按 0 算，不是 NaN', () => {
  const row = normalizeInjection({ mode: 'action', after_action: '1', delay: '', text: '喂' });

  assert.equal(row.delay, 0);
});

// ── 跑动现场：静默是线索 ─────────────────────────────────────────────────────

import { withQuiet, merge } from './benchmark-timeline.js';

test('每一段静默都画出来，不只是最长的那段', () => {
  // 排查时要看的是「时间去哪儿了」。每行左边虽然有 +Xs，但那要人自己做减法 ——
  // 一屏十几行减下来，小停顿根本不会被注意到。
  const rows = withQuiet([
    { at: 0, kind: 'arrive' }, { at: 4, kind: 'speak_start' },
    { at: 304, kind: 'scenario_stop' }]);

  assert.deepEqual(rows.filter((r) => r.quiet).map((r) => r.quiet), ['4.0', 300]);
});

test('长静默和短停顿画得不一样', () => {
  // 短停顿是节奏，长静默是事故。一个样子画，长的就淹在短的里面了。
  const rows = withQuiet([{ at: 0 }, { at: 3 }, { at: 100 }]);

  assert.equal(rows.filter((r) => r.quiet)[0].loud, false);
  assert.equal(rows.filter((r) => r.quiet)[1].loud, true);
});

test('1.5 秒以下当作连续', () => {
  // 再往下，标注本身比它描述的停顿还长。
  const rows = withQuiet([{ at: 0 }, { at: 1.2 }, { at: 2.0 }]);

  assert.equal(rows.filter((r) => r.quiet).length, 0);
});

test('时间不详的行不参与静默计算', () => {
  // 拿 null 当 0，会凭空算出一段跨越整场的静默 —— 这个视图里最容易被当真的假象。
  const rows = withQuiet([{ at: null }, { at: 200 }, { at: 201 }]);

  assert.equal(rows.filter((r) => r.quiet).length, 0);
});

test('第一条事件前面不算静默', () => {
  // 归一之后第一条就是 0 —— 拿它和「不存在的上一条」比，会凭空多出一行。
  const rows = withQuiet([{ at: 40, kind: 'arrive' }]);

  assert.equal(rows.length, 1);
});

// ── 进度：跑着就说跑着 ───────────────────────────────────────────────────────

import { progressView } from './benchmark.js';

test('一次 repeat 都还没跑完，也要显示它在跑', () => {
  // 长程用例的常态：第一趟导览要好几分钟。这段时间刷新页面，面板不能说
  // 「没有正在进行的跑动」——机器人正在走。真机上就是这么露出来的。
  const view = progressView({ state: 'running', repeats: 3, cases: [] });

  assert.equal(view.live, true);
  assert.equal(view.show, true);
  assert.equal(view.total, 3);
});

test('空闲且没有结果时才说没有跑动', () => {
  assert.equal(progressView({ state: 'idle' }).show, false);
  assert.equal(progressView(null).show, false);
});

test('跑完了还留在面板上，不立刻清空', () => {
  const view = progressView({ state: 'done', repeats: 2, cases: [
    { outcome: 'ok' }, { outcome: 'failed' }] });

  assert.equal(view.live, false);
  assert.equal(view.show, true);
  assert.equal(view.done, 2);
});

test('排队中的 repeat 不算已完成', () => {
  const view = progressView({ state: 'running', repeats: 3, cases: [
    { outcome: 'ok' }, { outcome: 'pending' }] });

  assert.equal(view.done, 1);
  assert.equal(view.total, 3);
});


// ── 合流：一条共享时间轴 ─────────────────────────────────────────────────────

test('两侧按时间合成一条流', () => {
  // 左右两栏各自排版时，同一秒在两边的高度不同 —— 时刻是对的，读的人却得拿眼睛
  // 找数字，等于没对齐。
  const rows = merge(
    [{ turn: 0, at: 5, says: [], calls: [{ name: 'speak', at: 8.7, args: '{}' }] }],
    [{ at: 0, kind: 'scenario_load' }, { at: 8.7, kind: 'speak_start' }]);

  assert.deepEqual(rows.map((r) => [r.at, r.side]),
                   [[0, 'world'], [5, 'agent'], [8.7, 'agent'], [8.7, 'world']]);
});

test('同一秒里各自原有的先后不被打乱', () => {
  const rows = merge(
    [{ turn: 0, at: 3, says: ['先说这句', '再说这句'], calls: [] }], []);

  assert.deepEqual(rows.map((r) => r.kind), ['turn', 'say', 'say']);
  assert.equal(rows[1].text, '先说这句');
});

test('时间不详的条目不跨侧去借时刻', () => {
  // 跨侧借，等于在两条流之间编一个并不存在的对应关系 —— 这个视图里最容易被当真的
  // 假象。一侧的第一条就没有时间，就让它保持没有。
  const rows = merge(
    [{ turn: 0, at: null, says: [], calls: [{ name: 'speak', args: '{}' }] }],
    [{ at: 40, kind: 'arrive' }]);

  assert.equal(rows[0].at, null);
  assert.equal(rows[0].side, 'agent');
});

test('时间不详的条目跟着**自己这一侧**的上一条', () => {
  const rows = merge([
    { turn: 0, at: 12, says: [], calls: [] },
    { turn: 1, at: null, says: ['配不上 spans 的那一轮'], calls: [] },
  ], [{ at: 500, kind: 'arrive' }]);

  assert.deepEqual(rows.filter((r) => r.side === 'agent').map((r) => r.at), [12, 12, 12]);
});

test('调用自带的时刻优先于所属轮次的时刻', () => {
  const rows = merge(
    [{ turn: 0, at: 5, says: [], calls: [{ name: 'speak', at: 30, args: '{}' }] }], []);

  assert.deepEqual(rows.map((r) => r.at), [5, 30]);
});


// ── 两次运行之间 ──────────────────────────────────────────────────────────────
//
// 这一组守的是纪律：说不出显著差异的时候要说「测不出」，而不是把两个均值之差当结论。

test('不显著时不给方向', () => {
  const note = compareNote({ available: true, significant: false,
                             median_delta: 2.4, paired: 5 }, '导览');

  assert.match(note, /测不出显著差异/);
  assert.doesNotMatch(note, /更好|更差/);
});

test('不显著时也不着色', () => {
  // 一个绿色的「测不出显著差异」仍然会被读成「变好了」—— 颜色比字先被看见。
  const note = compareNote({ available: true, significant: false,
                             median_delta: 2.4, paired: 5 });

  assert.match(note, /bm-cmp-none/);
  assert.doesNotMatch(note, /bm-cmp--up|bm-cmp--down/);
});

test('显著变好才说更好，并且带上样本量', () => {
  const note = compareNote({ available: true, significant: true,
                             median_delta: 12.5, paired: 6 }, '导览');

  assert.match(note, /更好/);
  assert.match(note, /n=6/);
  assert.match(note, /bm-cmp--up/);
});

test('样本不够时把理由说出来，不是静默省略', () => {
  const note = compareNote({ available: false, reason: '配对样本只有 1 条，无法判断显著性' });

  assert.match(note, /只有 1 条/);
  assert.doesNotMatch(note, /更好|更差/);
});
