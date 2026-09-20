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

import { blockerRows, runRefusal } from './benchmark.js';
import { parseList, weightNote, normalizeInjection } from './benchmark-editor.js';

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
  assert.match(message, /分不出来/);
});

test('多张真卡片都点出来，不是只说第一张', () => {
  const message = runRefusal({ detail: { unsafe: [
    { device: '天轶', tool: 'loco' }, { device: 'Go2', tool: 'switch_mode' },
  ] } });

  assert.match(message, /天轶 的 loco/);
  assert.match(message, /Go2 的 switch_mode/);
});

test('纯文本的拒绝理由原样说出来', () => {
  assert.match(runRefusal({ detail: '已经有一次基准测试在跑' }), /已经有一次基准测试在跑/);
});

test('结构化的理由不会退化成 [object Object]', () => {
  const message = runRefusal({ detail: { readiness: { missing_drivers: ['x'] } } });

  assert.doesNotMatch(message, /\[object Object\]/);
});

// ── 编辑器：几个必须自己算对的地方 ───────────────────────────────────────────

test('站序逗号分隔和换行分隔都认', () => {
  // 只认一种，另一种会安静地变成**一个**名字很长的站点 —— 跑起来一站都对不上，
  // 看着却像是 agent 走错了路。
  assert.deepEqual(parseList('P3, P4, P5'), ['P3', 'P4', 'P5']);
  assert.deepEqual(parseList('P3\nP4\nP5'), ['P3', 'P4', 'P5']);
  assert.deepEqual(parseList('入口，一号展区'), ['入口', '一号展区']);
});

test('站序里的空白与空行不会变成站点', () => {
  assert.deepEqual(parseList('  P3 ,, \n  P4  \n\n'), ['P3', 'P4']);
});

test('空的站序是空列表，不是一个空字符串站点', () => {
  assert.deepEqual(parseList(''), []);
  assert.deepEqual(parseList(null), []);
});

test('权重合计正好 100 时不啰嗦', () => {
  assert.equal(weightNote({ orchestration: 30, interruption: 25, long_horizon: 25,
                            safety: 15, latency: 5 }), '');
});

test('权重合计不是 100 只提示，不当成错误', () => {
  // 评分是按比例算的，105 照样能跑 —— 拦下来只会挡住人改权重。
  const note = weightNote({ orchestration: 40, interruption: 25, long_horizon: 25,
                            safety: 15, latency: 5 });

  assert.match(note, /110/);
  assert.match(note, /不影响跑/);
});

test('权重全是 0 要说出来', () => {
  assert.match(weightNote({}), /总分算不出来/);
});

test('一条插话只留一个触发方式', () => {
  // 两个都填，跑的时候按 after_arrival 走，而写的人以为按秒走。
  const byArrival = normalizeInjection({
    mode: 'arrive', after_arrival: 'P5', at: '90', delay: '6', text: '先等一下' });
  const bySecond = normalizeInjection({
    mode: 'at', after_arrival: 'P5', at: '90', delay: '0', text: '先等一下' });

  assert.deepEqual(byArrival, { text: '先等一下', delay: 6, after_arrival: 'P5' });
  assert.deepEqual(bySecond, { text: '先等一下', delay: 0, at: 90 });
});

test('插话的延时空着按 0 算，不是 NaN', () => {
  const row = normalizeInjection({ mode: 'arrive', after_arrival: 'P5', delay: '', text: '喂' });

  assert.equal(row.delay, 0);
});
