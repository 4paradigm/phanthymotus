// control/velocity 的条形几何。
//
// 两个真机上报上来的症状，同一处代码：
//   「数字是 0，却有半条 bar」—— 条从左边缘画，而量程 [-2,2] 下 v=0 落在中点
//   「数字是 0.4，却没有 bar」—— 推断量程下该轴只出现过一个值，lo === hi
//
// 轨道中间画着一条零位刻度线，所以条本来就该以零位为锚。
import test from 'node:test';
import assert from 'node:assert';
import { ControlRenderer } from './renderers/control.js';

function paint(values, limits, names, message = {}) {
  const r = Object.create(ControlRenderer);
  r._rows = { appendChild() {} };
  r._bars = values.map(() => ({
    row: {}, fill: { style: {} }, value: {}, name: { title: 'old hint' },
  }));
  r._limits = limits;
  r._names = names;
  r._ensureRows = () => {};
  r._paint({ values, stamp_ms: Date.now(), obs_stamp_ms: Date.now(), seq: 1, ...message });
  return r._bars;
}

const SIGNED = { lower: [-2, -2], upper: [2, 2], declared: true };

test('零值在有符号量程里画不出条', () => {
  const [bar] = paint([0, 0], SIGNED);
  assert.strictEqual(parseFloat(bar.fill.style.width), 0);
});

test('正值从零位向右', () => {
  const [bar] = paint([1, 0], SIGNED);
  assert.strictEqual(parseFloat(bar.fill.style.left), 50);   // 零位在中点
  assert.strictEqual(parseFloat(bar.fill.style.width), 25);  // 1/4 量程
});

test('负值从零位向左，长度相同', () => {
  const [neg] = paint([-1, 0], SIGNED);
  assert.strictEqual(parseFloat(neg.fill.style.left), 25);
  assert.strictEqual(parseFloat(neg.fill.style.width), 25);
});

test('量程不含负数时零位就在左端', () => {
  const [bar] = paint([0.4, 0], { lower: [0, 0], upper: [1, 1], declared: true });
  assert.strictEqual(parseFloat(bar.fill.style.left), 0);
  assert.strictEqual(parseFloat(bar.fill.style.width), 40);
});

test('没有量程时不画条，但会说明是没量程而不是零', () => {
  const [bar] = paint([0.4, 0], { lower: [0.4, 0], upper: [0.4, 0] });
  assert.strictEqual(parseFloat(bar.fill.style.width), 0);
  assert.match(bar.row.title, /尚无量程/);
});

test('真的是零时不要谎称没量程', () => {
  const [, bar] = paint([0.4, 0], { lower: [0.4, 0], upper: [0.4, 0] });
  assert.strictEqual(bar.row.title, '');
});


test('twist 六轴使用速度名称与方向说明', () => {
  const r = Object.create(ControlRenderer);
  r._names = null; r._limits = null; r._seen = null;
  r._absorbDescriptor({ mode: 'twist', values: [0, 0, 0, 0, 0, 0] });
  const bars = paint([0, 0, 0, 0, 0, 0], r._limits, r._names, { mode: 'twist' });
  assert.deepEqual(bars.map(b => b.name.textContent), ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']);
  assert.match(bars[0].name.title, /前进/);
  assert.match(bars[5].name.title, /转向/);
});

test('v2 双末端使用十四维位姿名称并清除旧提示', () => {
  const bars = paint(Array(14).fill(0), { lower: [], upper: [] }, null, { mode: 'eef_pose', schema: 'motus.control/2' });
  const axes = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'];
  assert.deepEqual(bars.map(b => b.name.textContent), [1, 2].flatMap(i => axes.map(a => `末端${i}.${a}`)));
  assert(bars.every(b => b.name.title === ''));
});

test('明确提供的描述符名称优先于 twist 默认名称', () => {
  const r = Object.create(ControlRenderer);
  r._names = null; r._limits = null; r._seen = null;
  r._absorbDescriptor({ mode: 'twist', values: [0], control_interface: { joint_names: ['forward'] } });
  const [bar] = paint([0], r._limits, r._names, { mode: 'twist' });
  assert.equal(bar.name.textContent, 'forward');
  assert.equal(bar.name.title, '');
});
