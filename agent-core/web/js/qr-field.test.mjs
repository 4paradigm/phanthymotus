/**
 * 插件配置页里的二维码字段 —— 模板替换与两条拒绝规则。
 *
 * 这里测的是判断逻辑，不是 DOM：仓库的前端测试跑在 node 里、没有 DOM，所以
 * 会出错的那部分（填什么、拒什么、算不算「要配置」）都放在纯函数里，DOM 组装
 * 保持薄到不值得测。
 *
 * 两条规则值得单独盯着，因为它们错了都不报错，只是悄悄错：
 *
 *  - `{origin}` 必须用**网卡地址**拼，不能用 `location.origin`。用错了二维码照
 *    样画得出来，扫出来是 `localhost` —— 手机拨不通，而且没有任何提示。
 *  - `{token}` 必须拒绝。放过去同样什么都不会发生，只是把一把能驱动电机的钥匙
 *    画进了一张谁都能拍的图里。
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { fillTemplate, referencesHost, requestsToken } from './local-address.js';
import { qrOptions, isDisplayOnly } from './qr-field.js';

const LOCAL = { host: '10.100.121.14', port: '15678', scheme: 'https' };

// ── 模板替换 ────────────────────────────────────────────────────────────────

test('四个占位符都被替换', () => {
  assert.equal(fillTemplate('{scheme}', LOCAL), 'https');
  assert.equal(fillTemplate('{host}', LOCAL), '10.100.121.14');
  assert.equal(fillTemplate('{port}', LOCAL), '15678');
  assert.equal(fillTemplate('{origin}', LOCAL), 'https://10.100.121.14:15678');
});

test('{origin} 拼的是网卡地址，不是 location.origin', () => {
  // 这条就是整个模板机制存在的理由：机器上本地开的浏览器 location.origin 是
  // localhost，编进二维码手机拨不通。
  const url = fillTemplate('{origin}/downloads/motus.apk', LOCAL);
  assert.equal(url, 'https://10.100.121.14:15678/downloads/motus.apk');
  assert.ok(!url.includes('localhost'), '不该出现 localhost');
  assert.ok(!url.includes('127.0.0.1'), '不该出现回环地址');
});

test('一条模板里多个占位符各自替换', () => {
  assert.equal(
    fillTemplate('motus://setup?h={host}&p={port}&s={scheme}', LOCAL),
    'motus://setup?h=10.100.121.14&p=15678&s=https');
});

test('不认识的占位符原样留着 —— 吞掉它只会让驱动作者以为自己写对了', () => {
  assert.equal(fillTemplate('{origin}/x?v={version}', LOCAL),
               'https://10.100.121.14:15678/x?v={version}');
});

test('没有占位符的固定外链原样返回', () => {
  const store = 'https://apps.apple.com/cn/app/id123456';
  assert.equal(fillTemplate(store, LOCAL), store);
});

test('端口为空时 {origin} 不留下一个光杆冒号', () => {
  // https 默认端口下 location.port 是空串
  assert.equal(fillTemplate('{origin}', { host: 'robot.local', port: '', scheme: 'https' }),
               'https://robot.local');
});

test('地址还没拿到时不产出半截 URL', () => {
  assert.equal(fillTemplate('{origin}/a', { host: '', port: '', scheme: 'https' }), '/a');
});

test('空模板不炸', () => {
  assert.equal(fillTemplate('', LOCAL), '');
  assert.equal(fillTemplate(undefined, LOCAL), '');
});

// ── 什么时候需要网卡选择器 ──────────────────────────────────────────────────

test('referencesHost 只对用到本机地址的模板为真', () => {
  assert.equal(referencesHost('{origin}/downloads/app.apk'), true);
  assert.equal(referencesHost('motus://setup?h={host}'), true);
  // 固定外链旁边摆一排网卡，会让人以为换网卡能换出不同的 App
  assert.equal(referencesHost('https://apps.apple.com/app/id123'), false);
  assert.equal(referencesHost(''), false);
});

// ── Token 拒绝 ──────────────────────────────────────────────────────────────

test('requestsToken 认出 {token}，大小写都算', () => {
  assert.equal(requestsToken('motus://bind?t={token}'), true);
  assert.equal(requestsToken('motus://bind?t={TOKEN}'), true);
  assert.equal(requestsToken('{origin}/downloads/app.apk'), false);
});

test('token 不在可替换的占位符里 —— 就算绕过拒绝也拼不出真值', () => {
  // 两道独立的闸：requestsToken 拦在渲染前，fillTemplate 本身也不认识它。
  // 只留一道的话，将来有人改了拒绝逻辑就会直接把 token 拼进 URL。
  const out = fillTemplate('x?t={token}', { ...LOCAL, token: 'SECRET' });
  assert.equal(out, 'x?t={token}');
  assert.ok(!out.includes('SECRET'));
});

// ── schema 解析 ─────────────────────────────────────────────────────────────

test('qrOptions 读出声明的选项', () => {
  const opts = qrOptions({ 'x-qr-options': [
    { label: 'iOS', url: 'https://apps.apple.com/app/id1' },
    { label: 'Android', url: '{origin}/downloads/a.apk' },
  ]});
  assert.deepEqual(opts.map(o => o.label), ['iOS', 'Android']);
  assert.equal(opts[1].url, '{origin}/downloads/a.apk');
});

test('写坏的选项被丢掉，而不是让整个弹窗白屏', () => {
  const opts = qrOptions({ 'x-qr-options': [
    { label: '好的', url: 'https://x.com' },
    { label: '没有 url' },
    { url: '   ' },
    null,
    'not an object',
  ]});
  assert.equal(opts.length, 1);
  assert.equal(opts[0].label, '好的');
});

test('没声明或声明成非数组时返回空数组', () => {
  assert.deepEqual(qrOptions({}), []);
  assert.deepEqual(qrOptions({ 'x-qr-options': 'ios,android' }), []);
  assert.deepEqual(qrOptions(undefined), []);
});

test('缺 label 的选项有个兜底名字，不是 undefined', () => {
  const opts = qrOptions({ 'x-qr-options': [{ url: 'https://x.com' }] });
  assert.equal(opts.length, 1);
  assert.ok(opts[0].label, 'label 不该为空');
  assert.ok(!String(opts[0].label).includes('undefined'));
});

// ── 「要配置」的判定 ────────────────────────────────────────────────────────

test('isDisplayOnly 只认 format: qr', () => {
  assert.equal(isDisplayOnly({ format: 'qr' }), true);
  assert.equal(isDisplayOnly({ type: 'string' }), false);
  assert.equal(isDisplayOnly({ format: 'password' }), false);
  assert.equal(isDisplayOnly(null), false);
});

/** sidebar.js 里算「已配置」用的那个判据，照搬一份钉住行为。 */
const needsInput = (props) => Object.values(props)
  .some(def => def.scope !== 'instance' && !isDisplayOnly(def));

test('只声明二维码的工具不算「待配置」—— 否则永久挂着警告三角', () => {
  // 它永远不会保存任何值，把它算成要配置的东西，那个黄色三角就再也下不去了
  assert.equal(needsInput({ install: { format: 'qr' } }), false);
});

test('二维码旁边还有真输入项时，仍然算待配置', () => {
  assert.equal(needsInput({
    install: { format: 'qr' },
    api_key: { type: 'string', format: 'password' },
  }), true);
});

test('只有 instance 字段时侧栏这一层不算待配置', () => {
  assert.equal(needsInput({ device_path: { type: 'string', scope: 'instance' } }), false);
});
