import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { openQQOnboarding } from './qq-onboarding.js';

function dom() {
  const elements = new Map();
  for (const id of ['[role=status]', 'img', '[data-retry]', '[data-close]']) {
    elements.set(id, { removeAttribute(name) { delete this[name]; } });
  }
  return { querySelector: name => elements.get(name) };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('selecting QQ starts QR, polls and distinguishes saved from ready', async t => {
  const panel = dom();
  const calls = [];
  const clock = [];
  const states = [
    { status: 'waiting', qr_data_url: 'data:image/svg+xml;base64,abc' },
    { status: 'gateway_pending', channel_id: 'qq_test' },
    { status: 'canvas_pending', channel_id: 'qq_test' },
    { status: 'connected', channel_id: 'qq_test' },
    { status: 'connected', channel_id: 'qq_test' },
  ];
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url.split('/').at(-1));
    return { ok: true, json: async () => ({ session_id: 'id', ...states.shift() }) };
  });
  globalThis.window = new EventTarget();
  t.mock.method(globalThis, 'setTimeout', func => clock.push(func));
  t.mock.method(globalThis, 'clearTimeout', () => {});
  let closed = false;
  const close = openQQOnboarding(panel, () => { closed = true; });
  await flush();
  assert.equal(panel.querySelector('img').hidden, false);
  assert.equal(panel.querySelector('[role=status]').textContent, '请使用 QQ 扫码创建并连接 QQ 机器人');
  for (const expected of ['正在连接 QQ Gateway', '开启智能控制', '消息输入已就绪']) {
    clock.shift()();
    await flush();
    assert.match(panel.querySelector('[role=status]').textContent, new RegExp(expected));
    assert.equal(panel.querySelector('img').hidden, true);
  }
  assert.equal(clock.length, 0);
  close();
  assert.equal(closed, true);
  assert.deepEqual(calls, ['start', 'status', 'status', 'status', 'cancel']);
});

test('cancel during start cancels the late session', async t => {
  const calls = [];
  let resolve;
  globalThis.window = new EventTarget();
  t.mock.method(globalThis, 'fetch', url => {
    calls.push(url.split('/').at(-1));
    return calls.length === 1 ? new Promise(done => { resolve = done; }) : Promise.resolve({ ok: true });
  });
  const close = openQQOnboarding(dom(), () => {});
  close();
  resolve({ ok: true, json: async () => ({ session_id: 'late', status: 'waiting' }) });
  await flush();
  assert.deepEqual(calls, ['start', 'cancel']);
});

test('refresh and opening Channels do not open the scanner', async () => {
  const source = await readFile(new URL('./channels.js', import.meta.url), 'utf8');
  assert.match(source, /platformSel\.addEventListener\('change',[\s\S]*?platformSel\.value !== 'qq'[\s\S]*?openQQOnboarding\(form/);
  assert.doesNotMatch(source.match(/function _open\(\) \{[^}]+}/)[0], /openQQOnboarding/);
});
