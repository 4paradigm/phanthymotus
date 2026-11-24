import assert from 'node:assert/strict';
import { test } from 'node:test';
import { openWecomOnboarding } from './wecom-onboarding.js';

function dom() {
  const elements = new Map();
  for (const id of ['[role=status]', 'img', '[data-retry]', '[data-close]']) {
    elements.set(id, { removeAttribute(name) { delete this[name]; } });
  }
  return { querySelector: name => elements.get(name) };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('WeCom QR displays authorization and connection states', async t => {
  const panel = dom();
  const calls = [];
  const clock = [];
  const states = [
    { status: 'waiting', qr_data_url: 'data:image/svg+xml;base64,abc' },
    { status: 'connecting', channel_id: 'wecom_test' },
    { status: 'connected', channel_id: 'wecom_test' },
  ];
  t.mock.method(globalThis, 'fetch', async url => {
    calls.push(url);
    return { ok: true, json: async () => ({ session_id: 'id', ...states.shift() }) };
  });
  globalThis.window = new EventTarget();
  t.mock.method(globalThis, 'setTimeout', func => clock.push(func));
  t.mock.method(globalThis, 'clearTimeout', () => {});
  const close = openWecomOnboarding(panel, () => {});
  await flush();
  assert.match(panel.querySelector('[role=status]').textContent, /企业微信扫码/);
  assert.equal(panel.querySelector('img').hidden, false);
  clock.shift()();
  await flush();
  assert.match(panel.querySelector('[role=status]').textContent, /正在连接企业微信/);
  assert.equal(panel.querySelector('img').hidden, true);
  clock.shift()();
  await flush();
  assert.match(panel.querySelector('[role=status]').textContent, /企业微信已连接/);
  assert.equal(clock.length, 0);
  close();
  assert.deepEqual(calls, [
    '/api/channel/wecom/qr/start', '/api/channel/wecom/qr/status',
    '/api/channel/wecom/qr/status', '/api/channel/wecom/qr/cancel',
  ]);
});

test('closing before QR creation finishes cancels the resulting session', async t => {
  const calls = [];
  let resolve;
  globalThis.window = new EventTarget();
  t.mock.method(globalThis, 'fetch', url => {
    calls.push(url);
    return calls.length === 1 ? new Promise(done => { resolve = done; }) : Promise.resolve({ ok: true });
  });
  const close = openWecomOnboarding(dom(), () => {});
  close();
  resolve({ ok: true, json: async () => ({ session_id: 'late', status: 'waiting' }) });
  await flush();
  assert.deepEqual(calls, ['/api/channel/wecom/qr/start', '/api/channel/wecom/qr/cancel']);
});
