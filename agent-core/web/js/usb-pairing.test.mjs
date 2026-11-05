import { test } from 'node:test';
import assert from 'node:assert/strict';
import { supportsUsbPairing, decodePairingResponse, callUsbPairing, pairingStep } from './usb-pairing.js';

test('only drivers declaring pairing expose the wizard', () => {
  assert.equal(supportsUsbPairing({ tools: ['hand', { name: 'glove_teleop' }] }), false);
  assert.equal(supportsUsbPairing({ tools: [{ name: 'usb_pairing' }] }), true);
  assert.equal(supportsUsbPairing(null), false);
});

test('MCP text responses and direct results decode consistently', () => {
  const result = { state: 'receiver', session_id: 'session-1' };
  assert.deepEqual(decodePairingResponse({ code: 200, data: [{ type: 'text', text: JSON.stringify(result) }] }), result);
  assert.deepEqual(decodePairingResponse({ code: 200, data: result }), result);
});

test('probe failure and malformed responses never advance the wizard', () => {
  assert.throws(() => decodePairingResponse({ code: 500, message: '检查电源' }), /检查电源/);
  assert.throws(() => decodePairingResponse({ code: 200, data: [{ type: 'text', text: 'not JSON' }] }), /无法读取/);
  assert.throws(() => decodePairingResponse({ code: 200, data: {} }), /有效配对状态/);
  assert.throws(() => pairingStep('unexpected'), /状态已变化/);
});

test('requests preserve the driver and session and never request motion', async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true, json: async () => ({ code: 200, data: { state: 'review' } }) };
  };
  await callUsbPairing('driver/one', 'hands', 'session-1', fetchImpl);
  assert.deepEqual(requests, [{ url: '/api/mcp/driver%2Fone/call',
    body: { tool: 'usb_pairing', arguments: { action: 'hands', session_id: 'session-1' } } }]);
  assert.equal(pairingStep('prepare').action, 'begin');
  assert.equal(pairingStep('receiver').action, 'receiver');
  assert.equal(pairingStep('hands').action, 'hands');
  assert.equal(pairingStep('review').action, 'apply');
  assert.equal(pairingStep('paired').action, undefined);
});

test('authentication and connection errors are surfaced', async () => {
  await assert.rejects(callUsbPairing('one', 'begin', '', async () => ({ ok: false, status: 401 })), /401/);
});
