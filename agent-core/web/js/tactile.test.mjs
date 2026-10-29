import assert from 'node:assert/strict';
import test from 'node:test';
import { TactileRenderer, parseTactilePayload, tactileColor } from './renderers/tactile.js';

const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const frame = () => ({
  side: 'left', state: 'ok', scalar_valid: true, matrix_valid: true,
  touch_scalar: Object.fromEntries(fingers.map(name => [name, 0])),
  touch_matrix: Object.fromEntries(fingers.map(name => [name,
    Array.from({ length: 12 }, () => Array(6).fill(0))])),
});

test('sensor/tactile accepts valid zeros and rejects stale or malformed readings', () => {
  assert.equal(TactileRenderer.canRender('sensor/tactile'), true);
  assert.equal(TactileRenderer.canRender('data/json'), false);
  const payload = frame();
  payload.touch_matrix.index[0][0] = 255;
  let parsed = parseTactilePayload(new TextEncoder().encode(JSON.stringify(payload)).buffer);
  assert.equal(parsed.matrices.index[0][0], 255);
  assert.equal(parsed.scalars.thumb, 0);

  payload.scalar_valid = false;
  payload.touch_matrix.middle = [[0]];
  parsed = parseTactilePayload(payload);
  assert.equal(parsed.scalars.thumb, null);
  assert.equal(parsed.matrices.middle, null);
  assert.equal(parseTactilePayload('{bad json'), null);
});

test('tactile panel draws data and marks a missing finger instead of inventing zero', () => {
  const labels = [];
  const ctx = {
    fillRect() {}, fillText(value) { labels.push(value); },
  };
  const canvas = {
    style: {}, getContext: () => ctx, remove() {},
  };
  const oldDocument = globalThis.document;
  globalThis.document = { createElement: () => canvas };
  try {
    const renderer = Object.assign({}, TactileRenderer);
    renderer.mount({ appendChild() {} });
    const payload = frame();
    payload.touch_matrix.thumb = null;
    payload.touch_scalar.thumb = null;
    renderer.onData(new TextEncoder().encode(JSON.stringify(payload)).buffer);
    assert.ok(labels.includes('PARTIAL DATA'));
    assert.ok(labels.includes('NO DATA'));
    assert.ok(labels.includes('Tip: --'));
    assert.ok(labels.includes('Tip: 0'));
    renderer.unmount();
  } finally {
    globalThis.document = oldDocument;
  }
  assert.equal(tactileColor(0), 'rgb(26,34,52)');
  assert.equal(tactileColor(255), 'rgb(244,63,94)');
});
