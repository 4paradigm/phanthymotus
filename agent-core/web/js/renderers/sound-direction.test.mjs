import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatSoundDirection, SoundDirectionRenderer } from './sound-direction.js';

test('valid direction displays signed angle and capture window', () => {
  assert.equal(formatSoundDirection({ state: 1, angle_deg: -90,
    window_start_us: 1700000000000000, window_end_us: 1700000000020000 }),
    '-90° · 1700000000000000–1700000000020000 µs');
});

test('no event clears angle', () => {
  assert.equal(formatSoundDirection({ state: 0, angle_deg: 90 }), '暂无声源');
});

test('live angle clears after direction messages stop', () => {
  const oldDocument = globalThis.document;
  const oldSetTimeout = globalThis.setTimeout;
  const oldClearTimeout = globalThis.clearTimeout;
  let expire;
  try {
    globalThis.document = { createElement: () => ({ style: {}, textContent: '', remove() {} }) };
    globalThis.setTimeout = (fn) => { expire = fn; return 1; };
    globalThis.clearTimeout = () => {};
    const container = { appendChild() {} };
    SoundDirectionRenderer.mount(container);
    SoundDirectionRenderer.onData(new TextEncoder().encode(JSON.stringify({
      state: 1, angle_deg: 90, window_start_us: 1, window_end_us: 2,
    })));
    assert.match(SoundDirectionRenderer._el.textContent, /90°/);
    assert.equal(typeof expire, 'function');
    expire();
    assert.equal(SoundDirectionRenderer._el.textContent, '暂无声源');
  } finally {
    SoundDirectionRenderer.unmount();
    globalThis.document = oldDocument;
    globalThis.setTimeout = oldSetTimeout;
    globalThis.clearTimeout = oldClearTimeout;
  }
});
