import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KvLatestRenderer } from './kv-latest.js';
import { TextRenderer } from './text.js';
import { directionDisplayData, SoundDirectionRenderer } from './sound-direction.js';

const direction = { state: 1, angle_deg: -90, published_us: 1700000000040000,
  window_start_us: 1700000000000000, window_end_us: 1700000000020000 };

test('typed direction uses the previous JSON display fields and signed angle', () => {
  assert.deepEqual(directionDisplayData(direction), {
    state: 'fresh', trigger: 'sound_activity', timestamp_ms: 1700000000040,
    angle: -90, unit: 'deg',
    audio_window_start_us: 1700000000000000, audio_window_end_us: 1700000000020000,
  });
  assert.equal(direction.angle_deg, -90); // Transport/ASR contract is unchanged.
});

test('no event contains no historical angle', () => {
  assert.deepEqual(directionDisplayData({ state: 0, angle_deg: 90 }), { state: 'no_event' });
});

function element() {
  return { children: [], classList: { remove() {}, add() {} },
    appendChild(child) { this.children.push(child); }, remove() {} };
}

for (const mode of ['log', 'kv']) {
  test(`${mode} mode reuses JSON rendering and retains the last reading with an explicit stale status`, () => {
    const oldDocument = globalThis.document;
    const oldSetTimeout = globalThis.setTimeout;
    const oldClearTimeout = globalThis.clearTimeout;
    let expire;
    const renderer = Object.assign({}, SoundDirectionRenderer, { mode });
    try {
      globalThis.document = { createElement: element };
      globalThis.setTimeout = fn => { expire = fn; return 1; };
      globalThis.clearTimeout = () => {};
      const container = element();
      renderer.mount(container);
      assert.equal(container.children[0].className,
        mode === 'kv' ? 'renderer-kv-latest' : 'renderer-text');
      renderer.onData(new TextEncoder().encode(JSON.stringify(direction)));
      if (mode === 'kv') {
        assert.equal(renderer._renderer._cells.angle.valueEl.textContent, '-90');
      } else {
        assert.match(renderer._renderer._textEl.children[0].innerHTML, /angle/);
      }
      expire();
      if (mode === 'kv') {
        assert.equal(renderer._renderer._cells.angle.valueEl.textContent, '-90');
        assert.equal(renderer._renderer._cells.state.valueEl.textContent, 'stale');
        assert.equal(renderer._renderer._cells.angle_status, undefined);
      } else {
        assert.match(renderer._renderer._textEl.children.at(-1).innerHTML, /stale/);
      }
      renderer.onData(new TextEncoder().encode(JSON.stringify({ state: 0 })));
      if (mode === 'kv') {
        assert.equal(renderer._renderer._cells.angle.valueEl.textContent, '-90');
        assert.equal(renderer._renderer._cells.state.valueEl.textContent, 'no_event');
      }
    } finally {
      renderer.unmount();
      globalThis.document = oldDocument;
      globalThis.setTimeout = oldSetTimeout;
      globalThis.clearTimeout = oldClearTimeout;
    }
  });
}

for (const angle of [0, 90, -90, 180, -179]) {
  test(`monitor does not change the transport angle ${angle}`, () => {
    assert.equal(directionDisplayData({ ...direction, angle_deg: angle }).angle, angle);
  });
}

for (const base of [TextRenderer, KvLatestRenderer]) {
  test(`${base.name} renders ASR wake angles as integers`, () => {
    const oldDocument = globalThis.document;
    const renderer = Object.assign({}, base);
    try {
      globalThis.document = { createElement: element };
      renderer.mount(element());
      renderer.onData(new TextEncoder().encode(JSON.stringify({
        wake_direction_deg: -29.678, audio_start_ts: 1700000000.123,
      })));
      if (base === KvLatestRenderer) {
        assert.equal(renderer._cells.wake_direction_deg.valueEl.textContent, '-30');
        assert.equal(renderer._cells.audio_start_ts.valueEl.textContent, '1700000000.123');
      } else {
        assert.match(renderer._textEl.children[0].innerHTML, /title="-30"/);
        assert.match(renderer._textEl.children[0].innerHTML, /1700000000.123/);
      }
    } finally {
      renderer.unmount();
      globalThis.document = oldDocument;
    }
  });
}
