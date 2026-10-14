import { test } from 'node:test';
import assert from 'node:assert/strict';
import { directionDisplayData, SoundDirectionRenderer } from './sound-direction.js';

const direction = { state: 1, angle_deg: -90, published_us: 1700000000040000,
  window_start_us: 1700000000000000, window_end_us: 1700000000020000 };

test('typed direction uses the previous JSON display fields and clockwise angle', () => {
  assert.deepEqual(directionDisplayData(direction), {
    state: 'fresh', trigger: 'sound_activity', timestamp_ms: 1700000000040,
    angle: 270, unit: 'deg', reference: 'robot_front_clockwise',
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
  test(`${mode} mode reuses JSON rendering and clears live direction on expiry`, () => {
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
        assert.equal(renderer._renderer._cells.angle.valueEl.textContent, '270');
      } else {
        assert.match(renderer._renderer._textEl.children[0].innerHTML, /angle/);
      }
      expire();
      if (mode === 'kv') {
        assert.equal(renderer._renderer._cells.angle.valueEl.textContent, '—');
        assert.equal(renderer._renderer._cells.state.valueEl.textContent, 'no_event');
      } else {
        assert.match(renderer._renderer._textEl.children.at(-1).innerHTML, /no_event/);
      }
    } finally {
      renderer.unmount();
      globalThis.document = oldDocument;
      globalThis.setTimeout = oldSetTimeout;
      globalThis.clearTimeout = oldClearTimeout;
    }
  });
}
