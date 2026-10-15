import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KvLatestRenderer } from './renderers/kv-latest.js';
import { TextRenderer } from './renderers/text.js';

function element() {
  return { children: [], classList: { remove() {}, add() {} },
    appendChild(child) { this.children.push(child); }, remove() {} };
}

for (const base of [TextRenderer, KvLatestRenderer]) {
  test(`${base.name}: dedicated direction data uses standard rendering and sensor-owned freshness`, () => {
    const oldDocument = globalThis.document;
    const renderer = Object.assign({}, base);
    const send = data => renderer.onData(new TextEncoder().encode(JSON.stringify(data)));
    try {
      assert.equal(TextRenderer.canRender('sensor/sound-direction'), true);
      globalThis.document = { createElement: element };
      renderer.mount(element());
      send({ state: 'fresh', angle_deg: -29.678,
        angle_convention: 'robot_front_0_left_positive_ccw',
        timestamp_ms: 1700000000040, audio_window_start_us: 1700000000000000,
        audio_window_end_us: 1700000000020000 });
      send({ state: 'stale' });
      if (base === KvLatestRenderer) {
        assert.equal(renderer._cells.angle_deg.valueEl.textContent, '-29.678');
        assert.equal(renderer._cells.angle, undefined);
        assert.equal(renderer._cells.unit, undefined);
        assert.equal(renderer._cells.state.valueEl.textContent, 'stale');
      } else {
        assert.match(renderer._textEl.children[0].innerHTML, /-29.678/);
        assert.match(renderer._textEl.children.at(-1).innerHTML, /stale/);
      }
      send({ state: 'no_event' });
      if (base === KvLatestRenderer) {
        assert.equal(renderer._cells.state.valueEl.textContent, 'no_event');
      } else {
        assert.match(renderer._textEl.children.at(-1).innerHTML, /no_event/);
      }
    } finally {
      renderer.unmount();
      globalThis.document = oldDocument;
    }
  });
}
