/**
 * Playback-start invariants for the audio renderer.
 *
 * WebKit's transient user activation expires seconds after the click, and an
 * audio graph left idle in the meantime comes back silent — the context still
 * reports 'running' and buffers still schedule at sane times, so nothing in the
 * playback path looks wrong. Measured on R1 in Safari: with ▶ pressed 7.9s
 * before the first chunk there was no sound; at 1.7s there was. A looping silent
 * source keeps the output rendering, which removes that timing dependency —
 * verified by a >10s gap playing audibly.
 *
 * Tested with a stub context because Web Audio has no headless output.
 */
import { test } from 'node:test';
import assert from 'node:assert';
import { AudioRenderer } from './renderers/audio.js';

function stubContext() {
  const started = [];
  const sources = [];
  const ctx = {
    state: 'suspended', sampleRate: 48000, currentTime: 0,
    resumed: false,
    resume() { this.resumed = true; this.state = 'running'; },
    createBuffer: (ch, len, rate) => ({ duration: len / rate, getChannelData: () => new Float32Array(len) }),
    createBufferSource: () => {
      const src = { buffer: null, loop: false, stopped: false,
                    connect() {}, start(when) { started.push({ when, loop: this.loop }); },
                    stop() { this.stopped = true; } };
      sources.push(src);
      return src;
    },
    destination: {},
  };
  const Ctor = function () { return ctx; };
  return { ctx, Ctor, started, sources };
}

/** A renderer instance cloned the way detail-panel.js clones it. */
function renderer(Ctor) {
  global.window = { AudioContext: Ctor };
  const r = Object.assign(Object.create(Object.getPrototypeOf(AudioRenderer)), AudioRenderer);
  r._sources = new Set();
  return r;
}

test('a looping source starts inside the gesture and keeps the output rendering', () => {
  const { Ctor, started } = stubContext();
  const r = renderer(Ctor);
  r._startPlay();
  assert.ok(started.some(s => s.loop), 'no keep-alive source was started');
});

test('the keep-alive stops when playback is paused', () => {
  const { Ctor, sources } = stubContext();
  const r = renderer(Ctor);
  r._startPlay();
  const keep = sources.find(s => s.loop);
  r._stopPlay();
  assert.equal(keep.stopped, true, 'a silent source would keep running forever');
  assert.equal(r._keepAlive, null);
});

test('the context is resumed as well', () => {
  const { ctx, Ctor } = stubContext();
  const r = renderer(Ctor);
  r._startPlay();
  assert.ok(ctx.resumed);
  assert.equal(r._playing, true);
});
