/**
 * Playback-start invariants for the audio renderer.
 *
 * WebKit only lets a context emit sound if a source was started during the user
 * gesture. This renderer withholds the first ~500ms in _PREBUF_CHUNKS, so its
 * first real source.start() is always outside the click — on Safari that meant a
 * waveform, a ⏸ button and silence, with nothing in the playback path looking
 * wrong. Tested with a stub context because Web Audio has no headless output.
 */
import { test } from 'node:test';
import assert from 'node:assert';
import { AudioRenderer } from './renderers/audio.js';

function stubContext({ throwOnSampleRate = false } = {}) {
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
  const Ctor = function (opts) {
    if (throwOnSampleRate && opts?.sampleRate) throw new TypeError('unsupported sampleRate');
    return ctx;
  };
  return { ctx, Ctor, started, sources };
}

/** A renderer instance cloned the way detail-panel.js clones it. */
function renderer(Ctor) {
  global.window = { AudioContext: Ctor };
  const r = Object.assign(Object.create(Object.getPrototypeOf(AudioRenderer)), AudioRenderer);
  r._sources = new Set();
  return r;
}

test('a source starts during the gesture, before any audio arrives', () => {
  const { Ctor, started } = stubContext();
  const r = renderer(Ctor);
  r._startPlay();
  assert.ok(started.length >= 1, 'WebKit stays silent without this');
});

test('a looping source keeps the output rendering after the gesture expires', () => {
  // WebKit's transient activation lapses seconds after the click; an idle graph
  // comes back silent while still reporting 'running'. Measured on R1: the first
  // chunk arrived 7.9s after ▶ and played to nobody.
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

test('unlocking happens once, not on every play/pause cycle', () => {
  const { Ctor, started } = stubContext();
  const r = renderer(Ctor);
  r._startPlay();
  r._stopPlay();
  r._startPlay();
  assert.equal(started.filter(s => !s.loop).length, 1, 'unlock ran more than once');
});

test('a context refusing a custom sampleRate still plays', () => {
  // Safari before 14.1 throws rather than ignoring the option; giving up on the
  // context would lose playback entirely, and buffers carry their own rate.
  const { Ctor, started } = stubContext({ throwOnSampleRate: true });
  const r = renderer(Ctor);
  r._startPlay();
  assert.ok(started.length >= 1);
});
