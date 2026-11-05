/**
 * Starting `remote_mic` / `remote_camera` with the device saved on the card.
 *
 * Both cards saved a device and nothing ever read it back — the streamers were
 * called with no options, so the browser opened its default. That is invisible
 * whenever the default is the one chosen, which is why it took a U1 Pro whose
 * default was not to find it. These tests pin the three states apart (nothing
 * chosen / chosen but gone / config unreadable) and the start/stop lifecycle
 * that the same change had to fix: two starts racing into two streams, and a
 * stop during a start that did not stop it.
 *
 * getUserMedia, AudioContext and WebSocket are faked on globalThis, so this
 * runs in node with no browser.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  fetchInstanceConfig, micOptions, cameraOptions, deviceConstraint, describeMediaError,
} from './browser-source.js';

// ── fakes ────────────────────────────────────────────────────────────────────

function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const media = { calls: [], pending: [], stoppedTracks: 0 };

function fakeStream(label) {
  const track = { label, stop() { media.stoppedTracks++; } };
  return {
    getTracks: () => [track],
    getAudioTracks: () => [track],
    getVideoTracks: () => [track],
  };
}

class FakeWebSocket {
  static OPEN = 1;
  static all = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    this.closed = false;
    FakeWebSocket.all.push(this);
    queueMicrotask(() => { this.readyState = 1; this.onopen && this.onopen(); });
  }
  send(data) { this.sent.push(data); }
  close() { this.closed = true; this.readyState = 3; }
}

class FakeAudioContext {
  static all = [];
  static startState = 'running';
  constructor() {
    this.sampleRate = 48000; this.closed = false; this.state = FakeAudioContext.startState;
    this.resumes = 0; FakeAudioContext.all.push(this);
  }
  resume() { this.resumes++; if (this.unlocked) this.state = 'running'; return Promise.resolve(); }
  createMediaStreamSource() { return { connect() {} }; }
  createScriptProcessor() { this.processor = { connect() {}, disconnect() {} }; return this.processor; }
  createGain() { return { gain: {}, connect() {} }; }
  get destination() { return {}; }
  close() { this.closed = true; return Promise.resolve(); }
}

Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    mediaDevices: {
      getUserMedia(constraints) {
        media.calls.push(constraints);
        const d = deferred();
        media.pending.push(d);
        return d.promise;
      },
    },
  },
});
globalThis.WebSocket = FakeWebSocket;
globalThis.AudioContext = FakeAudioContext;
globalThis.localStorage = { getItem: () => null };
globalThis.location = { search: '' };

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Resolve the oldest pending getUserMedia once it has been asked for. */
async function grant(label = 'Mic B') {
  for (let i = 0; i < 20 && !media.pending.length; i++) await flush();
  const d = media.pending.shift();
  assert.ok(d, 'getUserMedia was never called');
  d.resolve(fakeStream(label));
  await flush();
}

const gestureListeners = [];

const mic = await import('./mic-stream.js');

beforeEach(() => {
  mic.stopMicStream();
  media.calls.length = 0;
  media.pending.length = 0;
  media.stoppedTracks = 0;
  FakeWebSocket.all.length = 0;
  FakeAudioContext.all.length = 0;
  FakeAudioContext.startState = 'running';
  gestureListeners.length = 0;
});

const DEVICE = 'b'.repeat(64);

// ── the card's config → streamer options ────────────────────────────────────

test('a saved device becomes an exact constraint, not a preference', () => {
  // A bare `deviceId` is a hint the browser may ignore and hand back its
  // default — the very failure being fixed.
  assert.deepEqual(deviceConstraint(DEVICE), { deviceId: { exact: DEVICE } });
  assert.deepEqual(deviceConstraint(''), {});
  assert.deepEqual(mic.micConstraints(DEVICE).deviceId, { exact: DEVICE });
});

test('no saved device keeps the browser default', () => {
  for (const cfg of [{}, null, undefined, { device_id: '' }, { device_id: '   ' }, { device_id: 7 }]) {
    assert.deepEqual(micOptions(cfg), { deviceId: '' }, JSON.stringify(cfg));
  }
  assert.equal('deviceId' in mic.micConstraints(''), false);
});

test('the mic constraint keeps its processing switches alongside the device', () => {
  const c = mic.micConstraints(DEVICE, true);
  assert.equal(c.echoCancellation, false);
  assert.equal(c.sampleRate, 16000);
});

test('the camera card passes its device, fps and width — all three were dropped', () => {
  assert.deepEqual(cameraOptions({ device_id: DEVICE, fps: 5, width: 1280 }),
    { deviceId: DEVICE, fps: 5, width: 1280 });
});

test('unset camera numbers are left out so the streamer defaults apply', () => {
  // `null` and '' are not 0 here: Number() would turn both into 0 and a 0 fps
  // clamps to 1, which reads as a stalled camera.
  assert.deepEqual(cameraOptions({ fps: null, width: '' }), { deviceId: '' });
  assert.deepEqual(cameraOptions({ fps: 0, width: 'wide' }), { deviceId: '' });
  assert.deepEqual(cameraOptions(null), { deviceId: '' });
});

// ── reading the config ──────────────────────────────────────────────────────

function fakeFetch(status, body, seen = []) {
  return async (url, init) => {
    seen.push({ url, init });
    return { ok: status >= 200 && status < 300, status, json: async () => body };
  };
}

test('the config is read per instance and bypasses the HTTP cache', async () => {
  const seen = [];
  const cfg = await fetchInstanceConfig('agentcore', 'remote_mic', 'card 1',
    fakeFetch(200, { code: 200, data: { device_id: DEVICE } }, seen));
  assert.deepEqual(cfg, { device_id: DEVICE });
  assert.equal(seen[0].url, '/api/canvas/tool-config/agentcore/remote_mic/card%201');
  assert.equal(seen[0].init.cache, 'no-store');
});

test('nothing saved reads as an empty config', async () => {
  assert.deepEqual(
    await fetchInstanceConfig('agentcore', 'remote_mic', 'c', fakeFetch(200, { code: 200, data: null })), {});
});

test('a failed read is an error, never "no device chosen"', async () => {
  // Otherwise a 401 would quietly open the default microphone.
  await assert.rejects(
    fetchInstanceConfig('agentcore', 'remote_mic', 'c', fakeFetch(401, {})), /HTTP 401/);
  await assert.rejects(
    fetchInstanceConfig('agentcore', 'remote_mic', 'c', fakeFetch(200, { code: 500 })), /code=500/);
});

// ── error messages an operator can act on ───────────────────────────────────

function named(name, message = name) { const e = new Error(message); e.name = name; return e; }

test('a chosen device that is gone asks for it to be re-picked', () => {
  for (const name of ['OverconstrainedError', 'NotFoundError']) {
    const msg = describeMediaError(named(name), 'microphone', DEVICE);
    assert.match(msg, /remote_mic 配置里重新选择/, name);
  }
  assert.match(describeMediaError(named('OverconstrainedError'), 'camera', DEVICE),
    /remote_camera 配置里重新选择/);
});

test('permission, absence and a busy device are told apart', () => {
  assert.match(describeMediaError(named('NotAllowedError'), 'microphone'), /权限被拒绝/);
  assert.match(describeMediaError(named('NotFoundError'), 'microphone', ''), /没有可用的麦克风/);
  assert.match(describeMediaError(named('NotReadableError'), 'camera'), /被其他程序占用/);
  assert.match(describeMediaError(new Error('读取 remote_mic 配置失败（HTTP 401）'), 'microphone'),
    /HTTP 401/);
});

// ── the microphone lifecycle ────────────────────────────────────────────────

test('the saved device reaches getUserMedia', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({ deviceId: DEVICE }));
  await grant();
  assert.equal(await run, true);
  assert.deepEqual(media.calls[0].audio.deviceId, { exact: DEVICE });
  assert.equal(mic.isMicActive(), true);
});

test('two starts at once open one stream, not two', async () => {
  // start-project and the project_state event both arrive while the first
  // start is still waiting on getUserMedia; `isMicActive()` is false for both.
  const a = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  const b = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  assert.equal(a, b, 'the second caller joins the first start');
  await grant();
  await a;
  assert.equal(media.calls.length, 1);
  assert.equal(FakeWebSocket.all.length, 1);
});

test('a stop during getUserMedia cancels the start and releases the device', async () => {
  const states = [];
  const run = mic.startMicStream('wss://h/ws/mic', (s) => states.push(s), async () => ({}));
  await flush();
  mic.stopMicStream();               // the operator stops the project here
  await grant();                     // ...and only then does the browser answer
  assert.equal(await run, false);
  assert.equal(mic.isMicActive(), false);
  assert.equal(media.stoppedTracks, 1, 'the late stream is stopped, not leaked');
  assert.equal(FakeWebSocket.all.length, 0, 'no socket is opened after the stop');
  assert.deepEqual(states, [], 'never reported as recording');
});

test('a stop while the config is still being read cancels before any device opens', async () => {
  const cfg = deferred();
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, () => cfg.promise);
  mic.stopMicStream();
  cfg.resolve({ deviceId: DEVICE });
  assert.equal(await run, false);
  assert.equal(media.calls.length, 0);
});

test('a config read failure does not open the default microphone', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => {
    throw new Error('读取 remote_mic 配置失败（HTTP 401）');
  });
  await assert.rejects(run, /HTTP 401/);
  assert.equal(media.calls.length, 0);
  assert.equal(mic.isMicActive(), false);
});

test('a device that is gone fails loudly instead of falling back', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({ deviceId: DEVICE }));
  for (let i = 0; i < 20 && !media.pending.length; i++) await flush();
  media.pending.shift().reject(named('OverconstrainedError'));
  await assert.rejects(run, (err) => err.name === 'OverconstrainedError');
  assert.equal(media.calls.length, 1, 'no second getUserMedia without the device');
  assert.equal(mic.isMicActive(), false);
});

test('stop, change the device, start again: the new device is used', async () => {
  let saved = 'device-b';
  const opts = async () => ({ deviceId: saved });
  const first = mic.startMicStream('wss://h/ws/mic', () => {}, opts);
  await grant('B');
  await first;
  mic.stopMicStream();
  saved = 'device-c';
  const second = mic.startMicStream('wss://h/ws/mic', () => {}, opts);
  await grant('C');
  await second;
  assert.deepEqual(media.calls.map((c) => c.audio.deviceId.exact), ['device-b', 'device-c']);
});

test('a running stream still converts 48 kHz to 16 kHz Int16 and sends it', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  await grant();
  await run;
  const proc = FakeAudioContext.all[0].processor;
  proc.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array(4800).fill(0.5) } });
  const sent = FakeWebSocket.all[0].sent;
  assert.equal(sent.length, 1);
  assert.equal(new Int16Array(sent[0]).length, 1600);
});

test('a mic started without a gesture says so, and the first click unlocks it', async () => {
  // The post-refresh re-arm: nobody has clicked, so the AudioContext starts
  // suspended and onaudioprocess never runs — an open socket sending nothing.
  FakeAudioContext.startState = 'suspended';
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  await grant();
  await run;
  const ctx = FakeAudioContext.all[0];
  assert.equal(mic.isMicSuspended(), true);
  assert.ok(ctx.resumes >= 1, 'it tries resume() straight away');
  assert.deepEqual(gestureListeners.map((l) => l.type).sort(), ['keydown', 'pointerdown']);

  ctx.unlocked = true;                 // the browser now allows it...
  gestureListeners.find((l) => l.type === 'pointerdown').fn();   // ...on a click
  assert.equal(ctx.state, 'running');
  assert.equal(mic.isMicSuspended(), false);
});

test('a mic started by a click is not reported as suspended', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  await grant();
  await run;
  assert.equal(mic.isMicSuspended(), false);
  assert.equal(gestureListeners.length, 0);
});

test('stop releases the track, the audio context and the socket', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  await grant();
  await run;
  mic.stopMicStream();
  assert.equal(media.stoppedTracks, 1);
  assert.equal(FakeAudioContext.all[0].closed, true);
  assert.equal(FakeWebSocket.all[0].closed, true);
});

test('the legacy toggle still stops a start that is in flight', async () => {
  const run = mic.startMicStream('wss://h/ws/mic', () => {}, async () => ({}));
  const states = [];
  await mic.toggleMicStream('', (s) => states.push(s));
  assert.equal(await run, false);
  assert.equal(media.calls.length, 0, 'cancelled before any device was opened');
  assert.deepEqual(states, [false]);
});

// ── the camera lifecycle ────────────────────────────────────────────────────

const videos = [];
globalThis.document = {
  addEventListener(type, fn) { gestureListeners.push({ type, fn }); },
  body: { appendChild() {} },
  createElement(tag) {
    if (tag === 'video') {
      const v = { style: {}, videoWidth: 1280, videoHeight: 720, removed: false,
        play: async () => {}, remove() { this.removed = true; } };
      videos.push(v);
      return v;
    }
    return { getContext: () => ({ drawImage() {} }), toBlob() {} };   // canvas
  },
};
globalThis.performance ??= { now: () => Date.now() };

const camera = await import('./camera-stream.js');

test('the camera card\'s device reaches getUserMedia, and its width sizes the frame', async () => {
  camera.stopCameraStream();
  media.calls.length = 0; media.pending.length = 0;
  const run = camera.startCameraStream('wss://h/ws/camera', () => {},
    async () => cameraOptions({ device_id: DEVICE, fps: 5, width: 320 }));
  await grant('Cam B');
  assert.equal(await run, true);
  assert.deepEqual(media.calls[0].video, { deviceId: { exact: DEVICE } });
  camera.stopCameraStream();
});

test('a camera stop during getUserMedia cancels the start and releases the device', async () => {
  camera.stopCameraStream();
  media.calls.length = 0; media.pending.length = 0; media.stoppedTracks = 0;
  videos.length = 0; FakeWebSocket.all.length = 0;
  const run = camera.startCameraStream('wss://h/ws/camera', () => {}, async () => ({}));
  await flush();
  camera.stopCameraStream();
  await grant('Cam');
  assert.equal(await run, false);
  assert.equal(camera.isCameraActive(), false);
  assert.equal(media.stoppedTracks, 1);
  assert.equal(videos.length, 0, 'no video element is created after the stop');
  assert.equal(FakeWebSocket.all.length, 0);
});
