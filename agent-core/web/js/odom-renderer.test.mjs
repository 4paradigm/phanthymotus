import test from 'node:test';
import assert from 'node:assert/strict';
import { KvLatestRenderer } from './renderers/kv-latest.js';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { TextRenderer } from './renderers/text.js';

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.dataset = {};
    this.className = ''; this._text = ''; this.hidden = false;
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  set textContent(value) { this._text = String(value ?? ''); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); child.parentElement = null; }
  remove() { this.parentElement?.removeChild(this); }
  addEventListener(name, fn) { this['on' + name] = fn; }
  select() { document.activeElement = this; }
  querySelectorAll(selector) {
    const match = el => selector[0] === '.' ? el.className.split(' ').includes(selector.slice(1))
      : el.tagName === selector.toUpperCase();
    return this.children.flatMap(child => [...(match(child) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  get firstChild() { return this.children[0]; }
}

function mount(template, options) {
  globalThis.document = { createElement: tag => new Element(tag), createTextNode: text => {
    const el = new Element('text'); el.textContent = text; return el;
  } };
  const box = new Element();
  const renderer = Object.assign({}, template);
  renderer.mount(box, options);
  return { box, renderer, send: value => renderer.onData(new TextEncoder().encode(JSON.stringify(value)).buffer) };
}

test('ordinary JSON latest keeps its original accumulated-field update behavior', () => {
  const { box, renderer, send } = mount(KvLatestRenderer);
  try {
    send({ preserved: 7, reading: { n: 1 } });
    send({ reading: { n: 0 } });
    assert.match(box.textContent, /preserved7/);
    assert.match(box.textContent, /reading.n0/);
    assert.equal(box.querySelector('.odom-summary'), null);
    assert.equal(box.querySelector('.odom-raw'), null);
  } finally { renderer.unmount(); }
});

const odom = () => ({ schema: 'motus.odom/1', stamp_ms: 1234, frame: 'body',
  twist: [0, null, null, null, null, 0.5],
  pose: { position: [1, 2, 3], quaternion: [0, 0, 0, 1], frame_id: 'odom', child_frame_id: 'camera' },
  vendor: { displacement_m: 44.444, turn_angle_deg: 99.999,
    motus_slam: { status: 'tracking', segment_id: 'run-1', reason: 'ok', stamp_source: 'measurement' } },
});

test('odom summary rejects nonnumeric and nonfinite measured axes without changing raw data', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer } = mount(OdomRenderer);
  try {
    const raw = '{"schema":"motus.odom/1","pose":{"position":[1e400,2,false]},"twist":[1e400,true,null,null,null,"0"],"vendor":{"motus_slam":{"status":"tracking"}}}';
    renderer.onData(raw);
    const summary=box.querySelector('.odom-summary').textContent;
    assert.doesNotMatch(summary,/Infinity|false|true/);
    assert.match(summary,/位置—, 2, — m/);
    assert.match(summary,/vx—/);assert.match(summary,/vy—/);assert.match(summary,/wz—/);
    assert.match(box.querySelector('.odom-raw').textContent,/1e400/);
  } finally { renderer.unmount(); }
});

test('one odometry profile clears untrusted pose while preserving measured zero and raw history', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer, send } = mount(OdomRenderer, { format: 'state/odom' });
  try {
    const sample = odom(); send(sample);
    assert.equal(box.querySelector('.odom-log'), null, 'default is latest');
    const summary = () => box.querySelector('.odom-summary').textContent;
    assert.match(summary(), /1, 2, 3/);
    assert.match(summary(), /camera|odom/);
    assert.match(summary(), /vx0 m\/s/);
    assert.match(summary(), /vy—/);
    assert.doesNotMatch(summary(), /44.444|99.999/);
    for (const status of ['lost', 'degraded', 'unknown']) {
      sample.vendor.motus_slam.status = status; send(sample);
      assert.match(summary(), new RegExp(status));
      assert.doesNotMatch(summary(), /1, 2, 3/);
    }
    sample.pose = null; send(sample);
    assert.match(summary(), /位置—/);
    renderer.setMode('log');
    assert.ok(box.querySelector('.odom-log'));
    assert.match(box.querySelector('.odom-summary').textContent, /1, 2, 3/);
    assert.match(box.querySelector('.odom-raw').textContent, /44.444/);
    renderer.setMode('kv');
    assert.doesNotMatch(summary(), /1, 2, 3/);
    send({ schema: 'motus.odom/1', twist: [0, null, null, null, null, 0] });
    assert.match(summary(), /unknown/);
    assert.match(summary(), /位置—/);
  } finally { renderer.unmount(); }
});

test('standard odometry needs no presentation configuration and preserves exact raw history', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer, send } = mount(OdomRenderer, { format: 'state/odom', showModeToggle: false });
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let copied;
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
    clipboard: { writeText: async text => { copied = text; } },
  } });
  try {
    const sample = odom(); sample.vendor.diagnostic = '{"n":0}';
    const raw = JSON.stringify(sample, null, 2) + '\n';
    renderer.onData(new TextEncoder().encode(raw).buffer);
    assert.equal(box.querySelector('.odom-modes'), null);
    const summary = box.querySelector('.odom-summary').textContent;
    assert.match(summary, /位置1, 2, 3 m/); assert.match(summary, /vx0 m\/s/);
    assert.doesNotMatch(summary, /diagnostic|44.444/);
    assert.match(box.querySelector('.odom-detail').textContent, /完整位姿/);
    assert.equal(box.querySelector('.odom-raw').querySelector('pre').textContent, raw);
    await box.querySelector('.odom-copy').onclick(); assert.equal(copied, raw);
    renderer.setMode('log');
    assert.equal(box.querySelector('.odom-summary').textContent, summary);
    assert.equal(box.querySelector('.odom-raw').querySelector('pre').textContent, raw);
    sample.vendor.motus_slam.status = 'lost'; send(sample);
    const entries = box.querySelectorAll('.odom-record');
    assert.match(entries[0].querySelector('.odom-summary').textContent, /位置1, 2, 3 m/);
    assert.match(entries[1].querySelector('.odom-summary').textContent, /位置—/);
    for (let i = 0; i < 60; i++) send(sample);
    assert.equal(box.querySelectorAll('.odom-record').length, 50);
  } finally {
    renderer.unmount();
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator);
    else delete globalThis.navigator;
  }
});

test('receipt freshness is separate from unknown acquisition age and ignores keepalives', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const savedNow = Date.now, savedInterval = globalThis.setInterval, savedClear = globalThis.clearInterval;
  let now = 1000, tick, cleared = false;
  Date.now = () => now; globalThis.setInterval = fn => { tick = fn; return 42; };
  globalThis.clearInterval = () => { cleared = true; };
  const { box, renderer, send } = mount(OdomRenderer, { format: 'state/odom' });
  try {
    send(odom());
    assert.match(box.querySelector('.odom-summary').textContent, /测量年龄未知/);
    assert.match(box.querySelector('.odom-freshness').textContent, /0/);
    now = 7001; send({ type: 'ping' }); tick();
    assert.match(box.querySelector('.odom-freshness').textContent, /陈旧/);
    assert.match(box.querySelector('.odom-summary').textContent, /tracking/);
    renderer.unmount(); assert.equal(cleared, true);
  } finally { Date.now = savedNow; globalThis.setInterval = savedInterval; globalThis.clearInterval = savedClear; }
});

test('odometry preview prefers odometry without reordering output ports', async () => {
  const { odomPreview, OdomRenderer } = await import('./renderers/odom.js');
  const first = { topic: '/status', format: 'data/json' };
  const pose = { topic: '/odom', format: 'state/odom', metadata: { presentation: { fields: [] } } };
  const skill = { ws_path: '/ws/bus/status', render_hint: 'data/json', topic_out: [first, pose] };
  assert.deepEqual(odomPreview(skill), { path: '/ws/bus/odom', hint: 'state/odom', topic: pose });
  assert.equal(skill.topic_out[0], first);
  assert.equal(odomPreview({ ws_path: '/ws/bus/other', render_hint: 'data/json' }).path, '/ws/bus/other');
  assert.equal(OdomRenderer.canRender('state/odom'), true);
  assert.equal(OdomRenderer.canRender('data/json'), false);
  assert.equal(OdomRenderer.canRender('text/plain'), false);
});

test('detail and dashboard specialize odometry while ordinary JSON keeps its original text route', async () => {
  const { OdomRenderer, odomPreview } = await import('./renderers/odom.js');
  const profile = { schema: 'motus.presentation/1', fields: [{ path: 'voltage', label: '电压', unit: 'V' }] };
  for (const [view, format] of ['detail-panel', 'dashboard'].flatMap(view =>
    ['data/json', 'state/odom'].map(format => [view, format]))) {
    const elements = new Map(); const sockets = [];
    mount(KvLatestRenderer).renderer.unmount(); // install the same small DOM used above
    globalThis.document.getElementById = id => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    };
    const source = readFileSync(new URL(`./${view}.js`, import.meta.url), 'utf8');
    const inert = { canRender: () => false };
    const names = source.match(/const RENDERERS = \[([^\]]+)\]/)[1].split(',').map(name => name.trim());
    const context = { ...Object.fromEntries(names.map(name => [name, inert])),
      OdomRenderer, TextRenderer, odomPreview,
      ActivityRenderer: inert, document: globalThis.document, TextEncoder, TextDecoder, ArrayBuffer,
      console: { log() {}, debug() {}, warn() {} }, location: { protocol: 'http:', host: 'localhost' },
      setTimeout() {}, clearTimeout() {}, setInterval() {}, clearInterval() {},
      openDetailPanelMobile() {}, closeDetailPanelMobile() {}, onMotusEvent() {}, offMotusEvent() {},
      WebSocket: class { constructor() { sockets.push(this); } close() {} },
    };
    const api = view === 'detail-panel'
      ? '{ initDetailPanel, showTopicDetail, close: _cleanup, textLike: () => _textLike }'
      : '{ mountRenderer, close: () => _activeRenderer.unmount() }';
    const exposed = runInNewContext(source.replace(/^import .*;\n/gm, '').replace(/^export /gm, '') + '\n' + `(${api})`, context);
    try {
      if (view === 'detail-panel') {
        exposed.initDetailPanel(); exposed.showTopicDetail('/custom/status', format, { presentation: profile });
      } else exposed.mountRenderer({ id: 'custom', topic_out: [{ topic: '/custom/status', format,
        metadata: { presentation: profile } }], ws_path: '/ws/bus/custom/status', render_hint: format });
      const body = elements.get(view === 'detail-panel' ? 'detail-body' : 'renderer-content');
      if (format === 'state/odom') {
        assert.ok(body.querySelector('.renderer-odom'), `${view} must render odometry directly`);
        sockets[0].onmessage({ data: JSON.stringify(odom()) });
        assert.match(body.querySelector('.odom-summary').textContent, /位置1, 2, 3 m/);
        assert.match(body.querySelector('.odom-raw').textContent, /motus.odom/);
      } else {
        assert.ok(body.querySelector('.renderer-text'), `${view} must keep ordinary JSON on TextRenderer`);
        sockets[0].onmessage({ data: '{"voltage":0,"diagnostics":{"all":true}}' });
        assert.ok(body.querySelector('.log-entry'));
        assert.equal(body.querySelector('.renderer-odom'), null);
        assert.equal(body.querySelector('.odom-summary'), null);
      }
      if (view === 'detail-panel') assert.equal(exposed.textLike(), true, 'canvas stale notice cannot cover text or odometry');
    } finally { exposed.close(); }
  }
});

test('text and ASR continue to use the existing streaming text behavior', () => {
  const { box, renderer, send } = mount(TextRenderer);
  try {
    send({ text: '你好', asr_complete_ts: 1234 });
    assert.equal(box.querySelector('.asr-line').textContent, 'ASR你好');
    renderer.onData(new TextEncoder().encode('plain streaming text').buffer);
    assert.match(box.textContent, /plain streaming text/);
    assert.equal(box.querySelector('.renderer-odom'), null);
  } finally { renderer.unmount(); }
});

test('an unspecified or receive-based stamp never claims to be a known acquisition time', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer, send } = mount(OdomRenderer, 'legacy-state-key');
  try {
    const sample = odom(); delete sample.vendor.motus_slam.stamp_source;
    send(sample);
    assert.match(box.querySelector('.odom-summary').textContent, /采样时间未知/);
    sample.vendor.motus_slam.stamp_source = 'receive'; send(sample);
    assert.match(box.querySelector('.odom-summary').textContent, /采样时间未知/);
    sample.vendor.motus_slam.stamp_source = 'measurement'; send(sample);
    assert.match(box.querySelector('.odom-summary').textContent, /采样时间1234 ms/);
  } finally { renderer.unmount(); }
});

test('odometry rejects unrelated JSON and clears absent current pose without losing raw expansion', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer, send } = mount(OdomRenderer, { format: 'state/odom' });
  try {
    send(odom()); box.querySelector('.odom-raw').open = true;
    send({ schema: 'motus.odom/1', pose: null, twist: [0, null, null, null, null, 0],
      vendor: { motus_slam: { status: 'lost' } } });
    assert.doesNotMatch(box.querySelector('.odom-summary').textContent, /1, 2, 3/);
    assert.match(box.querySelector('.odom-summary').textContent, /位置—/);
    assert.equal(box.querySelector('.odom-raw').open, true);
    const current = box.querySelector('.odom-summary').textContent;
    for (const unrelated of [null, [1, 2], { pose: 'unrelated' }, { type: 'meta' }]) send(unrelated);
    assert.equal(box.querySelector('.odom-summary').textContent, current);
  } finally { renderer.unmount(); }
});

test('dashboard buffers odd-length odometry and preserves original receipt age on replay', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const source = readFileSync(new URL('./dashboard.js', import.meta.url), 'utf8');
  const start = source.indexOf('class SkillBuffer {');
  const end = source.indexOf('\n}\n', start) + 2;
  const sockets = []; let now = 1000;
  const SkillBuffer = runInNewContext(`const PREVIEW_CAPACITY = 100; (${source.slice(start, end)})`, {
    ArrayBuffer, TextEncoder, Date: { now: () => now }, console: { log() {} },
    WebSocket: class { constructor() { sockets.push(this); } close() {} },
  });
  const buffer = new SkillBuffer('ws://status', 'state/odom');
  const text = JSON.stringify(odom());
  const raw = new TextEncoder().encode(text + (text.length % 2 === 0 ? ' ' : '')).buffer;
  assert.equal(raw.byteLength % 2, 1);
  sockets[0].onmessage({ data: raw });
  const replay = [];
  now = 8000;
  buffer.attach((frame, hint, historical, receivedAt) => replay.push({ frame, receivedAt }));
  assert.equal(replay.length, 1);
  assert.equal(replay[0].receivedAt, 1000);
  const savedNow = Date.now; Date.now = () => now;
  const { box, renderer } = mount(OdomRenderer, { format: 'state/odom' });
  try {
    renderer.onDataSilent(replay[0].frame, 'state/odom', { receivedAt: replay[0].receivedAt });
    assert.match(box.querySelector('.odom-freshness').textContent, /7.0 s · 陈旧/);
  } finally { renderer.unmount(); Date.now = savedNow; buffer.destroy(); }
});

test('yaw requires explicit z-up xyzw axes, a valid quaternion and tracking', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const { box, renderer, send } = mount(OdomRenderer, { format: 'state/odom' });
  try {
    const sample = odom(); send(sample);
    const summary = () => box.querySelector('.odom-summary').textContent;
    assert.match(summary(), /yaw未知/);
    sample.vendor.motus_slam.pose_axes = { quaternion_order: 'xyzw', up_axis: 'z' };
    send(sample); assert.match(summary(), /yaw0 rad/);
    sample.pose.quaternion = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
    send(sample); assert.match(summary(), /yaw1.571 rad/);
    for (const axes of [{ quaternion_order: 'wxyz', up_axis: 'z' },
      { quaternion_order: 'xyzw', up_axis: 'x' }, { quaternion_order: 'xyzw', up_axis: 'y' },
      { up_axis: 'z' }]) {
      sample.vendor.motus_slam.pose_axes = axes; send(sample); assert.match(summary(), /yaw未知/);
    }
    sample.vendor.motus_slam.pose_axes = { quaternion_order: 'xyzw', up_axis: 'z' };
    for (const quaternion of [[0, 0, 0, 0], [0, 0, null, 1], [0, 0, 1], ['0', 0, 0, 1]]) {
      sample.pose.quaternion = quaternion; send(sample); assert.match(summary(), /yaw未知/);
    }
    sample.pose.quaternion = [0, 0, 0, 1]; sample.vendor.motus_slam.status = 'lost';
    send(sample); assert.match(summary(), /yaw未知/);
  } finally { renderer.unmount(); }
  const mounted = mount(OdomRenderer, { format: 'state/odom' });
  try {
    const sample = odom(); sample.pose.quaternion = [0, 0, 2 * Math.SQRT1_2, 2 * Math.SQRT1_2];
    sample.vendor.motus_slam.pose_axes = { quaternion_order: 'xyzw', up_axis: 'z' };
    mounted.send(sample);
    assert.match(mounted.box.querySelector('.odom-summary').textContent, /yaw1.571 rad/);
  } finally { mounted.renderer.unmount(); }
});

test('copying raw data works on HTTP pages without the secure Clipboard API', async () => {
  const { OdomRenderer } = await import('./renderers/odom.js');
  const savedNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {} });
  const { box, renderer } = mount(OdomRenderer, { format: 'state/odom' });
  let copied;
  document.execCommand = command => {
    if (command !== 'copy') return false;
    copied = document.activeElement.value; return true;
  };
  try {
    const raw = JSON.stringify(odom(), null, 2) + '\n';
    renderer.onData(new TextEncoder().encode(raw).buffer);
    await box.querySelector('.odom-copy').onclick();
    assert.equal(copied, raw);
    assert.equal(box.querySelector('textarea'), null);
    assert.equal(box.querySelector('.odom-copy').textContent, '已复制');
  } finally {
    renderer.unmount();
    if (savedNavigator) Object.defineProperty(globalThis, 'navigator', savedNavigator);
    else delete globalThis.navigator;
  }
});
