/**
 * 方案 A：`state/odom` 不再有专用渲染器，与 `data/json` 走同一套通用 JSON 渲染。
 *
 * 这个文件原来整套测试的是 renderers/odom.js 的固定字段分组、最新/日志模式和
 * 原始数据展开。那些模块已删除，现在验证的是替代契约：
 *  1. detail-panel / dashboard / monitor-dashboard 选渲染器时把 state/odom 归一为
 *     data/json（订阅与展示的 format 仍是真实值）；
 *  2. motus.odom/1 经 TextRenderer（日志）与 KvLatestRenderer（最新）可读，字段
 *     出现在输出里；非法/非 JSON 帧不崩溃；
 *  3. bus 对 state/odom 发的是二进制帧且长度可能为奇数，预览缓冲不能丢帧；
 *  4. 旧的 odom 专用模块与引用确实清理干净。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { TextRenderer } from './renderers/text.js';
import { KvLatestRenderer } from './renderers/kv-latest.js';

// ── 最小 fake DOM ─────────────────────────────────────────────────────────────
//
// 渲染器直接把 HTML 写进 innerHTML，所以 textContent 也把这段 HTML 算进去：
// 断言只关心键和值是否出现在输出里，不关心标签结构。

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.className = '';
    this.classList = { add() {}, remove() {}, toggle() {} };
    this._text = '';
    this._html = '';
    this.hidden = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
  }
  set textContent(value) { this._text = String(value ?? ''); this._html = ''; this.children = []; }
  get textContent() { return this._text + this._html + this.children.map(child => child.textContent).join(''); }
  set innerHTML(html) { this._html = String(html ?? ''); this._text = ''; this.children = []; }
  get innerHTML() { return this._html; }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) { this.children.splice(index, 1); child.parentElement = null; }
  }
  remove() { this.parentElement?.removeChild(this); }
  addEventListener(name, fn) { this['on' + name] = fn; }
  querySelectorAll(selector) {
    const match = el => selector[0] === '.'
      ? el.className.split(' ').includes(selector.slice(1))
      : el.tagName === selector.toUpperCase();
    return this.children.flatMap(child => [...(match(child) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  get firstChild() { return this.children[0]; }
}

function installDom() {
  const elements = new Map();
  const document = {
    createElement: tag => new Element(tag),
    createTextNode: text => { const el = new Element('text'); el.textContent = text; return el; },
    getElementById: id => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
  };
  globalThis.document = document;
  return { document, elements };
}

// ── 视图加载 ──────────────────────────────────────────────────────────────────
//
// 三个视图都是浏览器 ES module：没有构建步骤，直接在 vm 里去掉 import/export
// 后注入渲染器与 DOM 桩，沿用旧测试验证真实挂载路径的做法。

function _stubRenderer(name, canRender = () => false) {
  return { name, canRender, mount() {}, unmount() {}, onData() {}, onDataSilent() {}, onEvent() {} };
}

function _rendererNames(source) {
  return source.match(/const RENDERERS = \[([^\]]+)\]/)[1].split(',').map(name => name.trim());
}

function _loadView(view, apiExpression) {
  const source = readFileSync(new URL(`./${view}.js`, import.meta.url), 'utf8')
    .replace(/^import .*;\n/gm, '')
    .replace(/^export /gm, '');
  const sockets = [];
  const context = {};
  for (const name of _rendererNames(source)) context[name] = _stubRenderer(name);
  Object.assign(context, {
    TextRenderer, KvLatestRenderer,
    ActivityRenderer: _stubRenderer('activity', () => true),
    document: globalThis.document,
    console: { log() {}, debug() {}, warn() {}, error() {} },
    location: { protocol: 'http:', host: 'localhost' },
    setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
    openDetailPanelMobile() {}, closeDetailPanelMobile() {},
    onMotusEvent() {}, offMotusEvent() {},
    TextEncoder, TextDecoder, ArrayBuffer,
    WebSocket: class {
      constructor(url) { this.url = url; sockets.push(this); }
      close() {}
    },
  });
  return { exposed: runInNewContext(source + `\n(${apiExpression})`, context), sockets };
}

// ── 样例 ──────────────────────────────────────────────────────────────────────

const odom = () => ({
  schema: 'motus.odom/1',
  stamp_ms: 1234,
  frame: 'body',
  pose: { position: [1, 2, 3], quaternion: [0, 0, 0.7071, 0.7071], frame_id: 'odom', child_frame_id: 'camera' },
  twist: [0.5, null, null, null, null, 0.1],
  vendor: { displacement_m: 44.444, turn_angle_deg: 99.999,
    motus_slam: { status: 'tracking', segment_id: 'run-1', reason: 'ok', stamp_source: 'measurement', clock_id: 'clk' } },
});

const encode = value => new TextEncoder().encode(JSON.stringify(value)).buffer;

/** bus 对 state/* 走 send_bytes；JSON 文本的字节数奇偶随机，这里固定成奇数。 */
function oddLengthFrame(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const padded = bytes.byteLength % 2 === 0 ? new TextEncoder().encode(JSON.stringify(value) + ' ') : bytes;
  assert.equal(padded.byteLength % 2, 1);
  return padded.buffer;
}

// ── detail panel ──────────────────────────────────────────────────────────────

test('detail panel renders state/odom through the same JSON text route as data/json', () => {
  for (const format of ['state/odom', 'data/json']) {
    const { elements } = installDom();
    const { exposed, sockets } = _loadView('detail-panel',
      '{ initDetailPanel, showTopicDetail, close: _cleanup, textLike: () => _textLike }');
    try {
      exposed.initDetailPanel();
      exposed.showTopicDetail('/r1/state/odom', format);

      const body = elements.get('detail-body');
      assert.ok(body.querySelector('.renderer-text'), `${format} must mount TextRenderer`);
      assert.equal(body.querySelector('.renderer-odom'), null, 'no dedicated odometry renderer is left');
      // 归一的是渲染器选择，面板上显示的仍是订阅到的真实 format。
      assert.equal(elements.get('detail-subtitle').textContent, `format: ${format}`);
      assert.equal(exposed.textLike(), true, 'stale-canvas notice must not cover JSON logs');

      sockets[0].onmessage({ data: encode(odom()) });
      assert.ok(body.querySelector('.renderer-text').querySelector('.log-entry'),
        `${format} frame must render as a generic JSON log entry`);
      assert.match(body.textContent, /motus\.odom\/1/);
      assert.match(body.textContent, /segment_id/);
      assert.match(body.textContent, /tracking/);
      assert.match(body.textContent, /position/);
    } finally { exposed.close(); }
  }
});

// ── dashboard ─────────────────────────────────────────────────────────────────

test('dashboard keeps the declared output topic and renders state/odom as JSON', () => {
  installDom();
  const { exposed, sockets } = _loadView('dashboard',
    '{ mountRenderer, deliver: _deliverFrame, rendererHint: _rendererHint }');
  exposed.mountRenderer({
    id: 'bumi', name: 'Bumi', server_name: 'Bumi',
    render_hint: 'state/odom',
    ws_path: '/ws/bus/r1/state/odom',
    topic_out: [{ topic: '/r1/state/odom', format: 'state/odom' }, { topic: '/r1/view', format: 'image/jpeg' }],
    topic_in: [],
  });

  const content = globalThis.document.getElementById('renderer-content');
  const rendererWrap = content.children[1];
  const bodyEl = rendererWrap.children[1];
  assert.equal(sockets.length, 1);
  assert.equal(sockets[0].url, 'ws://localhost/ws/bus/r1/state/odom',
    'preview still targets the topic_out the card declares');
  assert.match(rendererWrap.children[0].innerHTML, /\/r1\/state\/odom/);
  assert.ok(bodyEl.querySelector('.renderer-text'), 'state/odom uses the generic JSON renderer');
  assert.equal(bodyEl.querySelector('.renderer-odom'), null);

  // 二进制 + 奇数长度：预览缓冲的 odom 传输特例必须保住，不能丢帧。
  sockets[0].onmessage({ data: oddLengthFrame(odom()) });
  const entry = bodyEl.querySelector('.log-entry');
  assert.ok(entry, 'odd-length binary odometry frame must reach the renderer');
  assert.match(entry.textContent, /motus\.odom\/1/);
  assert.match(entry.textContent, /tracking/);
  assert.match(bodyEl.textContent, /position/);
});

test('dashboard renders a state/odom in_topic with the same JSON renderer', () => {
  installDom();
  const { exposed } = _loadView('dashboard', '{ mountRenderer }');
  exposed.mountRenderer({
    id: 'navi', name: 'Navi',
    render_hint: 'control/twist',
    ws_path: '/ws/bus/r1/cmd',
    topic_out: [{ topic: '/r1/cmd', format: 'control/twist' }],
    topic_in: [{ topic: '/r1/state/odom', format: 'state/odom' }],
  });

  const content = globalThis.document.getElementById('renderer-content');
  const inBody = content.children[1].children[0].children[0];
  assert.ok(inBody.querySelector('.renderer-text'), 'in_topic state/odom uses TextRenderer');
  assert.equal(inBody.querySelector('.renderer-odom'), null);
});

test('dashboard routes replay frames silently and live frames with their format', () => {
  installDom();
  const { exposed } = _loadView('dashboard', '{ deliver: _deliverFrame }');
  const calls = [];
  const renderer = {
    onData: (...args) => calls.push(args),
    onDataSilent: (...args) => calls.push(['silent', ...args]),
  };
  const frame = new ArrayBuffer(2);
  exposed.deliver(renderer, frame, 'state/odom', true);
  exposed.deliver(renderer, frame, 'state/odom', false);
  // odom 与其它 format 没有任何区别：回放静默、实时带 format。
  assert.deepEqual(calls, [['silent', frame], [frame, 'state/odom']]);
});

test('preview buffer keeps history and the odometry binary exception', () => {
  const source = readFileSync(new URL('./dashboard.js', import.meta.url), 'utf8');
  const start = source.indexOf('class SkillBuffer {');
  const end = source.indexOf('\n}\n', start) + 2;
  const load = hint => {
    const sockets = [];
    const SkillBuffer = runInNewContext(`const PREVIEW_CAPACITY = 100; (${source.slice(start, end)})`, {
      ArrayBuffer, TextEncoder, console: { log() {} },
      WebSocket: class { constructor(url) { this.url = url; sockets.push(this); } close() {} },
    });
    return { buffer: new SkillBuffer('ws://preview', hint), socket: () => sockets[0] };
  };

  // JSON 日志：环形历史 100 帧，回放回调仍是 (frame, hint, true)。
  const { buffer, socket } = load('data/json');
  try {
    for (let i = 0; i < 102; i++) socket().onmessage({ data: new Uint16Array([i]).buffer });
    socket().onmessage({ data: '{"type":"ping"}' });
    const replay = [];
    buffer.attach((...args) => replay.push(args));
    assert.equal(replay.length, 100);
    assert.equal(new Uint16Array(replay[0][0])[0], 2);
    assert.equal(new Uint16Array(replay.at(-1)[0])[0], 101);
    assert.ok(replay.every(args => args.length === 3 && args[1] === 'data/json' && args[2] === true));
    socket().onmessage({ data: new Uint16Array([102]).buffer });
    assert.equal(replay.at(-1).length, 2, 'live callbacks are (frame, hint)');
  } finally { buffer.destroy(); }

  // 奇数长度的二进制帧：对普通 format 丢弃（原有 PCM 过滤），对 state/odom 放行。
  for (const [hint, accepted] of [['data/json', false], ['state/odom', true]]) {
    const preview = load(hint);
    try {
      preview.socket().onmessage({ data: new Uint8Array([1]).buffer });
      const replay = [];
      preview.buffer.attach((...args) => replay.push(args));
      assert.equal(replay.length, accepted ? 1 : 0, `${hint} odd-length frame handling`);
    } finally { preview.buffer.destroy(); }
  }
});

// ── 通用 JSON 渲染器 ──────────────────────────────────────────────────────────

test('odometry JSON is readable through both generic JSON renderers', () => {
  const { document } = installDom();

  // 日志模式：每帧一条，键和值都在输出里。
  const logBox = document.createElement('div');
  const logRenderer = Object.assign({}, TextRenderer);
  logRenderer.mount(logBox);
  try {
    assert.doesNotThrow(() => logRenderer.onData(encode(odom())));
    assert.equal(logBox.querySelectorAll('.log-entry').length, 1);
    for (const expected of [/schema/, /motus\.odom\/1/, /stamp_ms/, /status/, /tracking/, /segment_id/, /position/]) {
      assert.match(logBox.textContent, expected, `log view shows ${expected}`);
    }
    // 非 JSON 帧按纯文本展示，不抛异常。
    logRenderer.onData(new TextEncoder().encode('not json at all').buffer);
    assert.match(logBox.textContent, /not json at all/);
    // keepalive ping 不占一行。
    const before = logBox.querySelectorAll('.log-entry').length;
    logRenderer.onData(encode({ type: 'ping' }));
    assert.equal(logBox.querySelectorAll('.log-entry').length, before);
  } finally { logRenderer.unmount(); }

  // 最新模式：顶层字段各自一格，嵌套对象整体成一个可读值。
  const latestBox = document.createElement('div');
  const latestRenderer = Object.assign({}, KvLatestRenderer);
  latestRenderer.mount(latestBox);
  try {
    assert.doesNotThrow(() => latestRenderer.onData(encode(odom())));
    for (const expected of [/pose\.position/, /1, 2, 3/, /vendor\.motus_slam/, /tracking/, /stamp_ms/]) {
      assert.match(latestBox.textContent, expected, `latest view shows ${expected}`);
    }
    // 非法 JSON 被忽略而不是炸掉，已有内容保留。
    assert.doesNotThrow(() => latestRenderer.onData(new TextEncoder().encode('{"schema":').buffer));
    assert.match(latestBox.textContent, /pose\.position/);
  } finally { latestRenderer.unmount(); }
});

// ── monitor dashboard ────────────────────────────────────────────────────────

test('monitor dashboard routes state/odom through the same renderers as data/json', () => {
  installDom();
  const { exposed } = _loadView('monitor-dashboard',
    '{ createRenderer: _createRenderer, rendererHint: _rendererHint, isJsonFormat: _isJsonFormat }');

  assert.equal(exposed.rendererHint('state/odom'), 'data/json');
  assert.equal(exposed.isJsonFormat('state/odom'), true);
  for (const mode of ['log', 'kv']) {
    assert.equal(exposed.createRenderer('state/odom', mode).name,
      exposed.createRenderer('data/json', mode).name,
      `${mode} mode must not special-case odometry`);
  }
  assert.equal(exposed.createRenderer('state/odom', 'log').name, 'text');
  assert.equal(exposed.createRenderer('state/odom', 'kv').name, 'kv-latest');
  assert.equal(exposed.isJsonFormat('image/jpeg'), false);
});

// ── 清理 ──────────────────────────────────────────────────────────────────────

test('the dedicated odometry renderer modules are gone and unreferenced', async () => {
  await assert.rejects(() => import('./renderers/odom.js'), 'renderers/odom.js must be deleted');
  await assert.rejects(() => import('./renderers/odom-presentation.js'), 'odom-presentation.js must be deleted');
  for (const view of ['detail-panel.js', 'dashboard.js', 'monitor-dashboard.js']) {
    const source = readFileSync(new URL(`./${view}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /OdomRenderer|odom-presentation|renderers\/odom/,
      `${view} must not keep a dangling odometry renderer reference`);
  }
});
