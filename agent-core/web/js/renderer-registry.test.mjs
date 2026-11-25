/**
 * The one renderer registry (renderers/index.js).
 *
 * Three views used to keep their own RENDERERS arrays and they had drifted —
 * the one that gets forgotten is the one nobody had open while testing (the
 * comment on test_control_renderer_registered.py records the same incident for
 * ControlRenderer). This test scans the renderers directory itself, so a new
 * renderer file that nobody registers is a failure here, not a blank card in
 * whichever view lost the coin toss.
 *
 * It also pins the starvation fix: every renderer declares `channels`, and
 * pickRenderer(hint, channel) only ever returns a renderer that consumes the
 * requested channel. image/video/lidar are event-only; a host delivering topic
 * frames used to select them by canRender and starve them silently.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// mapping / pointcloud / skeleton import 裸说明符 `three`（浏览器靠 importmap
// 指到 CDN）。先注册 node 侧的解析钩子，再动态 import 注册表——静态 import
// 会被提升到 register() 之前，钩子来不及生效。
register('./test-support/bare-three-loader.mjs', import.meta.url);

const { RENDERERS, EXPLICIT, pickRenderer, instantiate } =
  await import('./renderers/registry.js');

const DIR = fileURLToPath(new URL('./renderers/', import.meta.url));

async function loadDirectoryRenderers() {
  const found = [];
  for (const entry of readdirSync(DIR)) {
    if (!entry.endsWith('.js') || entry === 'index.js' || entry.includes('.test.')) continue;
    const module = await import(new URL(`./renderers/${entry}`, import.meta.url).href);
    for (const [exportName, value] of Object.entries(module)) {
      if (value && typeof value === 'object' && value.canRender && value.mount && value.name) {
        found.push({ file: entry, exportName, renderer: value });
      }
    }
  }
  return found;
}

test('every renderer file is registered — in RENDERERS or EXPLICIT', async () => {
  const found = await loadDirectoryRenderers();
  assert.ok(found.length >= 16, `directory scan found only ${found.length} renderers`);
  const registered = new Set([...RENDERERS, ...Object.values(EXPLICIT)]);
  for (const { file, exportName, renderer } of found) {
    assert.ok(registered.has(renderer),
      `${file} 的 ${exportName} 没有注册进 renderers/index.js —— 加进 RENDERERS，或带理由放进 EXPLICIT`);
  }
});

test('every registered renderer declares channels it actually implements', () => {
  for (const r of RENDERERS) {
    assert.ok(Array.isArray(r.channels) && r.channels.length > 0,
      `${r.name} 没有声明 channels`);
    for (const ch of r.channels) {
      assert.ok(['data', 'event'].includes(ch), `${r.name} 的通道 ${ch} 不合法`);
      assert.ok(r[ch === 'data' ? 'onData' : 'onEvent'],
        `${r.name} 声明消费 ${ch}，但没有实现 ${ch === 'data' ? 'onData' : 'onEvent'}`);
    }
  }
});

test('registry names are unique', () => {
  const names = RENDERERS.map((r) => r.name);
  assert.equal(new Set(names).size, names.length, `重名: ${names.join(', ')}`);
});

test('control is tried before the catch-alls in the find list', () => {
  // `find` takes the first match; TextRenderer answers to broad hints and
  // ActivityRenderer is the fallback — ControlRenderer behind either of them
  // would never be reached (the original incident this whole file guards).
  const names = RENDERERS.map((r) => r.name);
  const control = names.indexOf('control');
  for (const catchAll of ['text', 'activity']) {
    assert.ok(control < names.indexOf(catchAll),
      `control 必须排在 ${catchAll} 之前，否则 control/* 永远轮不到`);
  }
});

test('pickRenderer with a channel never returns a renderer that cannot consume it', () => {
  const hints = ['image/jpeg', 'video/mjpeg', 'video/depth', 'sensor/battery',
    'sensor/lidar-2d', 'sensor/skeleton', 'sensor/pose2d', 'control/velocity',
    'audio/pcm-16k', 'text/asr', 'data/json', 'image/depth-z16', 'image', 'something/unknown'];
  for (const hint of hints) {
    for (const channel of ['data', 'event']) {
      const r = pickRenderer(hint, channel);
      assert.ok(r.channels.includes(channel),
        `pickRenderer(${hint}, ${channel}) 返回 ${r.name}（channels: ${r.channels}）——又回到饿死老路`);
    }
  }
});

test('the starvation cases are pinned: topic-frame hosts never pick event-only renderers', () => {
  assert.notEqual(pickRenderer('video/depth', 'data').name, 'video');
  assert.notEqual(pickRenderer('sensor/lidar-2d', 'data').name, 'lidar');
  assert.notEqual(pickRenderer('image', 'data').name, 'image');
  // …and the event channel still reaches them where events are actually fed.
  assert.equal(pickRenderer('video/mjpeg', 'event').name, 'video');
  assert.equal(pickRenderer('image', 'event').name, 'image');
});

test('data-channel picks match the formats each view shows today', () => {
  assert.equal(pickRenderer('image/jpeg', 'data').name, 'camera');
  assert.equal(pickRenderer('sensor/battery', 'data').name, 'battery');
  assert.equal(pickRenderer('sensor/skeleton', 'data').name, 'skeleton');
  assert.equal(pickRenderer('sensor/pose2d', 'data').name, 'pose2d');
  assert.equal(pickRenderer('audio/pcm-16k', 'data').name, 'audio');
  assert.equal(pickRenderer('control/velocity', 'data').name, 'control');
  // kv-latest 与 text 的 canRender 相同，谁都不进 find 列表抢对方的活；
  // kv 由 monitor 的模式开关显式选择。
  assert.equal(pickRenderer('data/json', 'data').name, 'text');
  assert.equal(EXPLICIT.kv_latest.name, 'kv-latest');
});

test('unfiltered pick keeps the dashboard both-channels behaviour', () => {
  assert.equal(pickRenderer('image').name, 'image');
  assert.equal(pickRenderer('video/mjpeg').name, 'video');
});

test('instantiate clones without sharing instance state', () => {
  const a = instantiate(pickRenderer('sensor/battery', 'data'));
  const b = instantiate(pickRenderer('sensor/battery', 'data'));
  assert.notEqual(a, b);
  a._parts = { mutated: true };
  assert.equal(b._parts, null, '一个实例的挂载状态泄漏到了另一个实例');
});
