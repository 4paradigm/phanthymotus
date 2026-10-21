import test from 'node:test';
import assert from 'node:assert/strict';
import { listCaptureFiles, downloadCaptureFile } from './vision-capture-download.js';

const directory = '/opt/phanthy-motus/data/vision_capture';
const info = { ok: false, photos_dir: `${directory}/photos`, videos_dir: `${directory}/videos`,
  active_recording: null };
const mcpResult = data => ({ code: 200, data: [{ type: 'text', text: JSON.stringify(data) }] });
const response = data => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });

function server(t, entries, getInfo = () => info) {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    const body = JSON.parse(options.body);
    if (url === '/api/mcp/bumi%20driver/call?timeout_s=20') {
      assert.deepEqual(body, { tool: 'vision_capture', arguments: { action: 'info' } });
      return response(mcpResult(getInfo()));
    }
    assert.equal(url, '/api/file/list');
    assert.ok([info.photos_dir, info.videos_dir].includes(body.path));
    return response({ code: 200, data: { files: entries() } });
  });
}

test('历史照片来自磁盘，刷新后不再列出已删除的文件，相机离线也能下载', async t => {
  let entries = [
    { name: 'IMG_20260101.jpg', type: 'file', size: 123 },
    { name: 'IMG_20260102.jpg', type: 'file', size: 456 },
    { name: 'subdirectory.jpg', type: 'dir', size: 0 },
    { name: 'empty.jpg', type: 'file', size: 0 },
    { name: 'notes.txt', type: 'file', size: 10 },
    { name: '../outside.jpg', type: 'file', size: 100 },
  ];
  server(t, () => entries);
  assert.deepEqual(await listCaptureFiles('bumi driver', 'photo'), [
    { name: 'IMG_20260102.jpg', path: `${directory}/photos/IMG_20260102.jpg`, size: 456 },
    { name: 'IMG_20260101.jpg', path: `${directory}/photos/IMG_20260101.jpg`, size: 123 },
  ]);
  entries = entries.filter(file => file.name !== 'IMG_20260101.jpg');
  assert.deepEqual((await listCaptureFiles('bumi driver', 'photo')).map(file => file.name), ['IMG_20260102.jpg']);
});

test('空目录显示空列表', async t => {
  server(t, () => []);
  assert.deepEqual(await listCaptureFiles('bumi driver', 'photo'), []);
});

test('目录使用 driver 的 info 返回值，不写死默认保存位置', async t => {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (url.startsWith('/api/mcp/')) return response(mcpResult({ photos_dir: '/shared/custom/photos/' }));
    assert.deepEqual(JSON.parse(options.body), { path: '/shared/custom/photos' });
    return response({ code: 200, data: { files: [{ name: 'photo.jpg', type: 'file', size: 100 }] } });
  });
  assert.equal((await listCaptureFiles('bumi driver', 'photo'))[0].path, '/shared/custom/photos/photo.jpg');
});

test('不列出正在录制的视频，包括读取目录期间刚开始的录像', async t => {
  let inspected = 0;
  server(t, () => [
    { name: 'video_old.mp4', type: 'file', size: 500 },
    { name: 'video_active.mp4', type: 'file', size: 100 },
  ], () => ++inspected === 1 ? info : {
    ...info, active_recording: { state: 'recording', path: `${directory}/videos/video_active.mp4` },
  });
  assert.deepEqual(await listCaptureFiles('bumi driver', 'video'), [
    { name: 'video_old.mp4', path: `${directory}/videos/video_old.mp4`, size: 500 },
  ]);
});

test('info 失败时不伪装成空目录，也不继续读取文件', async t => {
  t.mock.method(globalThis, 'fetch', async url => {
    assert.ok(url.startsWith('/api/mcp/'));
    return response({ code: 500, message: 'driver unavailable' });
  });
  await assert.rejects(listCaptureFiles('bumi driver', 'photo'), /driver unavailable/);
});

test('未共享或尚未创建的目录给出读取错误', async t => {
  t.mock.method(globalThis, 'fetch', async url => url.startsWith('/api/mcp/')
    ? response(mcpResult(info)) : new Response('Internal Server Error', { status: 500 }));
  await assert.rejects(listCaptureFiles('bumi driver', 'photo'), /目录/);
});

test('无法确认当前录像状态时不提供可能未完成的视频', async t => {
  let inspected = 0;
  t.mock.method(globalThis, 'fetch', async url => {
    if (url === '/api/file/list') return response({ code: 200, data: { files: [] } });
    return ++inspected === 1 ? response(mcpResult(info)) : new Response('', { status: 503 });
  });
  await assert.rejects(listCaptureFiles('bumi driver', 'video'));
});

test('下载经过现有 fetch 鉴权通道，保留文件名并释放临时 URL', async t => {
  const path = `${directory}/photos/照片 & 1.jpg`;
  let clicked = false, removed = false, revoked = false;
  const anchor = { click() { clicked = true; }, remove() { removed = true; } };
  t.mock.method(globalThis, 'fetch', async url => {
    assert.equal(url, '/api/file/download?path=' + encodeURIComponent(path));
    return new Response(new Uint8Array([255, 216, 255]), { headers: { 'Content-Type': 'image/jpeg' } });
  });
  t.mock.method(URL, 'createObjectURL', blob => { assert.equal(blob.size, 3); return 'blob:photo'; });
  t.mock.method(URL, 'revokeObjectURL', url => { assert.equal(url, 'blob:photo'); revoked = true; });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const originalDocument = globalThis.document;
  globalThis.document = {
    createElement(tag) { assert.equal(tag, 'a'); return anchor; },
    body: { append(element) { assert.equal(element, anchor); } },
  };
  t.after(() => { globalThis.document = originalDocument; });
  await downloadCaptureFile(path);
  assert.equal(anchor.href, 'blob:photo');
  assert.equal(anchor.download, '照片 & 1.jpg');
  assert.ok(clicked && removed);
  t.mock.timers.tick(60000);
  assert.ok(revoked);
});

test('已删除文件或鉴权失败不会被当成媒体下载', async t => {
  t.mock.method(URL, 'createObjectURL', () => assert.fail('must not download an error response'));
  for (const status of [401, 404, 500]) {
    t.mock.method(globalThis, 'fetch', async () => new Response('error', { status }));
    await assert.rejects(downloadCaptureFile('/shared/photos/gone.jpg'), new RegExp(String(status)));
  }
});
