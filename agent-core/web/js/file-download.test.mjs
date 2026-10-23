import test from 'node:test';
import assert from 'node:assert/strict';
import { fileDownloadSpec, listDownloadFiles, downloadFile } from './file-download.js';

const declaration = { protocol: 'motus.files/1', action: 'list_assets' };
const source = { id: 'card-export', mcpId: 'report driver', toolName: 'export_report', fileDownload: declaration };
const response = data => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });

test('只接受有效版本的能力声明，工具名称不决定是否支持下载', () => {
  assert.deepEqual(fileDownloadSpec({ 'x-file-download': declaration }), declaration);
  for (const spec of [undefined, {}, { protocol: 'future/2', action: 'list_assets' },
    { protocol: 'motus.files/1' }, { protocol: 'motus.files/1', action: '' }]) {
    assert.equal(fileDownloadSpec({ 'x-file-download': spec }), null);
  }
});

test('任意工具按声明调用列表动作，不读取 Bumi 目录；刷新后同步已删除文件', async t => {
  let files = [
    { name: 'report.csv', path: '/shared/report.csv', size: 42, mime_type: 'text/csv' },
    { name: 'photo.jpg', path: '/shared/photo.jpg', size: 500, mime_type: 'image/jpeg' },
    { name: 'clip.mp4', path: '/shared/clip.mp4', size: 800, mime_type: 'video/mp4' },
  ];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/mcp/report%20driver/call?timeout_s=20');
    assert.deepEqual(JSON.parse(options.body), { tool: 'export_report', arguments: { action: 'list_assets' } });
    return response({ code: 200, data: [{ type: 'text', text: JSON.stringify({ ok: true, files }) }] });
  });
  assert.deepEqual(await listDownloadFiles(source, 'all'), files);
  assert.deepEqual((await listDownloadFiles(source, 'photo')).map(file => file.name), ['photo.jpg']);
  assert.deepEqual((await listDownloadFiles(source, 'video')).map(file => file.name), ['clip.mp4']);
  files = [];
  assert.deepEqual(await listDownloadFiles(source, 'all'), []);
});

test('多实例列表携带实际来源卡片的 instance_id', async t => {
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.deepEqual(JSON.parse(options.body).arguments, { action: 'list_assets', instance_id: 'card-export' });
    return response({ code: 200, data: { ok: true, files: [] } });
  });
  assert.deepEqual(await listDownloadFiles({ ...source, multiInstance: true }, 'all'), []);
});

test('未知声明不派发请求，错误返回不冒充空列表', async t => {
  t.mock.method(globalThis, 'fetch', () => assert.fail('unsupported provider must not be called'));
  await assert.rejects(listDownloadFiles({ ...source, fileDownload: null }, 'all'));
  for (const data of [
    { code: 500, message: 'driver unavailable' },
    { code: 200, data: { ok: false, message: 'read failed' } },
    { code: 200, data: { ok: true } },
  ]) {
    t.mock.method(globalThis, 'fetch', async () => response(data));
    await assert.rejects(listDownloadFiles(source, 'all'));
  }
});

test('无效路径、空文件及目录项不会形成下载按钮', async t => {
  t.mock.method(globalThis, 'fetch', async () => response({ code: 200, data: { ok: true, files: [
    { name: '../x.jpg', path: '/shared/x.jpg', size: 1, mime_type: 'image/jpeg' },
    { name: 'x.jpg', path: 'relative/x.jpg', size: 1, mime_type: 'image/jpeg' },
    { name: 'empty.jpg', path: '/shared/empty.jpg', size: 0, mime_type: 'image/jpeg' },
    { name: 'folder', path: '/shared/folder', size: 0 },
  ] } }));
  assert.deepEqual(await listDownloadFiles(source, 'all'), []);
});

test('鉴权下载保留文件名并释放临时 URL，错误响应不会被保存', async t => {
  let clicked = false, removed = false, revoked = false;
  const anchor = { click() { clicked = true; }, remove() { removed = true; } };
  t.mock.method(globalThis, 'fetch', async url => {
    assert.equal(url, '/api/file/download?path=%2Fshared%2Freport.csv');
    return new Response('a,b\n1,2');
  });
  t.mock.method(URL, 'createObjectURL', blob => { assert.equal(blob.size, 7); return 'blob:file'; });
  t.mock.method(URL, 'revokeObjectURL', url => { assert.equal(url, 'blob:file'); revoked = true; });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const oldDocument = globalThis.document;
  globalThis.document = { createElement: () => anchor, body: { append() {} } };
  t.after(() => { globalThis.document = oldDocument; });
  await downloadFile('/shared/report.csv');
  assert.equal(anchor.download, 'report.csv');
  assert.ok(clicked && removed);
  t.mock.timers.tick(60000);
  assert.ok(revoked);
  for (const status of [401, 404, 500]) {
    t.mock.method(globalThis, 'fetch', async () => new Response('error', { status }));
    await assert.rejects(downloadFile('/shared/report.csv'), new RegExp(String(status)));
  }
});
