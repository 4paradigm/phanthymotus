import test from 'node:test';
import assert from 'node:assert/strict';
import { connectedFileSources, filePortHtml } from './file-download-card.js';

const spec = { protocol: 'motus.files/1', action: 'list_assets' };
const schema = { 'x-file-download': spec };

const cards = [
  { id: 'capture', mcpId: 'driver', toolName: 'arbitrary_export', driverName: 'Exporter' },
  { id: 'other', mcpId: 'driver', toolName: 'vision_capture' },
  { id: 'download', mcpId: 'agentcore', toolName: 'file_download' },
];
const mcps = [{ id: 'driver', tools: [{ name: 'arbitrary_export', inputSchema: schema },
  { name: 'vision_capture', inputSchema: {} }] }];
const resolved = card => ({ ...card, fileDownload: spec, multiInstance: false });
const link = { fromCardId: 'capture', toCardId: 'download', fromPortIdx: 'files',
  toPortIdx: 'files', format: 'file/reference', fromTopic: '' };

test('独立下载卡片仅查询当前连线上的文件来源', () => {
  assert.deepEqual(connectedFileSources('download', cards, [link], mcps), [resolved(cards[0])]);
  assert.deepEqual(connectedFileSources('another-download', cards, [link], mcps), []);
  assert.deepEqual(connectedFileSources('download', cards, [], mcps), []);
  assert.deepEqual(connectedFileSources('download', cards.slice(1), [link], mcps), []);
});

test('普通数据连线、错误端口和不支持的来源不会成为文件来源', () => {
  for (const change of [
    { format: 'data/json' }, { fromPortIdx: '0' }, { toPortIdx: '0' }, { fromCardId: 'other' },
  ]) assert.deepEqual(connectedFileSources('download', cards, [{ ...link, ...change }], mcps), []);
});

test('多来源按连线顺序保留，重复连线不会重复读取同一来源', () => {
  const second = { ...cards[0], id: 'capture2' };
  assert.deepEqual(connectedFileSources('download', [...cards, second], [
    link, { ...link, fromCardId: 'capture2' }, link,
  ], mcps), [resolved(cards[0]), resolved(second)]);
});

test('输出口由声明决定，名为 vision_capture 但未声明也不显示', () => {
  assert.match(filePortHtml(cards[0], 'out', schema), /file\/reference/);
  assert.equal(filePortHtml(cards[1], 'out', {}), '');
  assert.equal(filePortHtml(cards[0], 'out', { 'x-file-download': { protocol: 'unsupported', action: 'list_assets' } }), '');
});
