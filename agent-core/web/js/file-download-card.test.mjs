import test from 'node:test';
import assert from 'node:assert/strict';
import { connectedFileSources } from './file-download-card.js';

const cards = [
  { id: 'capture', mcpId: 'bumi', toolName: 'vision_capture', driverName: 'Bumi' },
  { id: 'other', mcpId: 'bumi', toolName: 'camera' },
  { id: 'download', mcpId: 'agentcore', toolName: 'file_download' },
];
const link = { fromCardId: 'capture', toCardId: 'download', fromPortIdx: 'files',
  toPortIdx: 'files', format: 'file/reference', fromTopic: '' };

test('独立下载卡片仅查询当前连线上的文件来源', () => {
  assert.deepEqual(connectedFileSources('download', cards, [link]), [cards[0]]);
  assert.deepEqual(connectedFileSources('another-download', cards, [link]), []);
  assert.deepEqual(connectedFileSources('download', cards, []), []);
  assert.deepEqual(connectedFileSources('download', cards.slice(1), [link]), []);
});

test('普通数据连线、错误端口和不支持的来源不会成为文件来源', () => {
  for (const change of [
    { format: 'data/json' }, { fromPortIdx: '0' }, { toPortIdx: '0' }, { fromCardId: 'other' },
  ]) assert.deepEqual(connectedFileSources('download', cards, [{ ...link, ...change }]), []);
});

test('多来源按连线顺序保留，重复连线不会重复读取同一来源', () => {
  const second = { ...cards[0], id: 'capture2', mcpId: 'bumi2' };
  assert.deepEqual(connectedFileSources('download', [...cards, second], [
    link, { ...link, fromCardId: 'capture2' }, link,
  ]), [cards[0], second]);
});
