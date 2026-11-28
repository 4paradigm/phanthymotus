import { test } from 'node:test';
import assert from 'node:assert/strict';
import { refreshRosbagCard } from './rosbag-manager.js';

test('canvas displays recorder failures and completed recordings', async () => {
  const original = globalThis.fetch;
  const button = {};
  try {
    for (const [state, label] of [['loading', '录制启动中'], ['recording', '录制中'],
      ['error', '录制异常'], ['completed', '录制已结束']]) {
      globalThis.fetch = async () => ({ok: true, json: async () => ({code: 200,
        data: [{type: 'text', text: JSON.stringify({state, message: 'detail'})}]})});
      await refreshRosbagCard({mcpId: 'p', toolName: 'rosbag_recorder'}, button);
      assert.match(button.textContent, new RegExp(label));
      assert.equal(button.title, 'detail');
    }
  } finally { globalThis.fetch = original; }
});

test('a late status does not overwrite the card after the project stops', async () => {
  const original = globalThis.fetch;
  const button = {textContent: '查看录制与文件'};
  try {
    globalThis.fetch = async () => ({ok: true, json: async () => ({code: 200, data: {state: 'recording'}})});
    await refreshRosbagCard({mcpId: 'p', toolName: 'rosbag_recorder'}, button, () => false);
    assert.equal(button.textContent, '查看录制与文件');
  } finally { globalThis.fetch = original; }
});
