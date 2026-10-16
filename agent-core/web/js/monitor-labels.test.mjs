import test from 'node:test';
import assert from 'node:assert/strict';
import {monitorTopicLabel} from './monitor-labels.js';

test('two rgb streams retain their publishing device names regardless of order', () => {
  for (const [name, topic] of [['机械臂 2', '/piper2/piper/rgb'], ['机械臂 1', '/agilex/piper/rgb']]) {
    const mcp = {name, tools: [{name: 'piper_photo', topic_out: [{topic}]}]};
    assert.equal(monitorTopicLabel(topic, mcp), `${name}｜相机画面`);
    assert.equal(monitorTopicLabel('/other/rgb', mcp), 'rgb');
  }
});

test('unrelated topics and missing device metadata keep their existing names', () => {
  assert.equal(monitorTopicLabel('/lidar/scan'), 'scan');
  assert.equal(monitorTopicLabel('/robot/rgb', {name: 'Camera', tools: ['camera']}), 'rgb');
  assert.equal(monitorTopicLabel('/'), '/');
});
