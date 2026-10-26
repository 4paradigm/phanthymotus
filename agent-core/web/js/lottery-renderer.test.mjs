import { test } from 'node:test';
import assert from 'node:assert/strict';

import { LotteryRenderer, lotteryFrameIndex, parseLotteryMessage } from './renderers/lottery.js';

test('lottery renderer only claims its own format', () => {
  assert.equal(LotteryRenderer.canRender('data/lottery'), true);
  assert.equal(LotteryRenderer.canRender('data/json'), false);
});

test('JSON snapshots parse from both websocket text and bytes', () => {
  const message = { state: 'result', draw_id: 'draw-1', result: '一等奖' };
  assert.deepEqual(parseLotteryMessage(JSON.stringify(message)), message);
  assert.deepEqual(parseLotteryMessage(new TextEncoder().encode(JSON.stringify(message))), message);
  assert.equal(parseLotteryMessage('not-json'), null);
});

test('animation frames are deterministic but never supplied as the result', () => {
  const a = Array.from({ length: 20 }, (_, tick) => lotteryFrameIndex(12345, tick, 6));
  const b = Array.from({ length: 20 }, (_, tick) => lotteryFrameIndex(12345, tick, 6));
  assert.deepEqual(a, b);
  assert.ok(a.every(index => index >= 0 && index < 6));
  assert.equal(lotteryFrameIndex(12345, 0, 0), -1);
});
