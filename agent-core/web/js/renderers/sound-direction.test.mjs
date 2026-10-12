import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatSoundDirection } from './sound-direction.js';

test('valid direction displays signed angle and capture window', () => {
  assert.equal(formatSoundDirection({ state: 1, angle_deg: -90,
    window_start_us: 1700000000000000, window_end_us: 1700000000020000 }),
    '-90° · 1700000000000000–1700000000020000 µs');
});

test('no event clears angle', () => {
  assert.equal(formatSoundDirection({ state: 0, angle_deg: 90 }), '暂无声源');
});
