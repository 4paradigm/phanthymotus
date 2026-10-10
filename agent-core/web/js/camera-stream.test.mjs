/**
 * The pure parts of the browser camera streamer.
 *
 * getUserMedia and canvas encoding need a browser, so what is tested here is
 * the arithmetic around them — which is where the mistakes that matter live.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULTS, achievedFps, clampFps, fitCapture } from './camera-stream.js';

test('the default frame rate matches what the pose card expects', () => {
  // 12, because a skeleton-action model classifies a clip and a 2.5 s window at
  // 5 fps is 13 real frames resampled up to the engine's 100 — mostly
  // interpolation. Sending fewer frames than the consumer needs is the kind of
  // mismatch that gets diagnosed as a bad model.
  assert.equal(DEFAULTS.fps, 12);
});

test('the frame rate is clamped to what a browser can sustain', () => {
  // Each frame is JPEG-encoded on the main thread, so an unbounded fps does not
  // produce more frames, it produces a backlog.
  assert.equal(clampFps(60), 30);
  assert.equal(clampFps(0), 1);
  assert.equal(clampFps(-5), 1);
  assert.equal(clampFps(12), 12);
});

test('a missing or unusable frame rate falls back to the default', () => {
  // `null` needs its own check: `Number(null)` is 0, not NaN, so an unset fps
  // would otherwise clamp to 1 and look like a stall.
  for (const value of [undefined, null, '', NaN, 'fast', {}]) {
    assert.equal(clampFps(value), DEFAULTS.fps, String(value));
  }
});

test('capture preserves the camera aspect ratio rather than forcing 4:3', () => {
  // A stretched body breaks the pose card's geometry: its angles are invariant
  // to scale and rotation but **not** to anisotropic scaling, which is exactly
  // what squeezing 16:9 into 4:3 is.
  const wide = fitCapture(1920, 1080, 640);
  assert.equal(wide.width, 640);
  assert.equal(wide.height, 360);
  assert.equal(wide.width / wide.height, 1920 / 1080);

  const tall = fitCapture(480, 640, 640);
  assert.equal(tall.height, 640);
  assert.equal(tall.width, 480);
});

test('capture dimensions are even, which JPEG chroma subsampling wants', () => {
  for (const [w, h] of [[1280, 721], [1023, 767], [641, 481]]) {
    const size = fitCapture(w, h, 640);
    assert.equal(size.width % 2, 0, `${w}x${h}`);
    assert.equal(size.height % 2, 0, `${w}x${h}`);
  }
});

test('a camera that reports no dimensions yet gets the defaults', () => {
  // `videoWidth` is 0 until the first frame decodes; sizing a canvas to 0 makes
  // every subsequent frame empty, with nothing logged anywhere.
  assert.deepEqual(fitCapture(0, 0), { width: DEFAULTS.width, height: DEFAULTS.height });
  assert.deepEqual(fitCapture(undefined, undefined),
                   { width: DEFAULTS.width, height: DEFAULTS.height });
});

test('the long edge is what gets scaled, whichever it is', () => {
  assert.equal(Math.max(...Object.values(fitCapture(1920, 1080, 320))), 320);
  assert.equal(Math.max(...Object.values(fitCapture(1080, 1920, 320))), 320);
});


test('the achieved rate is zero while nothing is streaming', () => {
  // It exists because a configured 12 fps and an achieved 1 fps are
  // indistinguishable from a frame counter alone, and the difference decides
  // whether anything temporal can work: at 1 fps a tracked person expires
  // between frames and the action model never accumulates a window.
  assert.equal(achievedFps(), 0);
});
