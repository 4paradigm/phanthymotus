import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CameraRenderer } from './renderers/camera.js';
import { KvLatestRenderer } from './renderers/kv-latest.js';

test('compressed PNG and JPEG streams select the camera renderer', () => {
  assert.equal(CameraRenderer.canRender('image/png'), true);
  assert.equal(CameraRenderer.canRender('image/jpeg'), true);
  assert.equal(CameraRenderer.canRender('image/depth-zlib'), false);
});

test('the latest panel removes children of replaced parents and retains unrelated fields', () => {
  const removed = [];
  const renderer = {
    ...KvLatestRenderer,
    _el: {},
    _cells: {},
    _updateCell(key, value) {
      this._cells[key] = { value, el: { remove: () => removed.push(key) } };
    },
  };
  const receive = payload => renderer.onData(new TextEncoder().encode(JSON.stringify(payload)));
  receive({ measurement: { min: 2, max: 5 }, status: 'ready' });
  receive({ measurement: { min: 3 } });
  assert.deepEqual(removed, ['measurement.max']);
  assert.equal(renderer._cells['measurement.min'].value, 3);
  receive({ measurement: null });
  assert.deepEqual(removed, ['measurement.max', 'measurement.min']);
  assert.equal(renderer._cells.measurement.value, null);
  assert.equal(renderer._cells.status.value, 'ready');
});
