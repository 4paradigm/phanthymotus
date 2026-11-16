import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

function renderer() {
  // Exercise the real packet handler; replace only browser WebGL imports.
  const source = fs.readFileSync(new URL('./renderers/mapping.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '')
    .replace('export const MappingRenderer', 'globalThis.MappingRenderer');
  class Vector3 { set(...values) { this.values = values; } }
  const context = vm.createContext({ THREE: { Vector3 }, ArrayBuffer, DataView, Float32Array });
  vm.runInContext(source, context);
  const r = context.MappingRenderer;
  r._positions = new Float32Array(3); r._colors = new Float32Array(3);
  r._points = { geometry: { attributes: { position: {}, color: {} }, setDrawRange() {} } };
  r._robotMesh = { visible: true, position: new Vector3(), rotation: new Vector3() };
  r._markerLabel = { textContent: '' };
  return r;
}
function packet(flags) {
  const buffer = new ArrayBuffer(29), view = new DataView(buffer);
  view.setFloat32(8, -.7, true); view.setUint8(12, flags); view.setUint32(13, 1, true);
  view.setFloat32(17, 1, true); view.setFloat32(21, 2, true); view.setFloat32(25, 3, true);
  return buffer;
}

test('mapped camera heading renders without a text overlay', () => {
  const r = renderer(); r.onData(packet(15));
  assert.equal(r._markerLabel.textContent, '');
  assert.equal(r._robotMesh.visible, true);
  assert.ok(Math.abs(r._robotMesh.rotation.values[1] - .7) < 1e-6);
  assert.deepEqual([...r._positions], [1, 3, -2]);
});
test('body, unavailable, and legacy markers preserve visibility without text', () => {
  const r = renderer();
  r.onData(packet(23)); assert.equal(r._robotMesh.visible, true);
  r.onData(packet(39)); assert.equal(r._robotMesh.visible, false);
  assert.equal(r._markerLabel.textContent, '');
  r.onData(packet(3)); assert.equal(r._robotMesh.visible, true);
  r.onData(packet(19)); assert.equal(r._markerLabel.textContent, '');
});
