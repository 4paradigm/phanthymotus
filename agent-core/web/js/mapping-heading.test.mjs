import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

function renderer() {
  // Exercise the real packet handler; replace only browser WebGL imports.
  const source = fs.readFileSync(new URL('./renderers/mapping.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '')
    .replace('export const MappingRenderer', 'globalThis.MappingRenderer');
  class Vector3 { constructor(...values) { this.values = values; } set(...values) { this.values = values; } }
  const context = vm.createContext({ THREE: { Vector3 }, ArrayBuffer, DataView, Float32Array });
  vm.runInContext(source, context);
  const r = context.MappingRenderer;
  r._positions = new Float32Array(3); r._colors = new Float32Array(3);
  const attribute = () => ({
    ranges: [],
    clearUpdateRanges() { this.ranges = []; },
    addUpdateRange(start, count) { this.ranges.push([start, count]); },
  });
  r._points = { geometry: { attributes: { position: attribute(), color: attribute() }, setDrawRange() {} } };
  r._robotMesh = { visible: true, position: new Vector3(), rotation: new Vector3() };
  return r;
}
function packet(flags) {
  const buffer = new ArrayBuffer(29), view = new DataView(buffer);
  view.setFloat32(8, -.7, true); view.setUint8(12, flags); view.setUint32(13, 1, true);
  view.setFloat32(17, 1, true); view.setFloat32(21, 2, true); view.setFloat32(25, 3, true);
  return buffer;
}

test('mapped camera heading rotates the mesh and updates point geometry', () => {
  const r = renderer(); r.onData(packet(15));
  assert.equal(r._robotMesh.visible, true);
  assert.ok(Math.abs(r._robotMesh.rotation.values[1] - .7) < 1e-6);
  assert.deepEqual([...r._positions], [1, 3, -2]);
  for (const attribute of Object.values(r._points.geometry.attributes)) {
    assert.deepEqual(attribute.ranges, [[0, 3]]);
    assert.equal(attribute.needsUpdate, true);
  }
});
test('heading flags control marker visibility', () => {
  const r = renderer();
  r.onData(packet(23)); assert.equal(r._robotMesh.visible, true);
  r.onData(packet(39)); assert.equal(r._robotMesh.visible, false);
  r.onData(packet(3)); assert.equal(r._robotMesh.visible, true);
  r.onData(packet(19)); assert.equal(r._robotMesh.visible, true);
});

test('depth display packet and mapping packet share forward-left-up axes', () => {
  const source = fs.readFileSync(new URL('./renderers/pointcloud.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '')
    .replace('export const PointCloudRenderer', 'globalThis.PointCloudRenderer');
  const context = vm.createContext({ ArrayBuffer, DataView, Float32Array });
  vm.runInContext(source, context);
  const cloud = context.PointCloudRenderer;
  cloud._buildAxisMap(null);
  cloud._frames = [];
  // Optical (right=2, down=-1, forward=3), display packet (-right, forward, down).
  const raw = new ArrayBuffer(20), view = new DataView(raw);
  view.setUint32(0, 12, true); view.setUint32(4, 1, true);
  [-2, 3, -1].forEach((value, i) => view.setFloat32(8 + i * 4, value, true));
  cloud.onData(raw);
  const map = renderer(), mapped = packet(3), mappedView = new DataView(mapped);
  [3, -2, 1].forEach((value, i) => mappedView.setFloat32(17 + i * 4, value, true));
  map.onData(mapped);
  assert.deepEqual([...cloud._frames[0].pos], [...map._positions]);
});
