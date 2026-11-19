import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

// Exercise the data handlers without Three/WebGL or a browser.
async function renderer(name, exported) {
  const source = readFileSync(new URL(`./renderers/${name}.js`, import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '');
  const module = await import(`data:text/javascript,${encodeURIComponent(source)}`);
  return { ...module[exported] };
}

test('a prismatic joint translates in metres without rotating the child', async () => {
  const skeleton = await renderer('skeleton', 'SkeletonRenderer');
  const axis = { x: 0, y: 0, z: 1 };
  const position = {
    copy(v) { Object.assign(this, v); return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; },
  };
  const rotations = [];
  skeleton._joints = {
    body_joint: { userData: { axis, jointType: 'prismatic' }, position,
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
    arm: { userData: { axis, jointType: 'revolute' },
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
  };
  skeleton._loaded = true;
  skeleton.onData(new TextEncoder().encode(JSON.stringify({
    joints: [{ name: 'body_joint', q: 0.6 }, { name: 'arm', q: 0.25 }],
  })));
  assert.deepEqual([position.x, position.y, position.z], [0, 0, 0.6]);
  assert.deepEqual(rotations, [[axis, 0.25]]);
});

test('a zero-point snapshot clears the displayed cloud', async () => {
  const cloud = await renderer('pointcloud', 'PointCloudRenderer');
  cloud._frames = [{ count: 4 }];
  const packet = new ArrayBuffer(8);
  new DataView(packet).setUint32(0, 16, true);
  cloud.onData(packet);
  assert.deepEqual(cloud._frames, []);
  assert.equal(cloud._displayDirty, true);
});
