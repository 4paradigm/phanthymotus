import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

// Load the renderer without WebGL and substitute the scene objects used by the tests.
function renderer(three = {}) {
  const source = readFileSync(new URL('./renderers/skeleton.js', import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '')
    .replace('export const SkeletonRenderer', 'const SkeletonRenderer');
  return new Function('THREE', `${source}\nreturn { ...SkeletonRenderer };`)(three);
}

test('successive lift readings replace the upper-body offset without moving its base', () => {
  const skeleton = renderer();
  const axis = { x: 0, y: 0, z: 1 };
  const position = {
    copy(v) { Object.assign(this, v); return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; },
  };
  const rotations = [];
  const base = { position: { x: 0, y: 0, z: 0 } };
  skeleton._joints = {
    body_joint: { userData: { axis, jointType: 'prismatic' }, position,
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
    arm: { userData: { axis, jointType: 'revolute' },
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
  };
  skeleton._links = { base_link: base };
  skeleton._loaded = true;
  const send = joints => skeleton.onData(new TextEncoder().encode(JSON.stringify({ joints })));
  send([{ name: 'body_joint', q: 0.62 }, { name: 'arm', q: 0.25 }]);
  assert.deepEqual([position.x, position.y, position.z], [0, 0, 0.62]);
  send([{ name: 'body_joint', q: 0.4 }]);
  send([{ name: 'body_joint', q: null }]);
  assert.deepEqual([position.x, position.y, position.z], [0, 0, 0.4]);
  assert.deepEqual(base.position, { x: 0, y: 0, z: 0 });
  assert.deepEqual(rotations, [[axis, 0.25]]);
});

test('base primitives render the box edges and a vertical rail at their URDF origins', () => {
  class Geometry {
    constructor(...args) { this.args = args; }
    dispose() { this.disposed = true; }
    rotateX(angle) { this.angle = angle; }
  }
  class Object3D {
    constructor(geometry) {
      this.geometry = geometry;
      this.position = { set: (...xyz) => { this.xyz = xyz; } };
      this.rotation = { set: () => {} };
    }
  }
  const skeleton = renderer({
    BoxGeometry: Geometry, EdgesGeometry: Geometry, CylinderGeometry: Geometry,
    Mesh: Object3D, LineSegments: Object3D, MeshPhongMaterial: class {}, LineBasicMaterial: class {},
  });
  const element = (attrs, elements = {}) => ({
    getAttribute: key => attrs[key] ?? null,
    querySelector: key => elements[key] ?? null,
  });
  const drawn = [];
  const link = { add: object => drawn.push(object) };
  skeleton._addPrimitiveVisual(link, element({}, {
    origin: element({ xyz: '0 0 0.1' }), 'geometry > box': element({ size: '0.65 0.51 0.2' }),
  }));
  skeleton._addPrimitiveVisual(link, element({}, {
    origin: element({ xyz: '0.163 0 0.803' }), 'geometry > cylinder': element({ radius: '0.008', length: '1.206' }),
  }));
  assert.equal(drawn.length, 2);
  assert.deepEqual(drawn[0].geometry.args[0].args, [0.65, 0.51, 0.2]);
  assert.deepEqual(drawn[0].xyz, [0, 0, 0.1]);
  assert.equal(drawn[1].geometry.angle, Math.PI / 2);
  assert.deepEqual(drawn[1].xyz, [0.163, 0, 0.803]);
});
