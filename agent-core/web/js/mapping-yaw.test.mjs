/** Regression guard: the ROS→Three.js axis remap and arrow yaw must agree.
 * Run: node --test agent-core/web/js/mapping-yaw.test.mjs
 *
 * The browser renderer uses Three.js, which is not installed by the host-side
 * Node test runner. Assert the production mapping expressions and independently
 * check their implied 3D geometry without a WebGL/DOM dependency.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const renderer = readFileSync(new URL('./renderers/mapping.js', import.meta.url), 'utf8');

test('map points, robot position and arrow use one ROS-to-Three convention', () => {
  assert.match(renderer, /pos\[idx\] = x;/);
  assert.match(renderer, /pos\[idx \+ 1\] = z;/);
  assert.match(renderer, /pos\[idx \+ 2\] = -y;/);
  assert.match(renderer, /_robotMesh\.position\.set\(robotX, 0\.2, -robotY\)/);
  assert.match(renderer, /_robotMesh\.rotation\.set\(0, robotYaw, 0\)/);
});

test('positive and negative ROS yaw both match arrow rotation in Three.js', () => {
  for (const yaw of [0, Math.PI / 2, -Math.PI / 2, Math.PI]) {
    const rosForward = [Math.cos(yaw), Math.sin(yaw), 0];
    const transformed = [rosForward[0], rosForward[2], -rosForward[1]];
    // A +Y axis rotation of Three.js local +X is (cos(yaw), 0, -sin(yaw)).
    const arrowForward = [Math.cos(yaw), 0, -Math.sin(yaw)];
    for (let axis = 0; axis < 3; axis++) {
      assert.ok(Math.abs(transformed[axis] - arrowForward[axis]) < 1e-12);
    }
  }
});
