import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { decodeSlamPreview, currentOdomMarker, writeTrajectoryPositions } from './renderers/slam-preview-codec.js';

function packet(meta) {
  const header = new TextEncoder().encode(JSON.stringify(meta));
  const result = new Uint8Array(4 + header.length + 17);
  new DataView(result.buffer).setUint32(0, header.length, true);
  result.set(header, 4);
  result[4 + header.length + 12] = 3;
  return result.buffer;
}

const metadata = {schema: 'motus.slam-preview/1', kind: 'unvalidated_odometry_preview',
  segment_id: 'run-1', frame_id: 'odom', stamp_ns: '1234567890',
  backend: 'raw', accepted_loops: 0};

test('preview decoder preserves metadata with its own binary snapshot', () => {
  const decoded = decodeSlamPreview(packet(metadata));
  assert.equal(decoded.meta.segment_id, 'run-1');
  assert.equal(decoded.meta.frame_id, 'odom');
  assert.equal(decoded.meta.stamp_ns, '1234567890');
  assert.equal(decoded.payload.byteLength, 17);
});

test('invalid metadata and unvalidated loop claims cannot be rendered', () => {
  assert.equal(decodeSlamPreview(packet({...metadata, accepted_loops: 1})), null);
  assert.equal(decodeSlamPreview(new ArrayBuffer(4)), null);
});

test('geometry hypothesis is labelled and cannot claim validated loops', () => {
  const meta = {...metadata, kind: 'unvalidated_geometry_preview',
    backend: 'geometry_experimental', candidate_count: 2};
  assert.equal(decodeSlamPreview(packet(meta)).meta.candidate_count, 2);
  assert.equal(decodeSlamPreview(packet({...meta, accepted_loops: 1})), null);
  assert.equal(decodeSlamPreview(packet({...meta, candidate_count: -1})), null);
});

test('dashboard keeps the latest preview, including odd-length packets', () => {
  const source = readFileSync(new URL('./dashboard.js', import.meta.url), 'utf8');
  const start = source.indexOf('class SkillBuffer {');
  const end = source.indexOf('\n}\n', start) + 2;
  const connections = [];
  const context = { ArrayBuffer, TextEncoder, console, setTimeout,
    WebSocket: class { constructor() { connections.push(this); } } };
  const SkillBuffer = runInNewContext(`const PREVIEW_CAPACITY = 100; (${source.slice(start, end)})`, context);
  const buffer = new SkillBuffer('ws://preview', 'sensor/slam-preview');
  connections[0].onmessage({data: new ArrayBuffer(3)});
  connections[0].onmessage({data: new ArrayBuffer(5)});
  const replay = [];
  buffer.attach(frame => replay.push(frame.byteLength));
  assert.deepEqual(replay, [5]);
  assert.equal(buffer._frames.length, 1);
});

test('only an explicit fresh estimated pose can place the SLAM marker', () => {
  const marker = {stamp_ms: 1700, published_ms: 1800, expires_ms: 2500, xyz: [1, 2, 3], yaw: .5};
  assert.equal(currentOdomMarker(metadata, 2000), null);
  assert.equal(currentOdomMarker({...metadata, odom_marker: marker}, 2000), marker);
  assert.equal(currentOdomMarker({...metadata, odom_marker: marker}, 2500), null);
  assert.equal(currentOdomMarker({...metadata, odom_marker: {...marker, yaw: NaN}}, 2000), null);
});

test('each preview replaces its bounded map-frame trajectory, including missing or invalid paths', () => {
  const positions = new Float32Array(64 * 3);
  const path = [[1, 2, 3], [-4, -5, -6]];
  assert.equal(writeTrajectoryPositions(path, positions), 2);
  assert.deepEqual([...positions.slice(0, 6)], [1, 3, -2, -4, -6, 5]);
  for (const invalid of [undefined, [], [path[0]], [...path, [Infinity, 0, 0]],
      [...path, [1e99, 0, 0]], [...path, ['3', 0, 0]], Array(65).fill(path[0])]) {
    assert.equal(writeTrajectoryPositions(invalid, positions), 0);
  }
  const full = Array(64).fill([1, -2, 3]);
  const decoded = decodeSlamPreview(packet({...metadata, trajectory_xyz: full}));
  assert.equal(writeTrajectoryPositions(decoded.meta.trajectory_xyz, positions), 64);
});
