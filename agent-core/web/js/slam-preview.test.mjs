import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { decodeSlamPreview } from './renderers/slam-preview-codec.js';

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
