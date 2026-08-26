/**
 * The TTS end-of-utterance marker must not be rendered as audio.
 *
 * Both TTS engines publish `\x01\x00\xff\xff\x01\x00\xff\xff` after every
 * utterance, and the speaker drivers consume it as
 * `len(pcm) == 8 and pcm == AUDIO_EOF_MAGIC`. The dashboard was the one consumer
 * that did not know about it: 8 bytes is 4 samples, so the panel reported
 * `音频流 0ms/帧` right after every announcement — indistinguishable from a
 * broken stream — and wrote the protocol bytes into the waveform ring.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { isAudioEof } from './renderers/audio.js';

const buf = (...bytes) => new Uint8Array(bytes).buffer;
const EOF = buf(0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff);

test('the marker is recognised', () => {
  assert.equal(isAudioEof(EOF), true);
});

test('real audio of the same length is not the marker', () => {
  assert.equal(isAudioEof(buf(0, 0, 0, 0, 0, 0, 0, 0)), false);
  assert.equal(isAudioEof(buf(0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xfe)), false);
});

test('a normal TTS chunk is not the marker', () => {
  // 3200 bytes is what the plugin logs as chunk_bytes.
  assert.equal(isAudioEof(new ArrayBuffer(3200)), false);
});

test('a prefix or a longer frame carrying the same bytes is not the marker', () => {
  assert.equal(isAudioEof(buf(0x01, 0x00, 0xff, 0xff)), false);
  assert.equal(isAudioEof(buf(0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff, 0x00, 0x00)), false);
});

test('empty and missing buffers are safe', () => {
  assert.equal(isAudioEof(new ArrayBuffer(0)), false);
  assert.equal(isAudioEof(null), false);
  assert.equal(isAudioEof(undefined), false);
});
