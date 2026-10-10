/**
 * The sensor/pose2d renderer's geometry and payload handling.
 *
 * Everything here is a plain function on purpose: the drawing itself is a few
 * canvas calls, but the parts that can be *wrong without raising* are the
 * transform (a stretched skeleton still draws, it just stands in a body shape
 * nobody has) and the visibility gate (an unfiltered draw puts a limb through
 * wherever (0, 0) lands).
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  ALERT_LABELS,
  ALERT_COLOUR,
  COCO_SKELETON,
  Pose2dRenderer,
  TRACK_COLOURS,
  fitTransform,
  parsePosePayload,
  personLabel,
  trackColour,
  visibleBones,
  visibleJoints,
} from './renderers/pose2d.js';

const SHOULDERS = 'left_shoulder/right_shoulder';

/** 17 keypoints, all invisible except the pair named. */
function keypoints(visible = {}) {
  const points = Array.from({ length: 17 }, () => [0, 0, 0]);
  for (const [index, point] of Object.entries(visible)) points[index] = point;
  return points;
}

function payload(overrides = {}) {
  return {
    timestamp: 1,
    count: 1,
    image_size: [640, 480],
    skeleton: COCO_SKELETON,
    persons: [{
      id: 1,
      posture: 'standing',
      posture_confidence: 0.82,
      bbox: [100, 50, 300, 450],
      keypoints: keypoints({ 5: [200, 200, 0.9], 6: [300, 200, 0.9] }),
    }],
    ...overrides,
  };
}

// ── registration ────────────────────────────────────────────────────────────

test('the renderer claims sensor/pose2d and nothing else', () => {
  assert.equal(Pose2dRenderer.canRender('sensor/pose2d'), true);
  // sensor/skeleton drives a URDF of the robot's own joints and needs a model
  // resource tool; claiming it here would put human keypoints into a renderer
  // that silently shows an empty panel.
  assert.equal(Pose2dRenderer.canRender('sensor/skeleton'), false);
  assert.equal(Pose2dRenderer.canRender('data/json'), false);
  assert.equal(Pose2dRenderer.canRender('image/jpeg'), false);
  assert.equal(Pose2dRenderer.canRender(undefined), false);
});

test('the bone table is the 19 COCO edges and references real joints', () => {
  assert.equal(COCO_SKELETON.length, 19);
  for (const [a, b] of COCO_SKELETON) {
    assert.ok(a >= 0 && a < 17, `${a} out of range`);
    assert.ok(b >= 0 && b < 17, `${b} out of range`);
    assert.notEqual(a, b);
  }
});

// ── payload ─────────────────────────────────────────────────────────────────

test('a pose payload is normalised into persons with keypoint objects', () => {
  const frame = parsePosePayload(payload());
  assert.equal(frame.count, 1);
  assert.deepEqual(frame.imageSize, { width: 640, height: 480 });
  const person = frame.persons[0];
  assert.equal(person.posture, 'standing');
  assert.equal(person.confidence, 0.82);
  assert.deepEqual(person.keypoints[5], { x: 200, y: 200, v: 0.9 });
});

test('an empty frame is a frame, not a parse failure', () => {
  // "Nobody in view" has to reach the canvas; only a non-pose message is
  // ignored, because the panel's own stale notice is what explains silence.
  const frame = parsePosePayload(payload({ count: 0, persons: [] }));
  assert.notEqual(frame, null);
  assert.deepEqual(frame.persons, []);
  assert.equal(frame.count, 0);
});

test('a message that is not a pose payload is refused', () => {
  for (const message of [null, undefined, 42, 'text', {}, { persons: 'nope' }]) {
    assert.equal(parsePosePayload(message), null);
  }
});

test('a compact payload without visibilities counts as fully visible', () => {
  // publish_keypoints: compact sends [x, y]. The producer choosing not to send
  // confidences cannot mean "every joint is invisible".
  const frame = parsePosePayload(payload({
    persons: [{ id: 1, posture: 'standing', keypoints: [[10, 20], [30, 40]] }],
  }));
  assert.deepEqual(frame.persons[0].keypoints[0], { x: 10, y: 20, v: 1 });
});

test('a payload carrying its own skeleton overrides the built-in table', () => {
  const frame = parsePosePayload(payload({ skeleton: [[0, 1]] }));
  assert.deepEqual(frame.skeleton, [[0, 1]]);
});

test('an absent skeleton falls back to the built-in table', () => {
  const message = payload();
  delete message.skeleton;
  assert.equal(parsePosePayload(message).skeleton, COCO_SKELETON);
});

test('a frame with no image_size falls back to the extent of its contents', () => {
  // A one-shot result echoed onto a card's topic can arrive without the frame
  // it came from. Without this the transform divides by zero and the panel goes
  // blank with nothing to explain it.
  const frame = parsePosePayload(payload({ image_size: [0, 0] }));
  assert.ok(frame.imageSize.width >= 300);
  assert.ok(frame.imageSize.height >= 450);
});

test('a frame with neither a size nor any content has no size to invent', () => {
  const frame = parsePosePayload(payload({ image_size: [0, 0], count: 0, persons: [] }));
  assert.equal(frame.imageSize, null);
});

// ── transform ───────────────────────────────────────────────────────────────

test('the source frame is letterboxed, not stretched', () => {
  // 640x480 into a 400x400 canvas: scale 0.625 both ways, bars top and bottom.
  const t = fitTransform({ width: 640, height: 480 }, 400, 400);
  assert.equal(t.scale, 0.625);
  assert.equal(t.offsetX, 0);
  assert.equal(t.offsetY, 50);
  assert.deepEqual(t.project(0, 0), [0, 50]);
  assert.deepEqual(t.project(640, 480), [400, 350]);
});

test('a centre pixel maps to the centre of the canvas', () => {
  const t = fitTransform({ width: 640, height: 480 }, 800, 300);
  assert.deepEqual(t.project(320, 240), [400, 150]);
});

test('an unusable size or canvas yields no transform rather than NaN', () => {
  assert.equal(fitTransform(null, 400, 400), null);
  assert.equal(fitTransform({ width: 0, height: 480 }, 400, 400), null);
  assert.equal(fitTransform({ width: 640, height: 480 }, 0, 400), null);
});

// ── visibility gate ─────────────────────────────────────────────────────────

test('only bones with both ends visible are drawn', () => {
  const frame = parsePosePayload(payload());
  const t = fitTransform(frame.imageSize, 640, 480);   // identity
  const bones = visibleBones(frame.persons[0], frame.skeleton, 0.3, t);
  // Of the 19 bones only shoulder-to-shoulder has both ends above threshold.
  assert.equal(bones.length, 1, `expected just ${SHOULDERS}`);
  assert.deepEqual(bones[0], [[200, 200], [300, 200]]);
});

test('invisible joints are skipped rather than drawn at the origin', () => {
  const frame = parsePosePayload(payload());
  const t = fitTransform(frame.imageSize, 640, 480);
  const joints = visibleJoints(frame.persons[0], 0.3, t);
  assert.equal(joints.length, 2);
  assert.ok(!joints.some(([x, y]) => x === 0 && y === 0),
            'a phantom limb at (0,0) is what the gate exists to prevent');
});

test('raising the threshold drops joints the model was unsure about', () => {
  const frame = parsePosePayload(payload({
    persons: [{ id: 1, posture: 'standing',
                keypoints: keypoints({ 5: [200, 200, 0.9], 6: [300, 200, 0.4] }) }],
  }));
  const t = fitTransform({ width: 640, height: 480 }, 640, 480);
  assert.equal(visibleBones(frame.persons[0], frame.skeleton, 0.3, t).length, 1);
  assert.equal(visibleBones(frame.persons[0], frame.skeleton, 0.5, t).length, 0);
});

test('a bone index outside the keypoint list is skipped, not fatal', () => {
  const frame = parsePosePayload(payload({ skeleton: [[5, 6], [5, 99]] }));
  const t = fitTransform(frame.imageSize, 640, 480);
  assert.equal(visibleBones(frame.persons[0], frame.skeleton, 0.3, t).length, 1);
});

test('non-finite coordinates are skipped', () => {
  const frame = parsePosePayload(payload({
    persons: [{ id: 1, posture: 'standing',
                keypoints: keypoints({ 5: [NaN, 200, 0.9], 6: [300, 200, 0.9] }) }],
  }));
  const t = fitTransform(frame.imageSize, 640, 480);
  assert.equal(visibleBones(frame.persons[0], frame.skeleton, 0.3, t).length, 0);
  assert.equal(visibleJoints(frame.persons[0], 0.3, t).length, 1);
});

// ── colour and label ────────────────────────────────────────────────────────

test('two tracks get two colours and keep them', () => {
  assert.notEqual(trackColour(1, 'standing'), trackColour(2, 'standing'));
  assert.equal(trackColour(1, 'standing'), trackColour(1, 'walking'));
  assert.ok(TRACK_COLOURS.includes(trackColour(7, 'standing')));
});

test('a person on the ground is drawn in the alert colour whatever their id', () => {
  for (const label of ALERT_LABELS) {
    assert.equal(trackColour(3, label), ALERT_COLOUR, label);
  }
  // `lying` as well as `falling down`: someone on the floor is worth looking at
  // even when nothing called it a fall.
  assert.ok(ALERT_LABELS.has('lying') && ALERT_LABELS.has('falling down'));
});

test('an id that is not a number still picks a colour', () => {
  assert.ok(TRACK_COLOURS.includes(trackColour(undefined, 'standing')));
  assert.ok(TRACK_COLOURS.includes(trackColour(-3, 'standing')));
});

test('the label shows both channels when both are there', () => {
  // They answer different questions — what shape the body is in, and what the
  // person is doing — so showing one hides the other.
  assert.equal(personLabel({ id: 2, posture: 'lying', activity: 'falling down',
                             confidence: 0.82 }), '#2 lying · falling down');
  assert.equal(personLabel({ id: 7, posture: 'upright', activity: 'hand waving',
                             confidence: 0.9 }), '#7 upright · hand waving');
});

test('an activity with no posture is still shown', () => {
  assert.equal(personLabel({ id: 2, posture: null, activity: 'hand waving',
                             confidence: null }), '#2 hand waving');
});

test('with no activity the posture is drawn, with its confidence', () => {
  assert.equal(personLabel({ id: 2, posture: 'standing', activity: null,
                             confidence: 0.82 }), '#2 standing 82%');
});

test('a person with neither is labelled unknown rather than blank', () => {
  assert.equal(personLabel({ id: 2, posture: null, activity: null,
                             confidence: null }), '#2 unknown');
});
