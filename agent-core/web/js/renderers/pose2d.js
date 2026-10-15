/**
 * pose2d.js — 2D human skeleton renderer for `sensor/pose2d`.
 *
 * Draws what the perception `pose` card publishes on
 * `{camera}/poses/skeleton`: COCO-17 keypoints per person, in source-frame
 * pixels, with the action label the rules decided on.
 *
 * **Why this is a separate topic from the card's `data/json` port.** A renderer
 * only ever sees one topic — detail-panel.js opens a single `/ws/bus/{topic}`
 * for the selected port — so an overlay cannot be assembled here from a camera
 * feed plus a keypoint feed. The card therefore publishes the skeleton on its
 * own topic, which also keeps 17 keypoints per person per frame out of the lean
 * topic that agent-core copies into the LLM's context.
 *
 * **Not `sensor/skeleton`.** That renderer drives a URDF of the *robot's* own
 * joints and needs a `model` resource tool to supply it, with joint names
 * matching exactly. Pointing it at human keypoints yields an empty panel and
 * nothing in any log.
 *
 * The geometry and payload handling are exported as plain functions so they can
 * be tested without a DOM — see pose2d.test.mjs.
 */

// The 19 bones, kept here rather than imported because the producer is Python
// (perception/plugins/vision_runtime.py COCO_SKELETON, same order). A payload
// that carries its own `skeleton` wins, so a future keypoint set does not need
// a frontend change to draw correctly.
export const COCO_SKELETON = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
  [5, 11], [6, 12], [5, 6],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

const BACKGROUND = '#1C1C1E';
const GRID = 'rgba(255,255,255,0.06)';
const LABEL = 'rgba(255,255,255,0.75)';

// One colour per track, so two people keep two colours between frames.
export const TRACK_COLOURS = [
  '#4D9EE8', '#7DD77D', '#F0B23C', '#B88AD8', '#5ACBD6', '#E8A0A0',
];

// Shared with the panel's own warning text (var(--orange)), so "something is
// wrong" looks the same whoever is saying it.
export const ALERT_COLOUR = '#d77757';

// Actions that mean a person is on the ground. Drawn in the alert colour —
// `lying` as well as `fall`, because a person lying down is worth looking at
// even when the rules declined to call it a fall.
export const ALERT_LABELS = new Set(['falling down', 'lying']);

export function trackColour(id, label) {
  if (ALERT_LABELS.has(label)) return ALERT_COLOUR;
  const index = Number.isFinite(id) ? Math.abs(Math.trunc(id)) : 0;
  return TRACK_COLOURS[index % TRACK_COLOURS.length];
}

/**
 * Normalise one bus message into what the draw step needs, or null.
 *
 * Returns null rather than a half-filled object for anything unusable: an
 * empty `persons` list is a valid frame ("nobody in view") and must be drawn,
 * whereas a message that is not a pose payload at all must not clear the
 * canvas — the panel's own stale notice is what explains silence.
 */
export function parsePosePayload(message) {
  if (!message || typeof message !== 'object') return null;
  if (!Array.isArray(message.persons)) return null;
  const persons = message.persons
    .map(normalisePerson)
    .filter(person => person !== null);
  return {
    imageSize: readImageSize(message.image_size) || inferImageSize(persons),
    skeleton: Array.isArray(message.skeleton) && message.skeleton.length
      ? message.skeleton
      : COCO_SKELETON,
    keypointNames: Array.isArray(message.keypoint_names) ? message.keypoint_names : null,
    count: Number.isFinite(message.count) ? message.count : persons.length,
    persons,
  };
}

function normalisePerson(raw) {
  if (!raw || !Array.isArray(raw.keypoints)) return null;
  const keypoints = raw.keypoints.map(point => (
    Array.isArray(point)
      // Visibility is optional so a `compact` payload ([x, y]) also draws; it
      // then counts as fully visible, which is what "the producer chose not to
      // send confidences" has to mean.
      ? { x: Number(point[0]), y: Number(point[1]),
          v: point.length > 2 ? Number(point[2]) : 1 }
      : { x: NaN, y: NaN, v: 0 }
  ));
  return {
    id: Number.isFinite(raw.id) ? raw.id : 0,
    // Two channels. `posture` is the shape the body is in and everybody has
    // one; `activity` is what they are doing and may legitimately be absent.
    // The label drawn prefers the activity — "falling down" says more than
    // "lying" — and falls back to the posture.
    posture: typeof raw.posture === 'string' ? raw.posture : null,
    activity: typeof raw.activity === 'string' ? raw.activity : null,
    confidence: Number.isFinite(raw.posture_confidence) ? raw.posture_confidence : null,
    bbox: Array.isArray(raw.bbox) && raw.bbox.length === 4 ? raw.bbox.map(Number) : null,
    keypoints,
  };
}

function readImageSize(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const [width, height] = value.map(Number);
  if (!(width > 0 && height > 0)) return null;
  return { width, height };
}

/**
 * Frame size when the producer did not state one.
 *
 * A one-shot result echoed onto a card's topic can arrive without the frame it
 * came from, so there is no `image_size`. Falling back to the extent of what
 * *is* in the message keeps the skeleton on screen; without this the transform
 * divides by zero and the panel goes blank with nothing to explain it.
 */
function inferImageSize(persons) {
  let maxX = 0;
  let maxY = 0;
  for (const person of persons) {
    for (const point of person.keypoints) {
      if (Number.isFinite(point.x)) maxX = Math.max(maxX, point.x);
      if (Number.isFinite(point.y)) maxY = Math.max(maxY, point.y);
    }
    if (person.bbox) {
      maxX = Math.max(maxX, person.bbox[2]);
      maxY = Math.max(maxY, person.bbox[3]);
    }
  }
  if (!(maxX > 0 && maxY > 0)) return null;
  return { width: Math.ceil(maxX * 1.05), height: Math.ceil(maxY * 1.05) };
}

/**
 * Letterbox the source frame into the canvas, preserving aspect ratio.
 *
 * Stretching instead would be the kind of error nothing reports: every bone
 * still draws, the skeleton merely stands in a body shape nobody has.
 */
export function fitTransform(imageSize, canvasWidth, canvasHeight) {
  if (!imageSize || !(canvasWidth > 0 && canvasHeight > 0)) return null;
  const { width, height } = imageSize;
  if (!(width > 0 && height > 0)) return null;
  const scale = Math.min(canvasWidth / width, canvasHeight / height);
  return {
    scale,
    offsetX: (canvasWidth - width * scale) / 2,
    offsetY: (canvasHeight - height * scale) / 2,
    project: (x, y) => [x * scale + (canvasWidth - width * scale) / 2,
                        y * scale + (canvasHeight - height * scale) / 2],
  };
}

/**
 * The bones with both ends visible, as canvas-space segments.
 *
 * The engine emits a coordinate for all 17 keypoints whether it saw them or
 * not, so an unfiltered draw puts a limb through wherever (0, 0) lands. Gating
 * on visibility is what makes a partly occluded person look partly occluded
 * instead of deformed.
 */
export function visibleBones(person, skeleton, minConfidence, transform) {
  const out = [];
  for (const edge of skeleton) {
    const a = person.keypoints[edge[0]];
    const b = person.keypoints[edge[1]];
    if (!a || !b) continue;
    if (!(a.v >= minConfidence) || !(b.v >= minConfidence)) continue;
    if (!Number.isFinite(a.x) || !Number.isFinite(b.x)) continue;
    out.push([transform.project(a.x, a.y), transform.project(b.x, b.y)]);
  }
  return out;
}

export function visibleJoints(person, minConfidence, transform) {
  return person.keypoints
    .filter(point => point.v >= minConfidence && Number.isFinite(point.x))
    .map(point => transform.project(point.x, point.y));
}

/** The one-line summary drawn over each person.
 *
 * Both channels when both are there — `upright · hand waving` — because they
 * answer different questions and showing only one hides the other. Posture
 * first: it is the one that is always available, so the label does not change
 * shape as an activity comes and goes. */
export function personLabel(person) {
  const parts = [];
  if (person.posture) parts.push(person.posture);
  if (person.activity) parts.push(person.activity);
  if (!parts.length) parts.push('unknown');
  // The confidence belongs to the posture, so it is only shown when the
  // posture is the only thing being reported.
  const confidence = (person.confidence !== null && person.posture && !person.activity)
    ? ` ${Math.round(person.confidence * 100)}%` : '';
  return `#${person.id} ${parts.join(' · ')}${confidence}`;
}

/** Which label decides the colour — the alerting one wins. */
export function personLabelForColour(person) {
  if (ALERT_LABELS.has(person.activity)) return person.activity;
  if (ALERT_LABELS.has(person.posture)) return person.posture;
  return person.activity || person.posture || 'unknown';
}

const MIN_CONFIDENCE = 0.3;

export const Pose2dRenderer = {
  name: 'pose2d',
  canRender: (hint) => hint === 'sensor/pose2d',
  _el: null,
  _canvas: null,
  _ctx: null,
  _ro: null,
  _frame: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-pose2d';
    this._el.style.cssText = 'width:100%;height:100%;position:relative';
    this._canvas = document.createElement('canvas');
    this._el.appendChild(this._canvas);
    container.appendChild(this._el);

    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);
  },

  _resize() {
    if (!this._canvas) return;
    this._canvas.width = this._el.clientWidth || 400;
    this._canvas.height = this._el.clientHeight || 300;
    this._ctx = this._canvas.getContext('2d');
    this._draw();
  },

  onData(buffer) {
    if (!this._canvas) return;
    let parsed = null;
    try {
      parsed = parsePosePayload(JSON.parse(new TextDecoder().decode(buffer)));
    } catch {
      return;                 // not a pose payload; keep the last frame
    }
    if (parsed === null) return;
    this._frame = parsed;
    this._draw();
  },

  _draw() {
    const c = this._ctx;
    if (!c) return;
    const W = this._canvas.width;
    const H = this._canvas.height;
    c.clearRect(0, 0, W, H);
    c.fillStyle = BACKGROUND;
    c.fillRect(0, 0, W, H);

    const frame = this._frame;
    if (!frame) return;

    const transform = fitTransform(frame.imageSize, W, H);
    if (!transform) {
      // Nobody in view and no frame size to draw a viewport for: say so rather
      // than leaving an empty rectangle that also means "not connected".
      c.fillStyle = LABEL;
      c.font = '12px system-ui, sans-serif';
      c.fillText(frame.count === 0 ? '画面里没有人' : '缺少画面尺寸，无法定位骨架',
                 10, 20);
      return;
    }

    // The source frame's outline, so an empty frame still reads as a camera
    // view rather than as a dead panel.
    const [originX, originY] = transform.project(0, 0);
    const [cornerX, cornerY] = transform.project(frame.imageSize.width,
                                                frame.imageSize.height);
    c.strokeStyle = GRID;
    c.lineWidth = 1;
    c.strokeRect(originX, originY, cornerX - originX, cornerY - originY);

    for (const person of frame.persons) {
      const colour = trackColour(person.id, personLabelForColour(person));
      c.strokeStyle = colour;
      c.lineWidth = 2;
      c.lineCap = 'round';
      for (const [[x1, y1], [x2, y2]] of
           visibleBones(person, frame.skeleton, MIN_CONFIDENCE, transform)) {
        c.beginPath();
        c.moveTo(x1, y1);
        c.lineTo(x2, y2);
        c.stroke();
      }
      c.fillStyle = colour;
      for (const [x, y] of visibleJoints(person, MIN_CONFIDENCE, transform)) {
        c.beginPath();
        c.arc(x, y, 2.5, 0, Math.PI * 2);
        c.fill();
      }
      if (person.bbox) {
        const [bx, by] = transform.project(person.bbox[0], person.bbox[1]);
        c.font = '12px system-ui, sans-serif';
        c.fillText(personLabel(person), bx, Math.max(by - 5, 12));
      }
    }

    c.fillStyle = LABEL;
    c.font = '11px system-ui, sans-serif';
    c.fillText(`${frame.count} 人`, 8, H - 8);
  },

  unmount() {
    this._ro?.disconnect();
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._frame = null;
  },
};
