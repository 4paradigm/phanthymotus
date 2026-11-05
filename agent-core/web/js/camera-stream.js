/**
 * camera-stream.js — browser camera capture, JPEG encode, WebSocket send.
 *
 * The visual counterpart to mic-stream.js: a video device on whatever machine
 * has the dashboard open becomes a live image source for the robot. Frames go
 * out as JPEG and the server republishes them on `/remote_control/camera` as
 * `image/jpeg` — the format the on-robot camera drivers emit — so any vision
 * card consumes it with no special case.
 *
 *   import { startCameraStream, stopCameraStream, isCameraActive } from './camera-stream.js';
 *   await startCameraStream(wsUrl, onStateChange, async () => ({ deviceId, fps: 12 }));
 *   stopCameraStream();
 *
 * Same start/stop contract as mic-stream.js: start is idempotent, and stop
 * cancels a start that has not finished yet.
 */
import { deviceConstraint } from './browser-source.js';

let _stream = null;
let _ws = null;
let _timer = null;
let _active = false;
let _video = null;
let _canvas = null;
let _rvfcArmed = false;
let _sent = 0;
let _startedAt = 0;
// Bumped by every start and stop; an in-flight start checks it after each
// await and backs out, releasing what it holds, once it has changed.
let _generation = 0;
let _starting = null;

export const DEFAULTS = { width: 640, height: 480, fps: 12, quality: 0.7 };

/** Frames per second clamped to something a browser can actually sustain while
 *  also JPEG-encoding each frame on the main thread. */
export function clampFps(fps) {
  // `null` is checked separately because `Number(null)` is 0, not NaN — so a
  // card whose fps has never been set would otherwise stream at 1 fps, which
  // looks like a stall and is not what "unset" means.
  if (fps === null || fps === undefined || fps === '') return DEFAULTS.fps;
  const value = Number(fps);
  if (!Number.isFinite(value)) return DEFAULTS.fps;
  return Math.min(30, Math.max(1, Math.round(value)));
}

/**
 * Size the capture canvas to the camera's real aspect ratio.
 *
 * Forcing 640x480 on a 16:9 webcam would letterbox or stretch, and a stretched
 * body breaks the pose card's angles — the geometry rules are invariant to
 * scale and rotation but not to anisotropic scaling, which is what a wrong
 * aspect ratio is.
 */
export function fitCapture(videoWidth, videoHeight, target = DEFAULTS.width) {
  if (!(videoWidth > 0 && videoHeight > 0)) {
    return { width: DEFAULTS.width, height: DEFAULTS.height };
  }
  const scale = target / Math.max(videoWidth, videoHeight);
  return {
    width: Math.max(2, Math.round(videoWidth * scale / 2) * 2),
    height: Math.max(2, Math.round(videoHeight * scale / 2) * 2),
  };
}

export function isCameraActive() {
  return _active;
}

/** Frames actually sent per second since the stream started.
 *
 * Reported because the configured rate and the achieved rate can differ by an
 * order of magnitude — a backgrounded tab clamps timers to 1 Hz — and a stream
 * running at a twelfth of its setting is otherwise indistinguishable from a
 * slow robot. */
export function achievedFps() {
  if (!_active || !_startedAt) return 0;
  const seconds = (performance.now() - _startedAt) / 1000;
  return seconds > 0.5 ? _sent / seconds : 0;
}

/**
 * Ask for camera permission now, and release it immediately.
 *
 * The permission prompt is a human clicking a dialog, and it was sitting inside
 * the card's 10 s self-check window: the first start after a page load timed
 * out with "waiting for the browser camera", and every start after that
 * succeeded because the grant was remembered. Warming it before the start
 * request takes the human out of the critical path.
 *
 * Resolves true if permission is held, false otherwise. Never throws — a
 * refusal is an answer, not a failure.
 */
export async function warmPermission(deviceId) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? deviceConstraint(deviceId) : true, audio: false,
    });
    stream.getTracks().forEach(t => t.stop());
    return true;
  } catch {
    return false;
  }
}

/** List the browser's video inputs, for the card's device picker. */
export async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter(d => d.kind === 'videoinput')
      .map(d => ({ deviceId: d.deviceId, label: d.label || 'Camera' }));
  } catch {
    return [];
  }
}

/** The `video` constraint for getUserMedia. Exported for the tests. */
export function cameraConstraints(deviceId) {
  return deviceId ? deviceConstraint(deviceId) : { facingMode: 'user' };
}

/**
 * Start the browser camera. Already running or starting → the same promise.
 *
 * Both start paths (start-project, post-refresh re-arm) can arrive together,
 * and `_active` is only set once the video plays and the WebSocket is open, so
 * a plain "is it active" check let both through and opened two streams.
 *
 * `getOptions` runs after the generation is taken, so a stop during the
 * config read cancels the start too.
 *
 * @param {string} wsUrl        e.g. wss://host/ws/camera
 * @param {(active:boolean)=>void} onStateChange
 * @param {() => Promise<{fps?:number, deviceId?:string, width?:number, quality?:number}>} [getOptions]
 * @returns {Promise<boolean>} true when streaming; false if stopped while starting
 */
export function startCameraStream(wsUrl, onStateChange, getOptions) {
  if (_active) return Promise.resolve(true);
  if (_starting) return _starting;
  const gen = ++_generation;
  const run = _start(gen, wsUrl, onStateChange, getOptions);
  _starting = run;
  const clear = () => { if (_starting === run) _starting = null; };
  run.then(clear, clear);
  return run;
}

/** Stop the camera, cancelling a start still in progress. No-op when idle. */
export function stopCameraStream() {
  _generation++;
  _starting = null;
  _stopCamera();
}

/**
 * Toggle the browser camera. Kept for old callers; prefer start/stop.
 * @param {string} wsUrl        e.g. wss://host/ws/camera
 * @param {(active:boolean)=>void} onStateChange
 * @param {{fps?:number, deviceId?:string, width?:number, quality?:number}} opts
 */
export async function toggleCameraStream(wsUrl, onStateChange, opts = {}) {
  if (_active || _starting) {
    stopCameraStream();
    onStateChange(false);
    return;
  }
  await startCameraStream(wsUrl, onStateChange, async () => opts);
}

async function _start(gen, wsUrl, onStateChange, getOptions) {
  // Held locally until everything is up, then committed. A cancelled start
  // finds the module state already cleared by stopCameraStream, so only these
  // need releasing.
  let stream = null, video = null, ws = null;
  const cancelled = () => gen !== _generation;
  const release = () => {
    if (ws) { try { ws.close(); } catch { /* already closed */ } }
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (video) { video.srcObject = null; video.remove(); }
  };

  try {
    const opts = (getOptions ? await getOptions() : null) || {};
    if (cancelled()) return false;

    const fps = clampFps(opts.fps ?? DEFAULTS.fps);
    const quality = opts.quality ?? DEFAULTS.quality;
    const target = opts.width ?? DEFAULTS.width;

    stream = await navigator.mediaDevices.getUserMedia({
      video: cameraConstraints(opts.deviceId),
      audio: false,
    });
    if (cancelled()) { release(); return false; }
    const track = stream.getVideoTracks()[0];
    console.info(`[camera] capturing: ${track ? track.label || '(unlabelled)' : '(no track)'}`
      + (opts.deviceId ? ' (selected device)' : ' (browser default)'));

    video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    // Attached to the document, off-screen. A detached <video> is not
    // *presented*, and requestVideoFrameCallback fires per presented frame —
    // so a video element that never enters the DOM delivers a trickle. Measured
    // on a live session: 2 fps out of a 12 fps request, with the network, the
    // server and the DDS publish all verified clean at 12.4 fps.
    video.style.cssText =
      'position:fixed;left:-10000px;top:0;width:2px;height:2px;opacity:0;pointer-events:none';
    document.body.appendChild(video);
    await video.play();
    if (cancelled()) { release(); return false; }

    const size = fitCapture(video.videoWidth, video.videoHeight, target);
    const canvas = document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext('2d');

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    let wsTimer;
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error('WebSocket connection failed'));
      wsTimer = setTimeout(() => reject(new Error('WS timeout')), 5000);
    }).finally(() => clearTimeout(wsTimer));
    if (cancelled()) { release(); return false; }

    // Commit. From here on _stopCamera owns them.
    _stream = stream; _video = video; _canvas = canvas; _ws = ws;
    stream = video = ws = null;

    let sending = false;
    const interval = 1000 / fps;
    let lastSent = 0;

    const sendFrame = () => {
      if (!_ws || _ws.readyState !== WebSocket.OPEN || !_video) return;
      // Skip rather than queue. A frame that arrives late is worse than one
      // that never arrives: the action rules measure velocity between frames,
      // so a backlog delivered in a burst reads as motion that did not happen.
      if (sending || _ws.bufferedAmount > 1 << 20) return;
      const now = performance.now();
      if (now - lastSent < interval * 0.9) return;
      lastSent = now;
      sending = true;
      _sent++;
      ctx.drawImage(_video, 0, 0, _canvas.width, _canvas.height);
      _canvas.toBlob(async (blob) => {
        try {
          if (blob && _ws && _ws.readyState === WebSocket.OPEN) {
            _ws.send(await blob.arrayBuffer());
          }
        } finally {
          sending = false;
        }
      }, 'image/jpeg', quality);
    };

    // Driven by the video's own decoded frames where the browser offers it.
    //
    // setInterval alone is not enough: browsers clamp timers in a hidden or
    // background tab to **once per second**, so the stream silently collapses
    // to 1 fps the moment the dashboard is not the foreground tab. Measured on
    // a live session: frames arriving 1003 ms apart, p95 1052 — not jitter, a
    // 1 Hz timer. At that rate tracks expire between frames and the action
    // model never accumulates the seconds of history it needs, so the activity
    // channel can never fire.
    //
    // requestVideoFrameCallback fires per decoded frame and is throttled far
    // less aggressively. The interval stays as a fallback for browsers without
    // it, and `sendFrame` rate-limits either way.
    // **Both**, not either. They fail in different situations and the rate
    // limiter in sendFrame makes running both harmless:
    //
    //   setInterval alone  — clamped to 1 Hz in a hidden or background tab
    //   rvfc alone         — depends on the video being presented, and is
    //                        throttled with the page's rendering
    //
    // Each covers the other's gap, and whichever fires first in a given window
    // sends the frame; the second finds `now - lastSent` too small and returns.
    if (typeof _video.requestVideoFrameCallback === 'function') {
      const onFrame = () => {
        if (!_active) return;
        sendFrame();
        if (_video) _video.requestVideoFrameCallback(onFrame);
      };
      _rvfcArmed = true;
      _video.requestVideoFrameCallback(onFrame);
    }
    _timer = setInterval(sendFrame, interval);

    _active = true;
    _sent = 0;
    _startedAt = performance.now();
    onStateChange(true);
    return true;
  } catch (err) {
    release();
    if (cancelled()) return false;   // the stop already got what it asked for
    _stopCamera();
    onStateChange(false);
    throw err;
  }
}

function _stopCamera() {
  if (_timer) { clearInterval(_timer); _timer = null; }
  if (_ws) { try { _ws.close(); } catch { /* already closed */ } _ws = null; }
  if (_stream) { _stream.getTracks().forEach(t => t.stop()); _stream = null; }
  if (_video) {
    _video.srcObject = null;
    _video.remove();              // it is in the document now
    _video = null;
  }
  _canvas = null;
  _rvfcArmed = false;
  _active = false;
}
