/** camera.js — Renders image/jpeg (live JPEG stream), image/depth-z16 (raw depth), image/depth-zlib (zlib-compressed depth) */

// ── Depth colour ────────────────────────────────────────────────────────────
//
// This panel is not a scientific depth visualisation; it is a proximity read.
// The operator is asking one question — how close is the nearest thing, and
// which side is it on — so the ramp is built for that, not for showing 0–5 m
// evenly.
//
// Two things the previous ramp got wrong:
//
//   Direction. It ran blue(near) → green → yellow → red(far), so the far wall
//   was the loudest thing on screen and an obstacle at arm's length was a calm
//   blue. Red belongs to what you are about to hit.
//
//   Where the colour budget goes. It spread hue linearly over five metres, so
//   0.5 m (about to collide) and 1.5 m (fine) differed by a slight hue shift,
//   while 3 m and 4 m — equally irrelevant — each got a full band. The stops
//   below are placed at the distances this robot actually acts on:
//   obstacle_stop 0.8 m and slow_distance 1.8 m (see actucore navi config), so
//   most of the contrast lands where a decision changes.
//
// The look is aerial perspective: near things are dense and saturated, distant
// things wash out towards the panel's own background. Lightness rises
// monotonically with distance and saturation falls, so it survives greyscale
// and all three common colour-vision deficiencies — the reading never depends
// on telling red from green, which the old ramp did.
//
// The cool stops are lighter than they look like they should be, deliberately:
// a teal picked for its hue alone lands darker than the gold before it, which
// puts a dip in the lightness curve and costs exactly the greyscale-safety the
// ramp is built for. Check the curve, not the swatches, after editing these.
const DEPTH_STOPS = [
  // Inside 2 m — the band the robot actually makes decisions in. Eight stops
  // here against three beyond, because this is where a metre matters.
  [0.00, 0x5E, 0x14, 0x10],   // contact — near-black oxblood
  [0.30, 0xA8, 0x21, 0x0F],   // crimson
  [0.60, 0xDC, 0x5A, 0x14],   // burnt orange
  [0.80, 0xE8, 0x74, 0x1E],   // obstacle_stop: forward motion is cut here
  [1.10, 0xF0, 0x9C, 0x2E],   // amber — the dashboard's own accent
  [1.45, 0xF3, 0xC4, 0x52],   // gold
  [1.80, 0xC9, 0xCC, 0x8E],   // slow_distance: the far edge of caution
  [2.10, 0xB4, 0xD2, 0xCC],   // clear — the first cool colour
  // Beyond 2 m the reading is "not my problem"; it washes out towards the
  // panel's own ground so the eye passes over it.
  [3.20, 0xC6, 0xD6, 0xD6],
  [5.00, 0xE6, 0xE4, 0xDD],
];

const DEPTH_MAX_MM = 5000;

// Where the near band ends, and how much of the ramp it gets. Linear distance
// would hand 0–2 m only 40% of the 256 steps and spend the rest on distances
// nobody acts on; this gives the decision band nearly three quarters of the
// available contrast. The two numbers are deliberately separate so the split
// can be re-tuned without touching the stops.
const DEPTH_NEAR_M = 2.0;
const DEPTH_NEAR_SHARE = 0.72;

// Index (0-255) -> metres. The inverse of the warp applied per pixel below;
// they have to agree, so both live here rather than at the two call sites.
function depthMetresAt(i) {
  const f = i / 255;
  return f <= DEPTH_NEAR_SHARE
    ? (f / DEPTH_NEAR_SHARE) * DEPTH_NEAR_M
    : DEPTH_NEAR_M + ((f - DEPTH_NEAR_SHARE) / (1 - DEPTH_NEAR_SHARE))
                     * (DEPTH_MAX_MM / 1000 - DEPTH_NEAR_M);
}

// 256-entry lookup, built once. 640x480 is 307k pixels a frame at up to 15 fps;
// interpolating per pixel would be ~4.6M interpolations a second for a result
// that only has 256 distinct values anyway.
const DEPTH_LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const metres = depthMetresAt(i);
    let k = 0;
    while (k < DEPTH_STOPS.length - 2 && metres > DEPTH_STOPS[k + 1][0]) k++;
    const [d0, r0, g0, b0] = DEPTH_STOPS[k];
    const [d1, r1, g1, b1] = DEPTH_STOPS[k + 1];
    const t = Math.max(0, Math.min(1, (metres - d0) / (d1 - d0)));
    lut[i * 3]     = r0 + (r1 - r0) * t;
    lut[i * 3 + 1] = g0 + (g1 - g0) * t;
    lut[i * 3 + 2] = b0 + (b1 - b0) * t;
  }
  return lut;
})();

// Millimetres -> LUT index, with the near band stretched. Kept branch-light:
// this runs 307k times a frame.
const DEPTH_NEAR_MM = DEPTH_NEAR_M * 1000;
const DEPTH_NEAR_SCALE = DEPTH_NEAR_SHARE * 255 / DEPTH_NEAR_MM;
const DEPTH_FAR_SCALE = (1 - DEPTH_NEAR_SHARE) * 255 / (DEPTH_MAX_MM - DEPTH_NEAR_MM);
const DEPTH_FAR_BASE = DEPTH_NEAR_SHARE * 255;

// One depth frame into RGBA. `u16` is millimetres, 0 meaning "no reading" —
// which is drawn transparent rather than as contact, because an unknown pixel
// rendered as an obstacle at zero range is the most alarming possible lie.
export function paintDepth(u16, rgba, count) {
  for (let i = 0; i < count; i++) {
    const d = u16[i];
    const idx = i * 4;
    if (d === 0) { rgba[idx + 3] = 0; continue; }
    const mm = d < DEPTH_MAX_MM ? d : DEPTH_MAX_MM;
    const s = (mm <= DEPTH_NEAR_MM
      ? mm * DEPTH_NEAR_SCALE
      : DEPTH_FAR_BASE + (mm - DEPTH_NEAR_MM) * DEPTH_FAR_SCALE) | 0;
    rgba[idx]     = DEPTH_LUT[s * 3];
    rgba[idx + 1] = DEPTH_LUT[s * 3 + 1];
    rgba[idx + 2] = DEPTH_LUT[s * 3 + 2];
    rgba[idx + 3] = 255;
  }
}

export const CameraRenderer = {
  name: 'camera',
  canRender: (hint) => hint === 'image/jpeg',
  _el: null,
  _img: null,
  _fps: 0,
  _frameCount: 0,
  _lastFpsTime: 0,
  _label: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-camera';
    this._el.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;justify-content:center;position:relative;background:#000';
    this._img = document.createElement('img');
    this._img.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain';
    this._label = document.createElement('span');
    this._label.style.cssText = 'position:absolute;top:6px;right:8px;font-size:11px;color:#fff;background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:3px';
    this._el.appendChild(this._img);
    this._el.appendChild(this._label);
    container.appendChild(this._el);
    this._frameCount = 0;
    this._lastFpsTime = performance.now();
  },

  onData(buffer, hint) {
    if (!this._img) return;
    const blob = new Blob([buffer], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const old = this._img.src;
    this._img.src = url;
    if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);

    // FPS counter
    this._frameCount++;
    const now = performance.now();
    if (now - this._lastFpsTime >= 1000) {
      this._fps = this._frameCount;
      this._frameCount = 0;
      this._lastFpsTime = now;
      if (this._label) this._label.textContent = `${this._fps} fps`;
    }
  },

  unmount() {
    if (this._img?.src?.startsWith('blob:')) URL.revokeObjectURL(this._img.src);
    this._el?.remove();
    this._el = null;
    this._img = null;
    this._label = null;
  },
};


export const DepthRenderer = {
  name: 'depth',
  canRender: (hint) => hint === 'image/depth-z16',
  _el: null,
  _canvas: null,
  _ctx: null,
  _label: null,
  _fps: 0,
  _frameCount: 0,
  _lastFpsTime: 0,
  _width: 640,
  _height: 480,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-depth';
    this._el.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;justify-content:center;position:relative;background:#000';
    this._canvas = document.createElement('canvas');
    this._canvas.width = this._width;
    this._canvas.height = this._height;
    this._canvas.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain';
    this._ctx = this._canvas.getContext('2d');
    this._label = document.createElement('span');
    this._label.style.cssText = 'position:absolute;top:6px;right:8px;font-size:11px;color:#fff;background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:3px';
    this._el.appendChild(this._canvas);
    this._el.appendChild(this._label);
    container.appendChild(this._el);
    this._frameCount = 0;
    this._lastFpsTime = performance.now();
  },

  onData(buffer, hint) {
    if (!this._ctx) return;
    const u16 = new Uint16Array(buffer);
    const w = this._width;
    const h = this._height;

    // Expect w*h uint16 pixels
    if (u16.length < w * h) return;

    const imgData = this._ctx.createImageData(w, h);
    const rgba = imgData.data;

    paintDepth(u16, rgba, w * h);

    this._ctx.putImageData(imgData, 0, 0);

    // FPS counter
    this._frameCount++;
    const now = performance.now();
    if (now - this._lastFpsTime >= 1000) {
      this._fps = this._frameCount;
      this._frameCount = 0;
      this._lastFpsTime = now;
      if (this._label) this._label.textContent = `${this._fps} fps`;
    }
  },

  unmount() {
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._label = null;
  },
};


/**
 * DepthZlibRenderer — renders zlib-compressed 16-bit depth images.
 * Driver publishes CompressedImage with format="16UC1; compressedDepth zlib",
 * data = zlib.compress(raw_uint16_buffer, level=1).
 * Decompresses in browser then paints with the same ramp as DepthRenderer.
 */
export const DepthZlibRenderer = {
  name: 'depth-zlib',
  canRender: (hint) => hint === 'image/depth-zlib',
  _el: null,
  _canvas: null,
  _ctx: null,
  _label: null,
  _fps: 0,
  _frameCount: 0,
  _lastFpsTime: 0,
  _width: 640,
  _height: 480,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-depth';
    this._el.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;justify-content:center;position:relative;background:#000';
    this._canvas = document.createElement('canvas');
    this._canvas.width = this._width;
    this._canvas.height = this._height;
    this._canvas.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain';
    this._ctx = this._canvas.getContext('2d');
    this._label = document.createElement('span');
    this._label.style.cssText = 'position:absolute;top:6px;right:8px;font-size:11px;color:#fff;background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:3px';
    this._el.appendChild(this._canvas);
    this._el.appendChild(this._label);
    container.appendChild(this._el);
    this._frameCount = 0;
    this._lastFpsTime = performance.now();
  },

  async onData(buffer, hint) {
    if (!this._ctx) return;

    // Decompress zlib using native DecompressionStream('deflate') which handles zlib format
    let raw;
    try {
      const ds = new DecompressionStream('deflate');
      const writer = ds.writable.getWriter();
      writer.write(new Uint8Array(buffer));
      writer.close();
      const reader = ds.readable.getReader();
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      const totalLen = chunks.reduce((s, c) => s + c.length, 0);
      raw = new Uint8Array(totalLen);
      let offset = 0;
      for (const c of chunks) { raw.set(c, offset); offset += c.length; }
    } catch (e) {
      console.warn('[depth-zlib] decompress failed:', e);
      return;
    }

    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
    const w = this._width;
    const h = this._height;
    if (u16.length < w * h) return;

    const imgData = this._ctx.createImageData(w, h);
    const rgba = imgData.data;

    paintDepth(u16, rgba, w * h);

    this._ctx.putImageData(imgData, 0, 0);

    this._frameCount++;
    const now = performance.now();
    if (now - this._lastFpsTime >= 1000) {
      this._fps = this._frameCount;
      this._frameCount = 0;
      this._lastFpsTime = now;
      if (this._label) this._label.textContent = `${this._fps} fps`;
    }
  },

  unmount() {
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._label = null;
  },
};