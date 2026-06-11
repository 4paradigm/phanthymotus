/** camera.js — Renders image/jpeg (live JPEG stream) and image/depth-z16 (depth colormap) */

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

    // Colorize: 0=black, max_range(5m=5000mm)=bright
    const maxDist = 5000;
    for (let i = 0; i < w * h; i++) {
      const d = u16[i];
      const norm = Math.min(d / maxDist, 1.0);
      // Turbo-ish colormap: blue(near) → green → yellow → red(far)
      const r = Math.min(255, Math.max(0, (norm < 0.5 ? 0 : (norm - 0.5) * 2 * 255) | 0));
      const g = Math.min(255, Math.max(0, (norm < 0.5 ? norm * 2 * 255 : (1 - norm) * 2 * 255) | 0));
      const b = Math.min(255, Math.max(0, (norm < 0.5 ? (1 - norm * 2) * 255 : 0) | 0));
      const idx = i * 4;
      rgba[idx]     = r;
      rgba[idx + 1] = g;
      rgba[idx + 2] = b;
      rgba[idx + 3] = d === 0 ? 0 : 255; // transparent for no-data
    }

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
