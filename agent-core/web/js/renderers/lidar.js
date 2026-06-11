/**
 * lidar.js — Simple LiDAR / point-cloud top-down view renderer.
 * Expects mcp_result payload.result to be an array of {x, y} or {x, y, z} points
 * in robot-frame coordinates (metres).
 */
export const LidarRenderer = {
  name: 'lidar',
  canRender: (hint) => hint && hint.startsWith('sensor/lidar'),
  _el: null,
  _canvas: null,
  _ctx: null,
  _raf: null,
  _points: [],
  _scale: 60,   // pixels per metre

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-lidar';
    this._el.style.cssText = 'width:100%;height:100%;position:relative';

    this._canvas = document.createElement('canvas');
    this._el.appendChild(this._canvas);
    container.appendChild(this._el);

    this._resize();
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(this._el);
    this._ro = ro;
    this._draw();
  },

  _resize() {
    if (!this._canvas) return;
    this._canvas.width  = this._el.clientWidth  || 400;
    this._canvas.height = this._el.clientHeight || 400;
    this._ctx = this._canvas.getContext('2d');
    this._draw();
  },

  _draw() {
    const c = this._ctx;
    if (!c) return;
    const W = this._canvas.width, H = this._canvas.height;
    c.clearRect(0, 0, W, H);

    c.fillStyle = '#1C1C1E';
    c.fillRect(0, 0, W, H);

    // Grid
    c.strokeStyle = 'rgba(255,255,255,0.06)';
    c.lineWidth = 1;
    const cx = W / 2, cy = H / 2;
    for (let d = this._scale; d < Math.max(W, H); d += this._scale) {
      c.beginPath(); c.arc(cx, cy, d, 0, Math.PI * 2); c.stroke();
    }
    c.beginPath(); c.moveTo(cx, 0); c.lineTo(cx, H); c.stroke();
    c.beginPath(); c.moveTo(0, cy); c.lineTo(W, cy); c.stroke();

    // Robot
    c.fillStyle = '#4D9EE8';
    c.beginPath(); c.arc(cx, cy, 6, 0, Math.PI * 2); c.fill();

    // Points
    c.fillStyle = '#D97757';
    this._points.forEach(({ x, y }) => {
      const px = cx + x * this._scale;
      const py = cy - y * this._scale;
      c.fillRect(px - 1.5, py - 1.5, 3, 3);
    });
  },

  onEvent(event) {
    if (event.type !== 'mcp_result') return;
    const result = event.payload?.result;
    if (Array.isArray(result)) {
      this._points = result.slice(0, 2000);  // cap for perf
      this._draw();
    }
  },

  unmount() {
    this._ro?.disconnect();
    cancelAnimationFrame(this._raf);
    this._el?.remove();
    this._el = null; this._canvas = null; this._ctx = null;
  },
};
