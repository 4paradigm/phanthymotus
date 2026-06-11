/**
 * mapping.js — 2D top-down SLAM mapping renderer.
 * Expects binary WebSocket data:
 *   [float32 robot_x, robot_y, robot_yaw] (12 bytes)
 *   [uint32 num_points] (4 bytes)
 *   [float32 x, float32 y] × N (8 bytes each)
 *
 * Accumulates points across frames to show the full map being built.
 */
export const MappingRenderer = {
  name: 'mapping',
  canRender: (hint) => hint === 'sensor/mapping',

  _el: null,
  _canvas: null,
  _ctx: null,
  _ro: null,
  _raf: null,

  // Accumulated map points (persisted across frames)
  _mapPoints: null,    // Float32Array [x0, y0, x1, y1, ...]
  _mapCount: 0,
  _mapCapacity: 200000,  // max accumulated points

  // Robot pose
  _robotX: 0,
  _robotY: 0,
  _robotYaw: 0,

  // POI tags (from pos_tag messages if available)
  _tags: [],

  // View transform
  _scale: 40,       // pixels per meter
  _offsetX: 0,      // pan offset in pixels
  _offsetY: 0,
  _isDragging: false,
  _dragStart: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-mapping';
    this._el.style.cssText = 'width:100%;height:100%;position:relative;overflow:hidden';

    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'width:100%;height:100%;display:block;cursor:grab';
    this._el.appendChild(this._canvas);
    container.appendChild(this._el);

    // Allocate map buffer
    this._mapPoints = new Float32Array(this._mapCapacity * 2);
    this._mapCount = 0;

    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);

    // Mouse pan
    this._canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    this._canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this._canvas.addEventListener('mouseup', () => this._onMouseUp());
    this._canvas.addEventListener('mouseleave', () => this._onMouseUp());
    // Wheel zoom
    this._canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });

    this._draw();
  },

  _resize() {
    if (!this._canvas) return;
    this._canvas.width = this._el.clientWidth || 400;
    this._canvas.height = this._el.clientHeight || 400;
    this._ctx = this._canvas.getContext('2d');
    this._draw();
  },

  _onMouseDown(e) {
    this._isDragging = true;
    this._dragStart = { x: e.clientX - this._offsetX, y: e.clientY - this._offsetY };
    this._canvas.style.cursor = 'grabbing';
  },

  _onMouseMove(e) {
    if (!this._isDragging || !this._dragStart) return;
    this._offsetX = e.clientX - this._dragStart.x;
    this._offsetY = e.clientY - this._dragStart.y;
    this._draw();
  },

  _onMouseUp() {
    this._isDragging = false;
    this._dragStart = null;
    if (this._canvas) this._canvas.style.cursor = 'grab';
  },

  _onWheel(e) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    this._scale = Math.max(5, Math.min(200, this._scale * factor));
    this._draw();
  },

  _draw() {
    const c = this._ctx;
    if (!c) return;
    const W = this._canvas.width, H = this._canvas.height;
    c.clearRect(0, 0, W, H);

    // Background
    c.fillStyle = '#1C1C1E';
    c.fillRect(0, 0, W, H);

    const cx = W / 2 + this._offsetX;
    const cy = H / 2 + this._offsetY;
    const s = this._scale;

    // Grid
    c.strokeStyle = 'rgba(255,255,255,0.05)';
    c.lineWidth = 1;
    const gridStep = s; // 1m grid
    for (let gx = cx % gridStep; gx < W; gx += gridStep) {
      c.beginPath(); c.moveTo(gx, 0); c.lineTo(gx, H); c.stroke();
    }
    for (let gy = cy % gridStep; gy < H; gy += gridStep) {
      c.beginPath(); c.moveTo(0, gy); c.lineTo(W, gy); c.stroke();
    }

    // Map points (accumulated)
    if (this._mapCount > 0) {
      c.fillStyle = 'rgba(120, 180, 255, 0.6)';
      const pts = this._mapPoints;
      const drawCount = Math.min(this._mapCount, this._mapCapacity);
      const ptSize = Math.max(2, Math.round(s / 30));
      const ptHalf = ptSize / 2;
      for (let i = 0; i < drawCount; i++) {
        const px = cx + pts[i * 2] * s;
        const py = cy - pts[i * 2 + 1] * s;
        c.fillRect(px - ptHalf, py - ptHalf, ptSize, ptSize);
      }
    }

    // POI tags
    c.font = '11px system-ui';
    c.textAlign = 'center';
    for (const tag of this._tags) {
      const tx = cx + tag.x * s;
      const ty = cy - tag.y * s;
      // Marker
      c.fillStyle = '#F5A623';
      c.beginPath(); c.arc(tx, ty, 5, 0, Math.PI * 2); c.fill();
      // Label
      c.fillStyle = '#F5A623';
      c.fillText(tag.name, tx, ty - 9);
    }

    // Robot
    const rx = cx + this._robotX * s;
    const ry = cy - this._robotY * s;

    c.save();
    c.translate(rx, ry);
    c.rotate(-this._robotYaw);

    // Robot body (triangle pointing forward)
    c.fillStyle = '#4DDB6A';
    c.beginPath();
    c.moveTo(8, 0);
    c.lineTo(-5, -5);
    c.lineTo(-5, 5);
    c.closePath();
    c.fill();

    // Direction line
    c.strokeStyle = '#4DDB6A';
    c.lineWidth = 2;
    c.beginPath();
    c.moveTo(0, 0);
    c.lineTo(12, 0);
    c.stroke();

    c.restore();

    // Info overlay
    c.fillStyle = 'rgba(255,255,255,0.5)';
    c.font = '10px monospace';
    c.textAlign = 'left';
    c.fillText(`pts: ${this._mapCount}  scale: ${s.toFixed(0)}px/m`, 8, H - 8);
    c.fillText(`robot: (${this._robotX.toFixed(2)}, ${this._robotY.toFixed(2)}) yaw=${(this._robotYaw * 180 / Math.PI).toFixed(1)}°`, 8, H - 22);
  },

  onData(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 16) return;

    const view = new DataView(buffer);
    // Header: robot pose
    this._robotX = view.getFloat32(0, true);
    this._robotY = view.getFloat32(4, true);
    this._robotYaw = view.getFloat32(8, true);

    // Point count
    const numPoints = view.getUint32(12, true);
    const expected = 16 + numPoints * 8;
    if (buffer.byteLength < expected) return;

    // Merge new points into accumulated map (ring buffer)
    for (let i = 0; i < numPoints; i++) {
      const off = 16 + i * 8;
      const x = view.getFloat32(off, true);
      const y = view.getFloat32(off + 4, true);

      const idx = this._mapCount < this._mapCapacity
        ? this._mapCount
        : (this._mapCount % this._mapCapacity);
      this._mapPoints[idx * 2] = x;
      this._mapPoints[idx * 2 + 1] = y;
      this._mapCount++;
    }

    this._draw();
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  unmount() {
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._mapPoints = null;
  },
};
