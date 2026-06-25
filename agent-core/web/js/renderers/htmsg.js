/**
 * htmsg.js — HTMSG Scene Graph renderer.
 *
 * Visualizes the hierarchical topological-metric semantic graph:
 * - Keyframe nodes (pose graph topology)
 * - Sequential edges between keyframes
 * - Loop closure edges (Phase 2, dashed)
 * - Semantic object nodes (Phase 3, labeled)
 * - Robot current position
 *
 * Data format (JSON via sensor/htmsg):
 * {
 *   type: "htmsg_graph",
 *   robot: {x, y, z, qw, qx, qy, qz},
 *   keyframes: [{id, x, y, z, ts}, ...],
 *   edges: [{from, to, type?}, ...],
 *   semantic_nodes: [{id, label, x, y, z, confidence}, ...]
 * }
 */
export const HTMSGRenderer = {
  name: 'htmsg',
  canRender: (hint) => hint === 'sensor/htmsg',

  _el: null,
  _canvas: null,
  _ctx: null,
  _ro: null,
  _data: null,

  // View state
  _scale: 40,        // pixels per metre
  _offsetX: 0,       // pan offset in pixels
  _offsetY: 0,
  _dragging: false,
  _dragStart: null,
  _followRobot: true,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-htmsg';
    this._el.style.cssText = 'width:100%;height:100%;position:relative;overflow:hidden';

    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'width:100%;height:100%;display:block';
    this._el.appendChild(this._canvas);

    // Follow button
    const btn = document.createElement('button');
    btn.textContent = '⊙ Follow';
    btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 10px;' +
      'background:rgba(40,40,40,0.85);color:#ccc;border:1px solid #555;border-radius:4px;' +
      'font-size:12px;cursor:pointer;z-index:2';
    btn.addEventListener('click', () => {
      this._followRobot = !this._followRobot;
      btn.style.borderColor = this._followRobot ? '#4D9EE8' : '#555';
      this._draw();
    });
    btn.style.borderColor = '#4D9EE8';
    this._el.appendChild(btn);
    this._btn = btn;

    container.appendChild(this._el);

    // Interaction
    this._canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.85 : 1.18;
      this._scale *= factor;
      this._scale = Math.max(5, Math.min(200, this._scale));
      this._draw();
    });
    this._canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this._dragging = true;
        this._dragStart = { x: e.clientX, y: e.clientY, ox: this._offsetX, oy: this._offsetY };
        this._followRobot = false;
        this._btn.style.borderColor = '#555';
      }
    });
    this._canvas.addEventListener('mousemove', (e) => {
      if (this._dragging && this._dragStart) {
        this._offsetX = this._dragStart.ox + (e.clientX - this._dragStart.x);
        this._offsetY = this._dragStart.oy + (e.clientY - this._dragStart.y);
        this._draw();
      }
    });
    this._canvas.addEventListener('mouseup', () => { this._dragging = false; });
    this._canvas.addEventListener('mouseleave', () => { this._dragging = false; });

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);
    this._resize();
  },

  _resize() {
    if (!this._canvas || !this._el) return;
    this._canvas.width = this._el.clientWidth || 400;
    this._canvas.height = this._el.clientHeight || 400;
    this._ctx = this._canvas.getContext('2d');
    this._draw();
  },

  onData(buffer, fmt) {
    try {
      const text = new TextDecoder().decode(buffer);
      const json = JSON.parse(text);
      if (json.type === 'meta' || json.type === 'ping') return;
      this._data = json;
      this._draw();
    } catch (e) { /* ignore parse errors */ }
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  _draw() {
    const c = this._ctx;
    if (!c) return;
    const W = this._canvas.width, H = this._canvas.height;
    const data = this._data;

    c.clearRect(0, 0, W, H);
    c.fillStyle = '#1A1A1F';
    c.fillRect(0, 0, W, H);

    if (!data) {
      c.fillStyle = '#666';
      c.font = '14px sans-serif';
      c.textAlign = 'center';
      c.fillText('Waiting for HTMSG data...', W / 2, H / 2);
      return;
    }

    const scale = this._scale;
    let cx = W / 2 + this._offsetX;
    let cy = H / 2 + this._offsetY;

    // Follow robot
    if (this._followRobot && data.robot) {
      cx = W / 2 - data.robot.x * scale;
      cy = H / 2 + data.robot.y * scale;
    }

    // Helper: world to screen
    const toScreen = (x, y) => [cx + x * scale, cy - y * scale];

    // Grid
    c.strokeStyle = 'rgba(255,255,255,0.05)';
    c.lineWidth = 1;
    const gridStep = scale >= 20 ? 1 : scale >= 10 ? 2 : 5;
    const gridPx = gridStep * scale;
    const startX = (cx % gridPx);
    const startY = (cy % gridPx);
    for (let x = startX; x < W; x += gridPx) {
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
    }
    for (let y = startY; y < H; y += gridPx) {
      c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke();
    }

    // Draw edges
    const keyframes = data.keyframes || [];
    const edges = data.edges || [];
    const kfMap = {};
    keyframes.forEach(kf => { kfMap[kf.id] = kf; });

    c.lineWidth = 1.5;
    edges.forEach(edge => {
      const a = kfMap[edge.from];
      const b = kfMap[edge.to];
      if (!a || !b) return;
      const [ax, ay] = toScreen(a.x, a.y);
      const [bx, by] = toScreen(b.x, b.y);

      if (edge.type === 'loop') {
        c.strokeStyle = 'rgba(255, 180, 50, 0.7)';
        c.setLineDash([4, 4]);
      } else {
        c.strokeStyle = 'rgba(100, 160, 255, 0.5)';
        c.setLineDash([]);
      }
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
    });
    c.setLineDash([]);

    // Draw keyframe nodes
    keyframes.forEach((kf, i) => {
      const [sx, sy] = toScreen(kf.x, kf.y);
      // Skip off-screen
      if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) return;

      const radius = 4;
      // Color by recency (newer = brighter)
      const t = keyframes.length > 1 ? i / (keyframes.length - 1) : 1;
      const r = Math.round(60 + t * 80);
      const g = Math.round(120 + t * 100);
      const b2 = Math.round(200 + t * 55);
      c.fillStyle = `rgb(${r},${g},${b2})`;
      c.beginPath();
      c.arc(sx, sy, radius, 0, Math.PI * 2);
      c.fill();

      // ID label for sparse display
      if (scale > 25 || i % 5 === 0) {
        c.fillStyle = 'rgba(200,200,200,0.6)';
        c.font = '9px sans-serif';
        c.textAlign = 'left';
        c.fillText(`#${kf.id}`, sx + 6, sy - 2);
      }
    });

    // Draw semantic nodes (Phase 3)
    const semanticNodes = data.semantic_nodes || [];
    semanticNodes.forEach(node => {
      const [sx, sy] = toScreen(node.x, node.y);
      if (sx < -30 || sx > W + 30 || sy < -30 || sy > H + 30) return;

      // Diamond shape
      const size = 8;
      c.fillStyle = `rgba(255, 120, 80, ${node.confidence || 0.8})`;
      c.beginPath();
      c.moveTo(sx, sy - size);
      c.lineTo(sx + size, sy);
      c.lineTo(sx, sy + size);
      c.lineTo(sx - size, sy);
      c.closePath();
      c.fill();

      // Label
      if (node.label) {
        c.fillStyle = '#FFA060';
        c.font = '11px sans-serif';
        c.textAlign = 'center';
        c.fillText(node.label, sx, sy - size - 4);
      }
    });

    // Draw robot
    if (data.robot) {
      const [rx, ry] = toScreen(data.robot.x, data.robot.y);
      // Compute yaw from quaternion
      const qw = data.robot.qw || 1, qz = data.robot.qz || 0;
      const yaw = Math.atan2(2 * qw * qz, 1 - 2 * qz * qz);

      c.save();
      c.translate(rx, ry);
      c.rotate(-yaw);  // Canvas Y is flipped

      // Triangle pointing forward
      c.fillStyle = '#4DE880';
      c.beginPath();
      c.moveTo(10, 0);
      c.lineTo(-6, 7);
      c.lineTo(-6, -7);
      c.closePath();
      c.fill();

      // Ring
      c.strokeStyle = '#4DE880';
      c.lineWidth = 2;
      c.beginPath();
      c.arc(0, 0, 12, 0, Math.PI * 2);
      c.stroke();

      c.restore();
    }

    // Legend
    c.fillStyle = '#888';
    c.font = '11px sans-serif';
    c.textAlign = 'left';
    const legendY = H - 12;
    c.fillText(`KF: ${keyframes.length}`, 10, legendY);
    if (semanticNodes.length > 0) {
      c.fillText(`Objects: ${semanticNodes.length}`, 80, legendY);
    }
    if (scale) {
      c.fillText(`${gridStep}m`, W - 40, legendY);
    }
  },

  unmount() {
    this._ro?.disconnect();
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._data = null;
  },
};
