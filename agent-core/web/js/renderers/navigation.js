/** navigation.js — Canvas renderers for canonical ROS navigation data. */

function _decode(buffer) {
  try {
    return JSON.parse(new TextDecoder().decode(buffer));
  } catch {
    return null;
  }
}

function _fmt(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
}

function _escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

export const OdometryRenderer = {
  name: 'odometry',
  canRender: (hint) => hint === 'sensor/odometry',
  _el: null,
  _pose: null,
  _velocity: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.style.cssText =
      'width:100%;height:100%;box-sizing:border-box;padding:14px;' +
      'display:flex;flex-direction:column;justify-content:center;gap:14px;' +
      'background:radial-gradient(circle at center,#15231b 0,#080b09 70%);color:#eef7f0';
    this._pose = document.createElement('div');
    this._velocity = document.createElement('div');
    this._el.append(this._pose, this._velocity);
    container.appendChild(this._el);
  },

  onData(buffer) {
    const data = _decode(buffer);
    if (!data?.position) return;
    const yawDeg = Number(data.yaw) * 180 / Math.PI;
    this._pose.innerHTML =
      `<div style="font-size:11px;opacity:.55;margin-bottom:5px">${_escapeHtml(data.frame_id || '—')} → ${_escapeHtml(data.child_frame_id || '—')}</div>` +
      `<div style="font-size:26px;font-weight:700;line-height:1.3">x ${_fmt(data.position.x)} m</div>` +
      `<div style="font-size:26px;font-weight:700;line-height:1.3">y ${_fmt(data.position.y)} m</div>` +
      `<div style="font-size:20px;color:#69dc83;margin-top:4px">yaw ${_fmt(yawDeg, 1)}°</div>`;
    this._velocity.innerHTML =
      `<div style="font-size:11px;opacity:.55;margin-bottom:5px">VELOCITY</div>` +
      `<div style="font-family:monospace;font-size:13px;line-height:1.7">` +
      `vx ${_fmt(data.linear_velocity?.x)} m/s<br>` +
      `vy ${_fmt(data.linear_velocity?.y)} m/s<br>` +
      `wz ${_fmt(data.angular_velocity?.z)} rad/s</div>`;
  },

  onDataSilent(buffer) { this.onData(buffer); },
  unmount() {
    this._el?.remove();
    this._el = null;
    this._pose = null;
    this._velocity = null;
  },
};

export const PathRenderer = {
  name: 'path',
  canRender: (hint) => hint === 'sensor/path',
  _el: null,
  _canvas: null,
  _ctx: null,
  _summary: null,
  _ro: null,
  _latest: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.style.cssText = 'width:100%;height:100%;position:relative;background:#070908;overflow:hidden';
    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'width:100%;height:100%;display:block';
    this._summary = document.createElement('div');
    this._summary.style.cssText =
      'position:absolute;left:9px;bottom:7px;padding:3px 7px;border-radius:4px;' +
      'background:rgba(0,0,0,.62);color:#dce7df;font:11px monospace';
    this._el.append(this._canvas, this._summary);
    container.appendChild(this._el);
    this._ctx = this._canvas.getContext('2d');
    this._ro = new ResizeObserver(() => this._draw());
    this._ro.observe(this._el);
    this._draw();
  },

  onData(buffer) {
    const data = _decode(buffer);
    if (!Array.isArray(data?.poses)) return;
    this._latest = data;
    this._draw();
  },

  onDataSilent(buffer) { this.onData(buffer); },

  _draw() {
    if (!this._canvas || !this._ctx || !this._el) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(this._el.clientWidth, 1);
    const height = Math.max(this._el.clientHeight, 1);
    this._canvas.width = Math.floor(width * ratio);
    this._canvas.height = Math.floor(height * ratio);
    this._ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const ctx = this._ctx;
    ctx.fillStyle = '#070908';
    ctx.fillRect(0, 0, width, height);

    const poses = this._latest?.poses || [];
    if (!poses.length) {
      ctx.fillStyle = '#79817c';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('等待 Nav2 生成路径', width / 2, height / 2);
      if (this._summary) this._summary.textContent = '0 poses';
      return;
    }

    let minX = poses[0].x, maxX = poses[0].x;
    let minY = poses[0].y, maxY = poses[0].y;
    let length = 0;
    for (let i = 0; i < poses.length; i++) {
      const p = poses[i];
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      if (i) length += Math.hypot(p.x - poses[i - 1].x, p.y - poses[i - 1].y);
    }
    const pad = 22;
    const spanX = Math.max(maxX - minX, 0.25);
    const spanY = Math.max(maxY - minY, 0.25);
    const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
    const offsetX = (width - spanX * scale) / 2;
    const offsetY = (height - spanY * scale) / 2;
    const project = (p) => ({
      x: offsetX + (p.x - minX) * scale,
      y: height - offsetY - (p.y - minY) * scale,
    });

    ctx.strokeStyle = 'rgba(255,255,255,.08)';
    ctx.lineWidth = 1;
    for (let x = pad; x < width; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
    for (let y = pad; y < height; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }

    ctx.strokeStyle = '#55d675';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    poses.forEach((pose, index) => {
      const p = project(pose);
      if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();

    const start = project(poses[0]);
    const goal = project(poses[poses.length - 1]);
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(start.x, start.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffb347';
    ctx.beginPath(); ctx.arc(goal.x, goal.y, 7, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();

    if (this._summary) {
      this._summary.textContent = `${this._latest.frame_id || '—'}  ${poses.length} poses  ${length.toFixed(2)} m`;
    }
  },

  unmount() {
    this._ro?.disconnect();
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._summary = null;
    this._ro = null;
    this._latest = null;
  },
};

export const CostmapRenderer = {
  name: 'costmap',
  canRender: (hint) => hint === 'sensor/costmap',
  _el: null,
  _canvas: null,
  _ctx: null,
  _summary: null,
  _legend: null,
  _ro: null,
  _latest: null,
  _plan: null,
  _odom: null,
  _planWs: null,
  _odomWs: null,
  _planReconnectTimer: null,
  _odomReconnectTimer: null,

  mount(container) {
    this._latest = null;
    this._plan = null;
    this._odom = null;
    this._el = document.createElement('div');
    this._el.style.cssText =
      'width:100%;height:100%;position:relative;background:#070908;overflow:hidden';
    this._canvas = document.createElement('canvas');
    this._canvas.style.cssText = 'width:100%;height:100%;display:block';
    this._summary = document.createElement('div');
    this._summary.style.cssText =
      'position:absolute;left:9px;bottom:7px;padding:3px 7px;border-radius:4px;' +
      'background:rgba(0,0,0,.68);color:#dce7df;font:11px monospace';
    this._legend = document.createElement('div');
    this._legend.style.cssText =
      'position:absolute;right:9px;bottom:7px;padding:3px 7px;border-radius:4px;' +
      'background:rgba(0,0,0,.68);color:#dce7df;font:10px monospace';
    this._legend.innerHTML =
      '<span style="color:#d9382e">Rejected (&gt;=99)</span>  ' +
      '<span style="color:#55d675">Path</span>';
    this._el.append(this._canvas, this._summary, this._legend);
    container.appendChild(this._el);
    this._ctx = this._canvas.getContext('2d');
    this._ro = new ResizeObserver(() => this._draw());
    this._ro.observe(this._el);
    this._connectAuxStream('plan', '/ws/bus/plan');
    this._connectAuxStream('odom', '/ws/bus/ubuntu/navigation/odom');
    this._draw();
  },

  onData(buffer) {
    const data = _decode(buffer);
    const width = Number(data?.width);
    const height = Number(data?.height);
    const resolution = Number(data?.resolution);
    if (
      data?.frame_id !== 'map'
      || !Number.isInteger(width) || width <= 0
      || !Number.isInteger(height) || height <= 0
      || !Number.isFinite(resolution) || resolution <= 0
      || !Array.isArray(data?.data) || data.data.length < width * height
    ) return;
    this._latest = data;
    this._draw();
  },

  onDataSilent(buffer) { this.onData(buffer); },

  _connectAuxStream(kind, path) {
    if (!this._el || this[`_${kind}Ws`]) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}${path}`);
    ws.binaryType = 'arraybuffer';
    this[`_${kind}Ws`] = ws;
    ws.onmessage = (event) => {
      if (this[`_${kind}Ws`] !== ws) return;
      let data;
      try {
        const text = event.data instanceof ArrayBuffer
          ? new TextDecoder().decode(event.data)
          : String(event.data);
        data = JSON.parse(text);
      } catch {
        return;
      }
      if (data?.type === 'ping' || data?.type === 'meta' || data?.type === 'error') return;
      if (kind === 'plan' && data?.frame_id === 'map' && Array.isArray(data?.poses)) {
        this._plan = data;
      }
      if (
        kind === 'odom' && data?.frame_id === 'map'
        && data?.child_frame_id === 'base_link' && data?.position
      ) {
        this._odom = data;
      }
      this._draw();
    };
    ws.onclose = () => {
      if (this[`_${kind}Ws`] !== ws) return;
      this[`_${kind}Ws`] = null;
      if (!this._el) return;
      const timerName = `_${kind}ReconnectTimer`;
      this[timerName] = setTimeout(() => {
        this[timerName] = null;
        this._connectAuxStream(kind, path);
      }, 5000);
    };
    ws.onerror = () => {};
  },

  _draw() {
    if (!this._canvas || !this._ctx || !this._el) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(this._el.clientWidth, 1);
    const height = Math.max(this._el.clientHeight, 1);
    this._canvas.width = Math.floor(width * ratio);
    this._canvas.height = Math.floor(height * ratio);
    this._ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const ctx = this._ctx;
    ctx.fillStyle = '#070908';
    ctx.fillRect(0, 0, width, height);

    const grid = this._latest;
    if (!grid) {
      ctx.fillStyle = '#79817c';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('等待 Nav2 全局代价地图', width / 2, height / 2);
      if (this._summary) this._summary.textContent = 'COSTMAP  waiting';
      return;
    }

    const gridWidth = Number(grid.width);
    const gridHeight = Number(grid.height);
    const resolution = Number(grid.resolution);
    const originX = Number(grid.origin?.x || 0);
    const originY = Number(grid.origin?.y || 0);
    const originYaw = Number(grid.origin?.yaw || 0);
    const cosYaw = Math.cos(originYaw);
    const sinYaw = Math.sin(originYaw);
    const gridMetersX = gridWidth * resolution;
    const gridMetersY = gridHeight * resolution;
    const worldFromLocal = (x, y) => ({
      x: originX + cosYaw * x - sinYaw * y,
      y: originY + sinYaw * x + cosYaw * y,
    });
    const corners = [
      worldFromLocal(0, 0),
      worldFromLocal(gridMetersX, 0),
      worldFromLocal(0, gridMetersY),
      worldFromLocal(gridMetersX, gridMetersY),
    ];
    const poses = Array.isArray(this._plan?.poses) ? this._plan.poses : [];
    const boundsPoints = [...corners, ...poses];
    if (this._odom?.position) boundsPoints.push(this._odom.position);
    let minX = Math.min(...boundsPoints.map((point) => Number(point.x)));
    let maxX = Math.max(...boundsPoints.map((point) => Number(point.x)));
    let minY = Math.min(...boundsPoints.map((point) => Number(point.y)));
    let maxY = Math.max(...boundsPoints.map((point) => Number(point.y)));
    const pad = 16;
    const spanX = Math.max(maxX - minX, resolution);
    const spanY = Math.max(maxY - minY, resolution);
    const scale = Math.max(
      0.01,
      Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY),
    );
    const offsetX = (width - spanX * scale) / 2;
    const offsetY = (height - spanY * scale) / 2;
    const project = (point) => ({
      x: offsetX + (Number(point.x) - minX) * scale,
      y: height - offsetY - (Number(point.y) - minY) * scale,
    });

    const gridCanvas = document.createElement('canvas');
    gridCanvas.width = gridWidth;
    gridCanvas.height = gridHeight;
    const gridCtx = gridCanvas.getContext('2d');
    const pixels = gridCtx.createImageData(gridWidth, gridHeight);
    for (let row = 0; row < gridHeight; row++) {
      for (let col = 0; col < gridWidth; col++) {
        const value = Number(grid.data[row * gridWidth + col]);
        const pixel = ((gridHeight - 1 - row) * gridWidth + col) * 4;
        let red = 10, green = 13, blue = 12;
        if (value < 0) {
          red = 43; green = 47; blue = 45;
        } else if (value >= 99) {
          red = 217; green = 56; blue = 46;
        }
        pixels.data[pixel] = red;
        pixels.data[pixel + 1] = green;
        pixels.data[pixel + 2] = blue;
        pixels.data[pixel + 3] = 255;
      }
    }
    gridCtx.putImageData(pixels, 0, 0);

    ctx.save();
    ctx.imageSmoothingEnabled = false;
    const cellScale = resolution * scale;
    ctx.transform(
      cosYaw * cellScale,
      -sinYaw * cellScale,
      sinYaw * cellScale,
      cosYaw * cellScale,
      offsetX + (originX - sinYaw * gridMetersY - minX) * scale,
      height - offsetY - (originY + cosYaw * gridMetersY - minY) * scale,
    );
    ctx.drawImage(gridCanvas, 0, 0);
    ctx.restore();

    if (poses.length) {
      ctx.strokeStyle = '#55d675';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      poses.forEach((pose, index) => {
        const point = project(pose);
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
      const goal = project(poses[poses.length - 1]);
      ctx.fillStyle = '#ffb347';
      ctx.beginPath();
      ctx.arc(goal.x, goal.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    if (this._odom?.position) {
      const robot = project(this._odom.position);
      const yaw = Number(this._odom.yaw || 0);
      ctx.save();
      ctx.translate(robot.x, robot.y);
      ctx.rotate(-yaw);
      ctx.fillStyle = '#72e98a';
      ctx.beginPath();
      ctx.moveTo(10, 0);
      ctx.lineTo(-7, -6);
      ctx.lineTo(-4, 0);
      ctx.lineTo(-7, 6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    if (this._summary) {
      this._summary.textContent =
        `COSTMAP  ${gridWidth}×${gridHeight}  ${resolution.toFixed(2)} m/cell  ` +
        `PATH ${poses.length}`;
    }
  },

  unmount() {
    if (this._planReconnectTimer) clearTimeout(this._planReconnectTimer);
    if (this._odomReconnectTimer) clearTimeout(this._odomReconnectTimer);
    this._planReconnectTimer = null;
    this._odomReconnectTimer = null;
    const planWs = this._planWs;
    const odomWs = this._odomWs;
    this._planWs = null;
    this._odomWs = null;
    planWs?.close();
    odomWs?.close();
    this._ro?.disconnect();
    this._el?.remove();
    this._el = null;
    this._canvas = null;
    this._ctx = null;
    this._summary = null;
    this._legend = null;
    this._ro = null;
    this._latest = null;
    this._plan = null;
    this._odom = null;
  },
};
