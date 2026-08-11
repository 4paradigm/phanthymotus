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
