/** Read-only display of motus.odom/1; no pose is inferred from velocity. */
export const OdomRenderer = {
  name: 'odom',
  canRender: hint => hint === 'state/odom',

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-odom';
    this._el.style.cssText = 'padding:10px 12px;white-space:pre-wrap;overflow:auto;font:12px/1.7 var(--font-mono,monospace)';
    this._el.textContent = '等待里程计数据';
    this._lastAt = null;
    this._timer = setInterval(() => {
      if (this._lastAt !== null && performance.now() - this._lastAt > 5000) {
        this._el.textContent = '已暂停：超过 5 秒没有新数据';
      }
    }, 1000);
    container.appendChild(this._el);
  },

  onData(buffer) {
    if (!this._el) return;
    let msg;
    try { msg = JSON.parse(new TextDecoder().decode(buffer)); } catch { return; }
    if (!msg || msg.schema !== 'motus.odom/1') return;
    this._lastAt = performance.now();
    const vendor = msg.vendor || {};
    const meta = vendor.motus_slam || vendor;
    const status = typeof meta.status === 'string' ? meta.status : '未知';
    const pose = status === 'lost' || status === 'initializing' ? null : msg.pose;
    const twist = Array.isArray(msg.twist) ? msg.twist : [];
    const vector = (v, n) => Array.isArray(v) && v.length === n
      ? v.map(x => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(3) : '—').join(', ') : '—';
    const number = (v, digits=3) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
    const text = v => typeof v === 'string' && v ? v : '—';
    this._el.textContent = [
      `状态: ${status}`,
      `分段: ${text(meta.segment_id)}`,
      `位置: ${vector(pose?.position, 3)} m`,
      `本段位移: ${number(pose ? vendor.displacement_m : null)} m`,
      `本段转角: ${number(vendor.turn_angle_deg, 1)} °`,
      `参考系: ${text(vendor.pose_reference)}`,
      `原因: ${text(vendor.reason)}`,
      `姿态 (x, y, z, w): ${vector(pose?.quaternion, 4)}`,
      `坐标系: ${text(pose?.frame_id)} → ${text(pose?.child_frame_id)}`,
      `速度 (vx, vy, vz): ${vector(twist.slice(0, 3), 3)} m/s`,
      `角速度 (wx, wy, wz): ${vector(twist.slice(3, 6), 3)} rad/s`,
      `采样时间: ${Number.isSafeInteger(msg.stamp_ms) ? msg.stamp_ms + ' ms' : '—'}`,
      `时钟: ${text(meta.clock_id)}；测量年龄: 未知`,
      `协方差: ${Array.isArray(pose?.covariance) && pose.covariance.length === 36 ? '已提供' : '未知'}`,
    ].join('\n');
  },

  onEvent() {},

  onDataSilent(buffer) { this.onData(buffer); },

  unmount() {
    clearInterval(this._timer);
    this._el?.remove();
    this._el = null;
  },
};

/** Prefer an advertised odometry output for preview without reordering ports. */
export function odomPreview(skill) {
  const topic = (skill.topic_out || []).find(t => t.format === 'state/odom' && t.topic);
  return topic
    ? {path: '/ws/bus' + topic.topic, hint: 'state/odom', topic}
    : {path: skill.ws_path, hint: skill.render_hint || 'activity', topic: skill.topic_out?.[0] || null};
}
