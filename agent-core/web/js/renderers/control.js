/**
 * control.js — motus.control/1 and session-bound motus.control/2 streams.
 *
 * Without this, `control/joint` and `control/velocity` fall through to the KV
 * panel, which shows a `values` array as one long line of numbers. On a stream
 * that is the thing actually driving the motors, at 30 Hz, that is unreadable
 * in exactly the situation where somebody is watching it because something is
 * wrong.
 *
 * Three things are drawn, because three different questions get asked of this
 * topic:
 *
 *  - **Per-joint bars against the declared limits.** "Is it near a limit" is
 *    not answerable from a number alone; it needs the range beside it. The
 *    limits come from the descriptor when the producer includes one, and from
 *    the observed extremes otherwise — marked as such, since an inferred range
 *    grows as the arm moves and a bar that never reaches its end may only mean
 *    nothing has gone further yet.
 *
 *  - **Age, not just rate.** `obs_stamp_ms` is the one field that catches a
 *    command generated just now from an old picture, which is the characteristic
 *    failure of remote inference. Shown next to the command's own age so the two
 *    cannot be confused.
 *
 *  - **`source` and `priority`.** A topic with several publishers is legitimate
 *    here (agent-core allows it precisely for this format), so "who is driving
 *    right now" is a real question, and a hand-over is otherwise invisible.
 *
 * Protocol: phanthymotus-driver/README_dev.md § "Continuous Control".
 */

// motus.control/1 `twist`: a body velocity in this order. Mirrors
// common/control/descriptor.py's MODES comment and loco_servo's
// AXIS_NAMES — three places name these, and they must not drift.
const TWIST_AXES = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'];

// `wz` is precise but not self-evident: it is angular velocity **about** the z
// axis, which is vertical — so it is turning on the spot, not "rotating in the z
// direction". A ground robot has exactly three live degrees of freedom, vx, vy
// and wz; the other three describe rising, rolling and pitching, which a chassis
// cannot do, which is why its descriptor pins them to zero.
const TWIST_HINTS = {
  vx: '前后（+ 前进）', vy: '左右平移（+ 左）', vz: '上下（底盘无此自由度）',
  wx: '翻滚（底盘无此自由度）', wy: '俯仰（底盘无此自由度）',
  wz: '转向，绕竖直轴（− 顺时针/右转）',
};

const BAR_COLOUR   = '#4D9EE8';
const WARN_COLOUR  = '#D97757';
// Within this fraction of a limit the bar turns warm. Not a threshold the sink
// acts on — the driver rejects at the limit itself — just the point at which a
// human watching should notice.
const NEAR_LIMIT   = 0.9;
// Beyond this the command is older than anything the receiver will execute at a
// typical ttl. Advisory only: the real ttl is per-message and enforced by the
// driver, not here.
const STALE_AGE_MS = 200;

export const ControlRenderer = {
  name: 'control',
  canRender: (hint) => !!hint && hint.startsWith('control/'),

  _el: null,
  _head: null,
  _rows: null,
  _bars: [],          // { row, fill, value, name }
  _names: null,
  _limits: null,      // { lower: [], upper: [], declared: bool }
  _seen: null,        // observed extremes, when no descriptor arrives
  _lastSeq: null,
  _gap: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-control';
    this._el.style.cssText =
      'width:100%;height:100%;overflow:auto;padding:10px 12px;' +
      'font-family:var(--font-mono,ui-monospace,monospace);font-size:12px';

    this._head = document.createElement('div');
    this._head.style.cssText =
      'display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;' +
      'color:var(--text-dim,#888);line-height:1.7';
    this._el.appendChild(this._head);

    this._rows = document.createElement('div');
    this._el.appendChild(this._rows);

    container.appendChild(this._el);
    this._bars = [];
    this._names = null;
    this._limits = null;
    this._seen = null;
    this._lastSeq = null;
    this._gap = null;
    this._setHead([['等待指令', '']]);
  },

  onData(buffer) {
    if (!this._el) return;
    let msg;
    try {
      msg = JSON.parse(new TextDecoder().decode(buffer));
    } catch {
      return;                                    // not our payload
    }
    if (msg.type === 'ping' || msg.type === 'meta') return;
    if (!Array.isArray(msg.values)) return;
    if (msg.schema === 'motus.control/2') {
      const identity = `${msg.boot_id}:${msg.session_id}:${msg.mapping_epoch}`;
      if (identity !== this._session) { this._lastSeq = null; this._gap = null; }
      this._session = identity;
    }

    this._absorbDescriptor(msg);
    this._ensureRows(msg.values.length);
    this._paint(msg);
  },

  // Some producers echo the descriptor they negotiated with; most send only
  // `dof`. Either way the joint names and limits are optional decoration, and
  // the renderer has to stay useful without them.
  _absorbDescriptor(msg) {
    const d = msg.control_interface;
    if (d && Array.isArray(d.joint_names)) this._names = d.joint_names;
    // `twist` is not a joint space: the six values are a body velocity, and
    // labelling them joint1..joint6 tells the reader the robot has six joints
    // it is driving. On a navigating chassis the only non-zero entry sat next
    // to "joint6", which is the yaw rate — the one name that makes the panel
    // readable is the one it was not using. Only when the producer has not
    // named them itself.
    else if (msg.mode === 'twist') this._names = TWIST_AXES;
    if (d && d.limits && Array.isArray(d.limits.lower) && Array.isArray(d.limits.upper)) {
      this._limits = { lower: d.limits.lower, upper: d.limits.upper, declared: true };
    }
    if (this._limits && this._limits.declared) return;

    // No declaration: track the extremes actually seen. Honest but weaker — a
    // bar that never fills may only mean nothing has gone further yet, which
    // is why the header says where the range came from.
    if (!this._seen) this._seen = { lower: [], upper: [] };
    msg.values.forEach((v, i) => {
      const lo = this._seen.lower[i];
      const hi = this._seen.upper[i];
      this._seen.lower[i] = lo === undefined ? v : Math.min(lo, v);
      this._seen.upper[i] = hi === undefined ? v : Math.max(hi, v);
    });
    this._limits = { lower: this._seen.lower, upper: this._seen.upper, declared: false };
  },

  _ensureRows(count) {
    while (this._bars.length > count) this._bars.pop().row.remove();
    while (this._bars.length < count) {
      const i = this._bars.length;
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:3px 0';

      const name = document.createElement('div');
      name.style.cssText =
        'width:96px;flex:none;color:var(--text-dim,#888);overflow:hidden;' +
        'text-overflow:ellipsis;white-space:nowrap';

      const track = document.createElement('div');
      track.style.cssText =
        'flex:1;height:10px;background:rgba(255,255,255,0.06);border-radius:5px;' +
        'position:relative;overflow:hidden';

      // The zero tick. It used to be rgba(255,255,255,0.18) — white, and so
      // **entirely invisible** on a light theme, which left a bar anchored at
      // zero looking like a block floating in space. Follows the theme now.
      //
      // Its position cannot be hardcoded at 50% either: that is only where zero
      // sits when the range is symmetric. In a [0, 1] range zero is at the left
      // edge. The real position is computed per frame in _paint.
      const zero = document.createElement('div');
      zero.style.cssText =
        'position:absolute;left:50%;top:0;bottom:0;width:1px;' +
        'background:var(--text-dim, rgba(0,0,0,0.35))';
      track.appendChild(zero);

      const fill = document.createElement('div');
      fill.style.cssText =
        'position:absolute;top:0;bottom:0;background:' + BAR_COLOUR + ';border-radius:5px';
      track.appendChild(fill);

      const value = document.createElement('div');
      value.style.cssText = 'width:78px;flex:none;text-align:right';

      row.append(name, track, value);
      this._rows.appendChild(row);
      this._bars.push({ row, fill, value, name, zero, idx: i });
    }
  },

  _paint(msg) {
    const now = Date.now();
    const lower = (this._limits && this._limits.lower) || [];
    const upper = (this._limits && this._limits.upper) || [];

    msg.values.forEach((v, i) => {
      const bar = this._bars[i];
      if (!bar) return;
      const label = (this._names && this._names[i]) || (msg.mode === 'eef_pose'
        ? `末端${Math.floor(i / 7) + 1}.${['x','y','z','qx','qy','qz','qw'][i % 7]}` : `joint${i + 1}`);
      bar.name.textContent = label;
      bar.name.title = TWIST_HINTS[label] || '';
      bar.value.textContent = Number(v).toFixed(3);

      const lo = lower[i];
      const hi = upper[i];
      // A bar needs both ends and a non-zero span; an inferred range starts
      // with neither, so the first frame draws no fill rather than a full one.
      //
      // With an inferred range this degenerate case lasts a long time: an axis
      // that has only ever held one value has lo === hi, so a plainly non-zero
      // 0.4 sits beside an empty track. Drawing nothing is honest — without a
      // range there is no proportion to draw — but it has to read as "no range
      // yet" rather than as "the value is zero".
      if (lo === undefined || hi === undefined || hi === lo) {
        bar.fill.style.width = '0';
        bar.row.title = (v === 0)
          ? ''
          : '尚无量程：这个轴目前只出现过一个取值，按已见极值推断不出范围';
        return;
      }
      bar.row.title = '';
      // The bar is anchored at **zero**, not at the left edge. The tick in the
      // track is zero, and drawing from the left contradicts it: with wz in
      // [-2, 2] a value of 0 computes frac = 0.5, so a value that is plainly
      // zero renders as a half-full bar. That is how it was reported from the
      // robot.
      const span = hi - lo;
      const clamp = (x) => Math.max(0, Math.min(1, x));
      const zeroFrac = clamp((0 - lo) / span);          // 零位在量程里的位置
      const valFrac  = clamp((v - lo) / span);
      const left  = Math.min(zeroFrac, valFrac);
      const width = Math.abs(valFrac - zeroFrac);
      if (bar.zero) bar.zero.style.left = (zeroFrac * 100).toFixed(2) + '%';
      bar.fill.style.left  = (left * 100).toFixed(2) + '%';
      bar.fill.style.width = (width * 100).toFixed(2) + '%';
      // Warn only near the ends of the range, keyed on where the value sits in
      // it rather than on the bar's length — for a zero-anchored bar, length
      // says how far from zero, not how close to a limit.
      const near = valFrac >= NEAR_LIMIT || valFrac <= 1 - NEAR_LIMIT;
      bar.fill.style.background = near ? WARN_COLOUR : BAR_COLOUR;
    });

    const age    = Number.isFinite(msg.stamp_ms) ? now - msg.stamp_ms : null;
    const obsAge = Number.isFinite(msg.obs_stamp_ms) ? now - msg.obs_stamp_ms : null;
    // A gap in `seq` is the visible half of what the receiver drops silently.
    if (Number.isFinite(msg.seq)) {
      if (this._lastSeq !== null && msg.seq > this._lastSeq + 1) {
        this._gap = msg.seq - this._lastSeq - 1;
      }
      this._lastSeq = msg.seq;
    }

    if (msg.schema === 'motus.control/2') {
      // Browser and robot monotonic clocks have unrelated origins. Never
      // subtract generated_ns from Date.now() or present a guessed packet age.
      this._setHead([
        ['会话', String(msg.session_id || '?')], ['mode', msg.mode || '?'],
        ['时效', '机器人单调时钟；见 Driver 反馈'], ['坐标系', msg.frame || '?'],
        ['seq / 输入', `${msg.seq ?? '?'} / ${msg.source_seq ?? '?'}`],
        ['映射代次', String(msg.mapping_epoch ?? '?')],
        ...(this._gap ? [['跳过序号', String(this._gap)]] : []),
        ['范围', this._limits && this._limits.declared ? '来自 descriptor' : '按已见极值推断'],
      ]);
      return;
    }
    this._setHead([
      ['source', `${msg.source || '?'}${Number.isFinite(msg.priority) ? ` (p${msg.priority})` : ''}`],
      ['mode', msg.mode || '?'],
      ['指令年龄', age === null ? '?' : `${age} ms`, age !== null && age > STALE_AGE_MS],
      // The field that separates "sent late" from "computed from an old
      // picture" — the characteristic failure of remote inference.
      ['观测年龄', obsAge === null ? '—' : `${obsAge} ms`,
       obsAge !== null && obsAge > STALE_AGE_MS * 2],
      ['seq', Number.isFinite(msg.seq) ? String(msg.seq) : '?'],
      ...(this._gap ? [['丢帧', `${this._gap}`, true]] : []),
      ...(msg.chunk && Number.isFinite(msg.chunk.index)
        ? [['chunk', `${msg.chunk.index}/${msg.chunk.size ?? '?'}`]] : []),
      ['范围', this._limits && this._limits.declared ? '来自 descriptor' : '按已见极值推断'],
    ]);
  },

  _setHead(pairs) {
    if (!this._head) return;
    this._head.innerHTML = '';
    for (const [label, value, warn] of pairs) {
      const item = document.createElement('div');
      item.innerHTML = `<span style="opacity:.7">${label}</span> `;
      const strong = document.createElement('span');
      strong.textContent = value;
      strong.style.color = warn ? WARN_COLOUR : 'var(--text,#eee)';
      item.appendChild(strong);
      this._head.appendChild(item);
    }
  },

  unmount() {
    this._el?.remove();
    this._el = null;
    this._head = null;
    this._rows = null;
    this._bars = [];
  },
};
