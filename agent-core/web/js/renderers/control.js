/**
 * control.js — motus.control/1 command streams.
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

// `wz` 精确但不自明：它是**绕 z 轴**（竖直轴）的角速度，也就是原地转向，而不是
// 「在 z 方向上转」。地面机器人恰好只有 vx/vy/wz 三个自由度是活的，其余三个描述
// 的是上下、侧翻、俯仰 —— 底盘做不到，所以 descriptor 把它们钉死为 0。
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

      // 零位刻度。原先是 rgba(255,255,255,0.18) —— 白色，在浅色主题上**完全看
      // 不见**，于是以零为锚的条看起来像凭空浮着一块。用边框色跟着主题走。
      //
      // 位置也不能写死 50%：那只在量程对称时才是零位。[0, 1] 的量程里零在最左。
      // 真实位置每帧在 _paint 里算。
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
      const label = (this._names && this._names[i]) || `joint${i + 1}`;
      bar.name.textContent = label;
      if (TWIST_HINTS[label]) bar.name.title = TWIST_HINTS[label];
      bar.value.textContent = Number(v).toFixed(3);

      const lo = lower[i];
      const hi = upper[i];
      // A bar needs both ends and a non-zero span; an inferred range starts
      // with neither, so the first frame draws no fill rather than a full one.
      //
      // 推断量程下这个退化情况会持续很久：一个轴只出现过一个取值时 lo === hi，
      // 于是 0.4 这样明显非零的数字旁边一直是空的。画不出条是诚实的（没有量程
      // 就没有比例可言），但要让人看出来是「还没有量程」而不是「值为零」。
      if (lo === undefined || hi === undefined || hi === lo) {
        bar.fill.style.width = '0';
        bar.row.title = (v === 0)
          ? ''
          : '尚无量程：这个轴目前只出现过一个取值，按已见极值推断不出范围';
        return;
      }
      bar.row.title = '';
      // 条形以**零位**为锚，不是以左边缘 —— 轨道中间那条刻度线就是零位，而
      // 从左边缘画会和它矛盾：wz 的量程是 [-2, 2] 时，v=0 算出 frac=0.5，于是
      // 一个明明是零的值显示成半条 bar。真机上就是这样报上来的。
      const span = hi - lo;
      const clamp = (x) => Math.max(0, Math.min(1, x));
      const zeroFrac = clamp((0 - lo) / span);          // 零位在量程里的位置
      const valFrac  = clamp((v - lo) / span);
      const left  = Math.min(zeroFrac, valFrac);
      const width = Math.abs(valFrac - zeroFrac);
      if (bar.zero) bar.zero.style.left = (zeroFrac * 100).toFixed(2) + '%';
      bar.fill.style.left  = (left * 100).toFixed(2) + '%';
      bar.fill.style.width = (width * 100).toFixed(2) + '%';
      // 接近量程两端才示警，按的是值在量程里的位置而不是条的长度 —— 一个以零
      // 为锚的条，长度说的是离零多远，不是离限位多近。
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
