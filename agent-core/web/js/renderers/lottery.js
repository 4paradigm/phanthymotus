/** Server-authoritative renderer for `data/lottery`. */

export function parseLotteryMessage(payload) {
  try {
    const text = typeof payload === 'string'
      ? payload
      : new TextDecoder().decode(payload);
    const value = JSON.parse(text);
    return value && typeof value === 'object' ? value : null;
  } catch {
    return null;
  }
}

// Deterministic decoration: displays given the same seed show the same rolling
// value at the same tick. It never decides the winner; only a result frame does.
export function lotteryFrameIndex(seed, tick, length) {
  if (!Number.isInteger(length) || length <= 0) return -1;
  let value = ((Number(seed) >>> 0) + Math.imul((tick | 0) + 1, 0x9e3779b1)) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x21f0aaad) >>> 0;
  value = (value ^ (value >>> 15)) >>> 0;
  return value % length;
}

const DICE = { '1': '⚀', '2': '⚁', '3': '⚂', '4': '⚃', '5': '⚄', '6': '⚅' };

function displayValue(mode, value) {
  const text = String(value ?? '');
  return mode === 'dice' ? (DICE[text] || text) : text;
}

export const LotteryRenderer = {
  name: 'lottery',
  canRender: hint => hint === 'data/lottery',

  mount(container) {
    this._timer = null;
    this._message = null;
    this._el = document.createElement('section');
    this._el.className = 'renderer-lottery';

    const toolbar = document.createElement('div');
    toolbar.className = 'lottery-toolbar';
    const status = document.createElement('span');
    status.className = 'lottery-status';
    status.textContent = '等待抽奖';
    const fullscreen = document.createElement('button');
    fullscreen.type = 'button';
    fullscreen.className = 'lottery-fullscreen';
    fullscreen.textContent = '⛶ 全屏展示';
    fullscreen.addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen?.();
      else this._el?.requestFullscreen?.();
    });
    toolbar.append(status, fullscreen);

    const stage = document.createElement('div');
    stage.className = 'lottery-stage';
    const value = document.createElement('div');
    value.className = 'lottery-value';
    value.textContent = '—';
    const caption = document.createElement('div');
    caption.className = 'lottery-caption';
    caption.textContent = '等待开始';
    stage.append(value, caption);
    this._el.append(toolbar, stage);
    container.appendChild(this._el);

    this._status = status;
    this._value = value;
    this._caption = caption;
  },

  onData(payload) {
    const message = parseLotteryMessage(payload);
    if (!message) return;
    this._apply(message);
  },

  onDataSilent(payload) {
    // The bus's cached snapshot is state, not historical animation. Applying it
    // is exactly how a late/fullscreen viewer catches up without a second draw.
    this.onData(payload);
  },

  _apply(message) {
    this._message = message;
    clearInterval(this._timer);
    this._timer = null;
    this._el.dataset.mode = message.mode || 'options';
    this._el.classList.toggle('spinning', message.state === 'spinning');
    this._el.classList.toggle('has-result', message.state === 'result');

    if (message.state === 'spinning') {
      const options = Array.isArray(message.options) ? message.options : [];
      const startedAt = Number(message.started_at) || Date.now();
      const paint = () => {
        const tick = Math.max(0, Math.floor((Date.now() - startedAt) / 80));
        const index = lotteryFrameIndex(message.animation_seed, tick, options.length);
        if (index >= 0) this._value.textContent = displayValue(message.mode, options[index]);
      };
      paint();
      this._timer = setInterval(paint, 80);
      this._status.textContent = '抽奖进行中';
      this._caption.textContent = message.mode === 'card' ? '正在翻牌…' : '正在随机选择…';
      return;
    }

    if (message.state === 'result') {
      // This is the only assignment that can reveal a winner. Never derive it
      // from the animation or result_index on the client.
      this._value.textContent = displayValue(message.mode, message.result);
      this._status.textContent = '抽奖完成';
      this._caption.textContent = '恭喜中奖';
      return;
    }

    this._value.textContent = '—';
    this._status.textContent = '等待抽奖';
    this._caption.textContent = '等待开始';
  },

  unmount() {
    clearInterval(this._timer);
    this._timer = null;
    this._el?.remove();
    this._el = null;
  },
};
