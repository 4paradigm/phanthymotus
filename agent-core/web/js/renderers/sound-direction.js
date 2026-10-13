export function formatSoundDirection(message) {
  if (message?.state !== 1) return '暂无声源';
  return `${message.angle_deg}° · ${message.window_start_us}–${message.window_end_us} µs`;
}

export const SoundDirectionRenderer = {
  name: 'sound-direction',
  canRender: (format) => format === 'sensor/sound-direction-v1',
  mount(container) {
    this._el = document.createElement('div');
    this._el.style.cssText = 'display:grid;place-items:center;width:100%;height:100%;font-size:18px';
    this._el.textContent = '暂无声源';
    container.appendChild(this._el);
  },
  onData(buffer) {
    try {
      const message = JSON.parse(new TextDecoder().decode(buffer));
      this._el.textContent = formatSoundDirection(message);
      if (this._clearTimer) clearTimeout(this._clearTimer);
      this._clearTimer = message?.state === 1 ? setTimeout(() => {
        if (this._el) this._el.textContent = '暂无声源';
        this._clearTimer = null;
      }, 2000) : null;
    } catch { /* Ignore malformed frames. */ }
  },
  unmount() {
    if (this._clearTimer) clearTimeout(this._clearTimer);
    this._clearTimer = null;
    this._el?.remove();
    this._el = null;
  },
};
