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
      this._el.textContent = formatSoundDirection(JSON.parse(new TextDecoder().decode(buffer)));
    } catch { /* Ignore malformed frames. */ }
  },
  unmount() {
    this._el?.remove();
    this._el = null;
  },
};
