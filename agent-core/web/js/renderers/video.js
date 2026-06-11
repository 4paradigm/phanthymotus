/** video.js — Renders MJPEG stream or base64 frame updates */
export const VideoRenderer = {
  name: 'video',
  canRender: (hint) => hint && hint.startsWith('video/'),
  _el: null,
  _img: null,

  mount(container, mcpId) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-video';
    this._img = document.createElement('img');
    this._img.className = 'mjpeg';
    this._img.alt = 'video stream';
    this._el.appendChild(this._img);
    container.appendChild(this._el);
  },

  onEvent(event) {
    if (!this._img) return;
    if (event.type === 'render' && event.payload?.url) {
      // MJPEG URL
      this._img.src = event.payload.url;
    }
    if (event.type === 'mcp_result') {
      const result = event.payload?.result;
      // base64 JPEG frame
      if (typeof result === 'string' && result.startsWith('/9j/')) {
        this._img.src = 'data:image/jpeg;base64,' + result;
      } else if (typeof result === 'string' && result.startsWith('data:')) {
        this._img.src = result;
      }
    }
  },

  unmount() { this._el?.remove(); this._el = null; this._img = null; },
};
