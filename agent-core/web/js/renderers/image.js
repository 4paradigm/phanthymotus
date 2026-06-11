/** image.js — Renders a static/updated image */
export const ImageRenderer = {
  name: 'image',
  canRender: (hint) => hint === 'image',
  _el: null,
  _img: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-image';
    this._el.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;justify-content:center';
    this._img = document.createElement('img');
    this._el.appendChild(this._img);
    container.appendChild(this._el);
  },

  onEvent(event) {
    if (!this._img) return;
    if (event.type === 'mcp_result') {
      const result = event.payload?.result;
      if (typeof result === 'string') {
        this._img.src = result.startsWith('data:') ? result : 'data:image/jpeg;base64,' + result;
      }
    }
  },

  unmount() { this._el?.remove(); this._el = null; this._img = null; },
};
