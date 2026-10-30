/** Five-finger L20 tactile matrix from a sensor/tactile JSON topic. */
const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const STOPS = [
  [0, 26, 34, 52], [38, 14, 116, 144], [89, 16, 185, 129],
  [140, 234, 179, 8], [191, 249, 115, 22], [255, 244, 63, 94],
];

export function tactileColor(value) {
  const n = Math.max(0, Math.min(255, value));
  const upper = STOPS.findIndex(stop => n <= stop[0]);
  if (upper <= 0) return `rgb(${STOPS[0].slice(1).join(',')})`;
  const a = STOPS[upper - 1], b = STOPS[upper];
  const t = (n - a[0]) / (b[0] - a[0]);
  return `rgb(${[1, 2, 3].map(i => Math.round(a[i] + (b[i] - a[i]) * t)).join(',')})`;
}

export function parseTactilePayload(input) {
  let data;
  try {
    data = typeof input === 'string' ? JSON.parse(input)
      : input instanceof ArrayBuffer || ArrayBuffer.isView(input)
        ? JSON.parse(new TextDecoder().decode(input)) : input;
  } catch { return null; }
  if (!data || typeof data !== 'object' || !data.touch_matrix || !data.touch_scalar) return null;
  const matrices = {}, scalars = {};
  for (const finger of FINGERS) {
    const rows = data.matrix_valid === false ? null : data.touch_matrix[finger];
    matrices[finger] = Array.isArray(rows) && rows.length === 12 && rows.every(row =>
      Array.isArray(row) && row.length === 6 && row.every(v => Number.isInteger(v) && v >= 0 && v <= 255))
      ? rows : null;
    const scalar = data.scalar_valid === false ? null : data.touch_scalar[finger];
    scalars[finger] = Number.isInteger(scalar) && scalar >= 0 && scalar <= 255 ? scalar : null;
  }
  return { side: data.side === 'left' || data.side === 'right' ? data.side : 'unknown',
           state: data.state, matrices, scalars };
}

export const TactileRenderer = {
  name: 'tactile',
  canRender: hint => hint === 'sensor/tactile',
  _canvas: null,

  mount(container) {
    this._canvas = document.createElement('canvas');
    this._canvas.width = 660;
    this._canvas.height = 300;
    this._canvas.style.cssText = 'display:block;width:100%;height:100%;object-fit:contain;background:#0f172a';
    container.appendChild(this._canvas);
  },

  onData(buffer) {
    const data = parseTactilePayload(buffer);
    const ctx = this._canvas?.getContext('2d');
    if (!data || !ctx) return;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 660, 300);
    ctx.font = '14px sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`LINKERHAND L20 · TACTILE MATRIX (${data.side.toUpperCase()})`, 20, 24);
    const available = FINGERS.filter(f => data.matrices[f] && data.scalars[f] !== null).length;
    const anyData = FINGERS.some(f => data.matrices[f] || data.scalars[f] !== null);
    const status = !anyData ? 'NO DATA' : available < 5 || data.state !== 'ok' ? 'PARTIAL DATA' : 'LIVE STREAM';
    ctx.fillStyle = status === 'LIVE STREAM' ? '#10b981' : '#fb923c';
    ctx.fillText(status, 540, 24);

    FINGERS.forEach((finger, index) => {
      const x = 24 + index * 108, y = 52, matrix = data.matrices[finger], scalar = data.scalars[finger];
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(finger.toUpperCase(), x, y - 10);
      if (matrix) {
        for (let row = 0; row < 12; row++) for (let col = 0; col < 6; col++) {
          ctx.fillStyle = tactileColor(matrix[row][col]);
          ctx.fillRect(x + col * 14 + 1, y + row * 14 + 1, 13, 13);
        }
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`Peak: ${Math.max(...matrix.flat())}`, x, y + 185);
      } else {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(x, y, 84, 168);
        ctx.fillStyle = '#fb923c';
        ctx.fillText('NO DATA', x + 8, y + 85);
      }
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(x, y + 198, 84, 6);
      if (scalar !== null) {
        ctx.fillStyle = tactileColor(scalar);
        ctx.fillRect(x, y + 198, 84 * scalar / 255, 6);
      }
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`Tip: ${scalar ?? '--'}`, x, y + 218);
    });
    ctx.fillStyle = '#64748b';
    ctx.fillText('12×6 taxels / finger · 0–255 raw values', 20, 294);
  },

  unmount() { this._canvas?.remove(); this._canvas = null; },
};
