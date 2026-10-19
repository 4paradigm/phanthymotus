/**
 * qrcode.js — 最小 QR 编码器（byte 模式，纠错等级 M，版本 1–14）。
 *
 * 为什么不用现成库：这个控制台跑在机器人上，装完就可能再也连不上外网，
 * 页面里已经有的两个 CDN（字体）失效只是掉字体，而二维码失效等于功能没了。
 * 而且 web/ 没有构建步骤，npm 依赖进不来。
 *
 * 覆盖到版本 14（365 字节）——一条 `https://<ip>:15678/?token=<token>` 撑死
 * 一百来字节，留了足够余量；超了就抛，由调用方退回「复制链接」。
 *
 * 正确性由 qrcode.test.mjs 对着 segno（Python 参考实现）逐模块比对保证，
 * 包括掩码选择与格式信息——这两处错了，图仍然画得出来，只是扫不出来。
 */

// ── GF(256) ─────────────────────────────────────────────────────────────────
// 本原多项式 0x11D，QR 规范指定。

const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
(function initGf() {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();

function gfMul(a, b) {
  if (a === 0 || b === 0) return 0;
  return EXP[LOG[a] + LOG[b]];
}

/** 生成多项式 x^n 的系数（最高次在前，首项恒为 1，故省略）。 */
function rsGenerator(n) {
  let poly = [1];
  for (let i = 0; i < n; i++) {
    const next = new Array(poly.length + 1).fill(0);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= poly[j];
      next[j + 1] ^= gfMul(poly[j], EXP[i]);
    }
    poly = next;
  }
  return poly;
}

/** 对一块数据码字算 n 个纠错码字。 */
function rsEncode(data, n) {
  const gen = rsGenerator(n);
  const res = new Uint8Array(n);
  for (const byte of data) {
    const factor = byte ^ res[0];
    res.copyWithin(0, 1);
    res[n - 1] = 0;
    if (factor !== 0) {
      for (let i = 0; i < n; i++) res[i] ^= gfMul(gen[i + 1], factor);
    }
  }
  return res;
}

// ── 版本表（纠错等级 M）────────────────────────────────────────────────────
// [每块纠错码字, 组1块数, 组1数据码字, 组2块数, 组2数据码字]

const EC_M = [
  null,
  [10, 1, 16, 0, 0],   // v1
  [16, 1, 28, 0, 0],
  [26, 1, 44, 0, 0],
  [18, 2, 32, 0, 0],
  [24, 2, 43, 0, 0],
  [16, 4, 27, 0, 0],
  [18, 4, 31, 0, 0],
  [22, 2, 38, 2, 39],
  [22, 3, 36, 2, 37],
  [26, 4, 43, 1, 44],  // v10
  [30, 1, 50, 4, 51],
  [22, 6, 36, 2, 37],
  [22, 8, 37, 1, 38],
  [24, 4, 40, 5, 41],  // v14
];

const MAX_VERSION = EC_M.length - 1;

/** 对齐图案中心坐标（版本 1–14）。 */
const ALIGN = [
  null,
  [], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34],
  [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50],
  [6, 30, 54], [6, 32, 58], [6, 34, 62], [6, 26, 46, 66],
];

function dataCodewords(version) {
  const [, b1, d1, b2, d2] = EC_M[version];
  return b1 * d1 + b2 * d2;
}

/** byte 模式下字符数指示符的位宽：版本 10 起变宽，漏掉这一跳整条码流会错位。 */
function lengthBits(version) {
  return version < 10 ? 8 : 16;
}

function byteCapacity(version) {
  return dataCodewords(version) - 1 - (lengthBits(version) / 8);
}

// ── 编码 ────────────────────────────────────────────────────────────────────

/**
 * 把字符串编成 QR 模块矩阵。
 *
 * @param {string} text
 * @returns {{version:number, size:number, get:(r:number,c:number)=>0|1}}
 */
export function qrMatrix(text) {
  const bytes = new TextEncoder().encode(text);

  let version = 0;
  for (let v = 1; v <= MAX_VERSION; v++) {
    if (bytes.length <= byteCapacity(v)) { version = v; break; }
  }
  if (!version) {
    throw new RangeError(`内容过长：${bytes.length} 字节，上限 ${byteCapacity(MAX_VERSION)}`);
  }

  const codewords = buildCodewords(bytes, version);
  const size = version * 4 + 17;
  const modules = new Uint8Array(size * size);
  const reserved = new Uint8Array(size * size);

  drawFunctionPatterns(modules, reserved, size, version);
  drawData(modules, reserved, size, codewords);

  // 八个掩码各评一次分，取最低。评分在写格式信息**之前**做（ISO 7.8，
  // 格式信息不属于被评估的区域）；分数算错不会让图画不出来，只会让某些
  // 扫码器认不出，所以这段的判据是能否被真实解码器读出，见测试。
  let best = null;
  for (let mask = 0; mask < 8; mask++) {
    const candidate = modules.slice();
    applyMask(candidate, reserved, size, mask);
    const score = penalty(candidate, size);
    if (!best || score < best.score) best = { score, mask, modules: candidate };
  }
  drawFormat(best.modules, size, best.mask);

  const final = best.modules;
  return {
    version,
    size,
    mask: best.mask,
    get: (r, c) => final[r * size + c],
  };
}

/** 码流：模式 + 长度 + 数据 + 补位，分块算纠错，再交错。 */
function buildCodewords(bytes, version) {
  const total = dataCodewords(version);
  const bits = [];
  const push = (value, width) => {
    for (let i = width - 1; i >= 0; i--) bits.push((value >> i) & 1);
  };

  push(0b0100, 4);                       // byte 模式
  push(bytes.length, lengthBits(version));
  for (const b of bytes) push(b, 8);

  // 终止符最多 4 位，剩余空间不足时截短
  const room = total * 8 - bits.length;
  push(0, Math.min(4, room));
  while (bits.length % 8 !== 0) bits.push(0);

  const data = new Uint8Array(total);
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0;
    for (let j = 0; j < 8; j++) byte = (byte << 1) | bits[i + j];
    data[i / 8] = byte;
  }
  // 填充字节 0xEC / 0x11 交替，规范指定
  for (let i = bits.length / 8, alt = 0; i < total; i++, alt++) {
    data[i] = alt % 2 === 0 ? 0xec : 0x11;
  }

  const [ecLen, b1, d1, b2, d2] = EC_M[version];
  const blocks = [];
  let offset = 0;
  for (let i = 0; i < b1; i++) {
    blocks.push(data.subarray(offset, offset + d1));
    offset += d1;
  }
  for (let i = 0; i < b2; i++) {
    blocks.push(data.subarray(offset, offset + d2));
    offset += d2;
  }
  const ecBlocks = blocks.map(b => rsEncode(b, ecLen));

  // 交错：先按列取遍所有数据块，再按列取遍所有纠错块
  const out = [];
  const maxData = Math.max(d1, d2 || 0);
  for (let i = 0; i < maxData; i++) {
    for (const block of blocks) if (i < block.length) out.push(block[i]);
  }
  for (let i = 0; i < ecLen; i++) {
    for (const block of ecBlocks) out.push(block[i]);
  }
  return out;
}

// ── 功能图案 ────────────────────────────────────────────────────────────────

function drawFunctionPatterns(m, reserved, size, version) {
  const set = (r, c, v) => {
    if (r < 0 || c < 0 || r >= size || c >= size) return;
    m[r * size + c] = v;
    reserved[r * size + c] = 1;
  };

  // 三个定位图案 + 分隔带（8x8 的范围整体占位）
  for (const [br, bc] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let r = -1; r <= 7; r++) {
      for (let c = -1; c <= 7; c++) {
        const inner = r >= 0 && r <= 6 && c >= 0 && c <= 6;
        const ring = inner && (r === 0 || r === 6 || c === 0 || c === 6);
        const core = inner && r >= 2 && r <= 4 && c >= 2 && c <= 4;
        set(br + r, bc + c, ring || core ? 1 : 0);
      }
    }
  }

  // 定时图案
  for (let i = 8; i < size - 8; i++) {
    const v = i % 2 === 0 ? 1 : 0;
    set(6, i, v);
    set(i, 6, v);
  }

  // 对齐图案：与定位图案重叠的三个角落要跳过
  const centers = ALIGN[version];
  for (const r of centers) {
    for (const c of centers) {
      const nearFinder = (r <= 8 && c <= 8)
        || (r <= 8 && c >= size - 9)
        || (r >= size - 9 && c <= 8);
      if (nearFinder) continue;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const edge = Math.max(Math.abs(dr), Math.abs(dc));
          set(r + dr, c + dc, edge === 1 ? 0 : 1);
        }
      }
    }
  }

  // 恒黑模块
  set(size - 8, 8, 1);

  // 格式信息占位（值在选定掩码后写）
  for (let i = 0; i < 9; i++) {
    if (i !== 6) { reserved[8 * size + i] = 1; reserved[i * size + 8] = 1; }
  }
  for (let i = 0; i < 8; i++) {
    reserved[8 * size + (size - 1 - i)] = 1;
    reserved[(size - 1 - i) * size + 8] = 1;
  }

  // 版本信息（版本 7 起）
  if (version >= 7) {
    const bits = versionBits(version);
    for (let i = 0; i < 18; i++) {
      const bit = (bits >> i) & 1;
      const r = Math.floor(i / 3);
      const c = size - 11 + (i % 3);
      set(r, c, bit);
      set(c, r, bit);
    }
  }
}

/**
 * BCH 余数。商的次数 = 被除数次数 − 生成多项式次数，循环次数必须按它来：
 * 多循环几次会把移位量移成负数，JS 的 `<<` 又把负数按 mod 32 解释，
 * 于是余数被一个完全无关的值污染——图照画，扫不出来。
 */
function bchRemainder(data, dataBits, generator, genDegree) {
  let rem = data << genDegree;
  for (let i = 0; i < dataBits; i++) {
    const bit = dataBits + genDegree - 1 - i;
    if (rem & (1 << bit)) rem ^= generator << (bit - genDegree);
  }
  return rem;
}

function versionBits(version) {
  return (version << 12) | bchRemainder(version, 6, 0x1f25, 12);
}

/** 纠错等级 M 的格式位是 00；BCH(15,5) 后与 0x5412 异或。 */
function formatBits(mask) {
  const data = (0b00 << 3) | mask;
  return ((data << 10) | bchRemainder(data, 5, 0x537, 10)) ^ 0x5412;
}

function drawFormat(m, size, mask) {
  const bits = formatBits(mask);
  for (let i = 0; i < 15; i++) {
    const bit = (bits >> i) & 1;
    // 左上角：沿第 8 列往下、第 8 行往右，跳过定时图案那一格
    if (i < 6) m[i * size + 8] = bit;
    else if (i === 6) m[7 * size + 8] = bit;
    else if (i === 7) m[8 * size + 8] = bit;
    else if (i === 8) m[8 * size + 7] = bit;
    else m[8 * size + (14 - i)] = bit;
    // 副本：右上 + 左下
    if (i < 8) m[8 * size + (size - 1 - i)] = bit;
    else m[(size - 15 + i) * size + 8] = bit;
  }
}

// ── 数据排布与掩码 ──────────────────────────────────────────────────────────

function drawData(m, reserved, size, codewords) {
  let bitIndex = 0;
  let upward = true;
  for (let right = size - 1; right > 0; right -= 2) {
    if (right === 6) right--;            // 第 6 列是定时图案，整列跳过
    for (let step = 0; step < size; step++) {
      const r = upward ? size - 1 - step : step;
      for (const c of [right, right - 1]) {
        if (reserved[r * size + c]) continue;
        const byte = codewords[bitIndex >> 3];
        const bit = byte === undefined ? 0 : (byte >> (7 - (bitIndex & 7))) & 1;
        m[r * size + c] = bit;
        bitIndex++;
      }
    }
    upward = !upward;
  }
}

const MASKS = [
  (r, c) => (r + c) % 2 === 0,
  (r) => r % 2 === 0,
  (r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

function applyMask(m, reserved, size, mask) {
  const fn = MASKS[mask];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (reserved[r * size + c]) continue;
      if (fn(r, c)) m[r * size + c] ^= 1;
    }
  }
}

/** 一行（或一列）里 1:1:3:1:1 图形的罚分。符号边界之外视为浅色。 */
function finderLike(seq, size) {
  const CORE = [1, 0, 1, 1, 1, 0, 1];
  let score = 0;
  let idx = 0;
  while (idx <= size - 7) {
    let hit = true;
    for (let k = 0; k < 7; k++) {
      if (seq[idx + k] !== CORE[k]) { hit = false; break; }
    }
    if (!hit) { idx++; continue; }
    let lightBefore = true;
    for (let k = Math.max(idx - 4, 0); k < idx; k++) if (seq[k]) { lightBefore = false; break; }
    let lightAfter = true;
    for (let k = idx + 7; k < Math.min(idx + 11, size); k++) if (seq[k]) { lightAfter = false; break; }
    if (lightBefore || lightAfter) { score += 40; idx += 7; }
    else idx += 4;          // 重叠的下一个可能起点
  }
  return score;
}

/** 规范的四条罚分规则。 */
function penalty(m, size) {
  const at = (r, c) => m[r * size + c];
  let score = 0;

  // 规则 1：同色连续 5 格以上
  for (let i = 0; i < size; i++) {
    for (const rowwise of [true, false]) {
      let run = 1;
      for (let j = 1; j < size; j++) {
        const cur = rowwise ? at(i, j) : at(j, i);
        const prev = rowwise ? at(i, j - 1) : at(j - 1, i);
        if (cur === prev) {
          run++;
        } else {
          if (run >= 5) score += run - 2;
          run = 1;
        }
      }
      if (run >= 5) score += run - 2;
    }
  }

  // 规则 2：2x2 同色块
  for (let r = 0; r < size - 1; r++) {
    for (let c = 0; c < size - 1; c++) {
      const v = at(r, c);
      if (v === at(r, c + 1) && v === at(r + 1, c) && v === at(r + 1, c + 1)) score += 3;
    }
  }

  // 规则 3：1:1:3:1:1 图形，前面或后面跟着 4 格浅色。符号之外算浅色 ——
  // 贴着边缘的那一处按 11 格模板去匹配会漏掉，而它恰恰是最容易被误认成
  // 定位图案的位置。
  const line = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (let k = 0; k < size; k++) line[k] = at(i, k);
    score += finderLike(line, size);
    for (let k = 0; k < size; k++) line[k] = at(k, i);
    score += finderLike(line, size);
  }

  // 规则 4：深色比例偏离 50%
  let dark = 0;
  for (let i = 0; i < size * size; i++) dark += m[i];
  const ratio = (dark * 100) / (size * size);
  score += Math.floor(Math.abs(ratio - 50) / 5) * 10;

  return score;
}

// ── 渲染 ────────────────────────────────────────────────────────────────────

/**
 * 渲染成 SVG 字符串。用 SVG 而不是 canvas：任意尺寸都锐利，不必操心
 * devicePixelRatio，打印出来也能扫。
 *
 * 按行做游程合并，模块多时比一格一个 rect 小一个数量级。
 */
export function qrSvg(text, { margin = 2, color = '#111', background = '#fff' } = {}) {
  const qr = qrMatrix(text);
  const dim = qr.size + margin * 2;
  const runs = [];
  for (let r = 0; r < qr.size; r++) {
    let start = -1;
    for (let c = 0; c <= qr.size; c++) {
      const on = c < qr.size && qr.get(r, c) === 1;
      if (on && start < 0) start = c;
      if (!on && start >= 0) {
        runs.push(`M${start + margin} ${r + margin}h${c - start}v1h-${c - start}z`);
        start = -1;
      }
    }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dim} ${dim}" `
       + `shape-rendering="crispEdges" role="img">`
       + `<rect width="${dim}" height="${dim}" fill="${background}"/>`
       + `<path fill="${color}" d="${runs.join('')}"/></svg>`;
}
