/**
 * QR 编码器 —— 一张扫不出来的二维码和一张扫得出来的二维码，看起来一模一样。
 *
 * 这是这个文件存在的全部理由：编码错了没有任何症状。开发时踩到的两处，
 * 都是画得出来但扫不动：
 *
 *  - BCH 的除法循环多转了几圈，移位量变成负数，而 JS 的 `<<` 把负数按 mod 32
 *    解释，于是格式信息被一个无关的值污染。图是好的，纠错等级和掩码号读出来
 *    是垃圾。
 *  - 版本 10 起字符数指示符从 8 位变 16 位。漏掉这一跳，整条码流错位一个字节。
 *
 * 所以这里不验「画出来像二维码」，而是把矩阵**解回去**：格式信息的 BCH 校验、
 * 每个块的 Reed-Solomon 伴随式、以及原文。伴随式全零意味着数据与纠错码自洽，
 * 这正是真实扫码器要做的判断。
 *
 * 编码器本身的外部验证在开发时做过一轮，不在这里重复：14 个版本的功能图案与
 * segno（Python 参考实现）逐模块一致；121 条随机长度的内容由 zxing-cpp
 * （多数手机扫码器用的解码库）全部解出。下面的定值就是那一轮的产物。
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { qrMatrix, qrSvg } from './qrcode.js';

// ── 把矩阵解回去 ─────────────────────────────────────────────────────────────
// 刻意不 import 编码器的内部函数：共用一份排布代码的话，排布写错了两边会
// 一起错，测试反而会通过。

const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
{
  let x = 1;
  for (let i = 0; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 0x100) x ^= 0x11d; }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
}
const mul = (a, b) => (a === 0 || b === 0) ? 0 : EXP[LOG[a] + LOG[b]];

/** 纠错等级 M 的分块表，和编码器里那份独立写一遍。 */
const BLOCKS_M = {
  1: [10, [16]], 2: [16, [28]], 3: [26, [44]], 4: [18, [32, 32]],
  5: [24, [43, 43]], 6: [16, [27, 27, 27, 27]], 7: [18, [31, 31, 31, 31]],
  8: [22, [38, 38, 39, 39]], 9: [22, [36, 36, 36, 37, 37]],
  10: [26, [43, 43, 43, 43, 44]], 11: [30, [50, 51, 51, 51, 51]],
  12: [22, [36, 36, 36, 36, 36, 36, 37, 37]],
  13: [22, [37, 37, 37, 37, 37, 37, 37, 37, 38]],
  14: [24, [40, 40, 40, 40, 41, 41, 41, 41, 41]],
};

const ALIGN_CENTERS = {
  1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
  7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
  11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
};

/** 哪些格子是功能图案（不承载数据）。 */
function functionMap(size, version) {
  const f = new Uint8Array(size * size);
  const mark = (r, c) => { if (r >= 0 && c >= 0 && r < size && c < size) f[r * size + c] = 1; };
  for (const [br, bc] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let r = -1; r <= 7; r++) for (let c = -1; c <= 7; c++) mark(br + r, bc + c);
  }
  for (let i = 0; i < size; i++) { mark(6, i); mark(i, 6); }
  for (const r of ALIGN_CENTERS[version]) for (const c of ALIGN_CENTERS[version]) {
    if ((r <= 8 && c <= 8) || (r <= 8 && c >= size - 9) || (r >= size - 9 && c <= 8)) continue;
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) mark(r + dr, c + dc);
  }
  for (let i = 0; i < 9; i++) { mark(8, i); mark(i, 8); }
  for (let i = 0; i < 8; i++) { mark(8, size - 1 - i); mark(size - 1 - i, 8); }
  if (version >= 7) {
    for (let i = 0; i < 18; i++) {
      const r = Math.floor(i / 3), c = size - 11 + (i % 3);
      mark(r, c); mark(c, r);
    }
  }
  return f;
}

const MASK_FN = [
  (r, c) => (r + c) % 2 === 0, (r) => r % 2 === 0, (r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

/** 读格式信息，校验 BCH 余数并取出纠错等级与掩码号。 */
function readFormat(grid, size) {
  const raw = [];
  for (const copy of [0, 1]) {
    let bits = 0;
    for (let i = 0; i < 15; i++) {
      let v;
      if (copy === 0) {
        if (i < 6) v = grid(i, 8);
        else if (i === 6) v = grid(7, 8);
        else if (i === 7) v = grid(8, 8);
        else if (i === 8) v = grid(8, 7);
        else v = grid(8, 14 - i);
      } else {
        v = i < 8 ? grid(8, size - 1 - i) : grid(size - 15 + i, 8);
      }
      bits |= v << i;
    }
    raw.push(bits);
  }
  assert.equal(raw[0], raw[1], '格式信息的两份副本不一致');
  const bits = raw[0] ^ 0x5412;
  // BCH(15,5)：整个码字应当能被 0x537 整除
  let rem = bits;
  for (let i = 14; i >= 10; i--) if (rem & (1 << i)) rem ^= 0x537 << (i - 10);
  assert.equal(rem, 0, '格式信息 BCH 校验不过');
  return { ecc: bits >> 13, mask: (bits >> 10) & 7 };
}

/** 完整解码：格式信息 → 去掩码 → 读码字 → 校验伴随式 → 取原文。 */
function decode(qr) {
  const size = qr.size, version = qr.version;
  const m = new Uint8Array(size * size);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) m[r * size + c] = qr.get(r, c);

  const { ecc, mask } = readFormat((r, c) => m[r * size + c], size);
  const fn = functionMap(size, version);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (!fn[r * size + c] && MASK_FN[mask](r, c)) m[r * size + c] ^= 1;
    }
  }

  const bits = [];
  let upward = true;
  for (let right = size - 1; right > 0; right -= 2) {
    if (right === 6) right--;
    for (let step = 0; step < size; step++) {
      const r = upward ? size - 1 - step : step;
      for (const c of [right, right - 1]) if (!fn[r * size + c]) bits.push(m[r * size + c]);
    }
    upward = !upward;
  }
  const cw = [];
  for (let i = 0; i + 8 <= bits.length; i += 8) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
    cw.push(b);
  }

  const [ecLen, sizes] = BLOCKS_M[version];
  const data = sizes.map(() => []), ecBlocks = sizes.map(() => []);
  let p = 0;
  for (let i = 0; i < Math.max(...sizes); i++) {
    for (let b = 0; b < sizes.length; b++) if (i < sizes[b]) data[b].push(cw[p++]);
  }
  for (let i = 0; i < ecLen; i++) for (let b = 0; b < sizes.length; b++) ecBlocks[b].push(cw[p++]);

  // 伴随式：码字多项式在 α^0..α^(ecLen-1) 上应当全为 0
  const syndromes = [];
  for (let b = 0; b < sizes.length; b++) {
    const poly = [...data[b], ...ecBlocks[b]];
    for (let s = 0; s < ecLen; s++) {
      let acc = 0;
      for (const co of poly) acc = mul(acc, EXP[s]) ^ co;
      if (acc !== 0) syndromes.push([b, s]);
    }
  }

  const flat = data.flat();
  let bi = 0;
  const take = (n) => {
    let v = 0;
    for (let k = 0; k < n; k++) { v = (v << 1) | ((flat[bi >> 3] >> (7 - (bi & 7))) & 1); bi++; }
    return v;
  };
  const mode = take(4);
  const len = take(version < 10 ? 8 : 16);
  const out = [];
  for (let i = 0; i < len; i++) out.push(take(8));
  return {
    ecc, mask, mode, syndromes,
    text: new TextDecoder().decode(new Uint8Array(out)),
  };
}

// ── 测试 ────────────────────────────────────────────────────────────────────

const REAL_PAYLOADS = [
  'https://10.100.121.14:15678/?token=abc',
  'https://192.168.55.101:15678/?token=TD98xtkBAHjsDSIls0U58V107QAUihPV',
  'https://10.100.129.72:15678/?token=' + 'x'.repeat(64),
  'https://10.100.130.6:15678/?token=' + 'Q7z-_'.repeat(30),
  'https://[fe80::1]:15678/?token=short',
];

test('真实的接入链接都能解回原文，且纠错码自洽', () => {
  for (const payload of REAL_PAYLOADS) {
    const got = decode(qrMatrix(payload));
    assert.equal(got.text, payload, `解出来不是原文：${payload.slice(0, 40)}`);
    assert.deepEqual(got.syndromes, [], `Reed-Solomon 伴随式非零：${payload.slice(0, 40)}`);
    assert.equal(got.mode, 0b0100, 'byte 模式');
    assert.equal(got.ecc, 0b00, '纠错等级应为 M');
  }
});

test('每个版本都能解回原文 —— 覆盖 v10 的长度指示符变宽与 v7 的版本信息块', () => {
  const seen = new Set();
  // 从 1 字节扫到 v14 的上限。短到 v1 的内容不可能是一条接入链接（光是
  // `https://…:15678/?token=` 就 35 字节了），但版本表是按长度分段的，
  // 每一段都要走到。
  for (let len = 1; len <= 362; len += 7) {
    const text = 'aA9-_:/?.'.repeat(45).slice(0, len);
    const qr = qrMatrix(text);
    seen.add(qr.version);
    const got = decode(qr);
    assert.equal(got.text, text, `v${qr.version} len=${len} 解码不符`);
    assert.deepEqual(got.syndromes, [], `v${qr.version} len=${len} 伴随式非零`);
    assert.equal(got.mask, qr.mask, `v${qr.version} 格式信息里的掩码号与实际不符`);
  }
  // 版本 10 是长度指示符 8→16 位的分界，版本 7 是版本信息块出现的分界
  for (const v of [1, 7, 10, 14]) assert.ok(seen.has(v), `没覆盖到版本 ${v}`);
});

test('UTF-8 原样进出', () => {
  const text = 'https://10.100.130.6:15678/?token=中文机器人控制台';
  assert.equal(decode(qrMatrix(text)).text, text);
});

test('尺寸与版本选择', () => {
  assert.equal(qrMatrix('A').version, 1);
  assert.equal(qrMatrix('A').size, 21);
  // v1-M 装 14 字节，第 15 个必须换版本
  assert.equal(qrMatrix('x'.repeat(14)).version, 1);
  assert.equal(qrMatrix('x'.repeat(15)).version, 2);
  for (let v = 1; v <= 14; v++) {
    const qr = qrMatrix('x'.repeat([14, 26, 42, 62, 84, 106, 122, 152, 180, 213, 251, 287, 331, 362][v - 1]));
    assert.equal(qr.version, v, `容量边界上的版本应为 ${v}`);
    assert.equal(qr.size, v * 4 + 17);
  }
});

test('功能图案在该在的位置', () => {
  const qr = qrMatrix('https://10.100.121.14:15678/?token=abc');
  const { size } = qr;
  for (const [br, bc] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    // 定位图案：外环黑、内圈白、中心 3x3 黑
    assert.equal(qr.get(br, bc), 1, '定位图案外环');
    assert.equal(qr.get(br + 1, bc + 1), 0, '定位图案内圈');
    assert.equal(qr.get(br + 3, bc + 3), 1, '定位图案中心');
  }
  for (let i = 8; i < size - 8; i++) {
    assert.equal(qr.get(6, i), i % 2 === 0 ? 1 : 0, `水平定时图案 @${i}`);
    assert.equal(qr.get(i, 6), i % 2 === 0 ? 1 : 0, `垂直定时图案 @${i}`);
  }
  assert.equal(qr.get(size - 8, 8), 1, '恒黑模块');
});

test('内容超出容量时抛出，而不是悄悄截断', () => {
  assert.throws(() => qrMatrix('x'.repeat(400)), RangeError);
});

test('qrSvg 画出能用的 SVG，静区留够 4 格', () => {
  const svg = qrSvg('https://10.100.121.14:15678/?token=abc', { margin: 4 });
  assert.match(svg, /^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  assert.match(svg, /viewBox="0 0 37 37"/);          // v3 是 29，加两边各 4
  assert.match(svg, /shape-rendering="crispEdges"/);  // 不做抗锯齿，边缘才干净
  assert.ok(svg.includes('</svg>'));
});
