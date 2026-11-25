/**
 * bare-three-loader.mjs — node 测试进程里的 importmap 替身。
 *
 * 用法（在测试文件顶层、任何静态 import 触发 three 之前）：
 *   import { register } from 'node:module';
 *   register(new URL('./bare-three-loader.mjs', import.meta.url));
 *   const { RENDERERS } = await import('./renderers/index.js');  // 动态 import
 *
 * node --test 每个测试文件一个子进程，钩子只影响本文件所在进程。
 */

const STUB_URL = new URL('./three-stub.mjs', import.meta.url).href;

export async function initialize() {
  // 无需状态；resolve 闭包直接引用 STUB_URL。
}

export function resolve(specifier, _context, nextResolve) {
  if (specifier === 'three') {
    return { url: STUB_URL, shortCircuit: true };
  }
  if (specifier.startsWith('three/addons/')) {
    return { url: STUB_URL, shortCircuit: true };
  }
  return nextResolve(specifier, _context);
}
