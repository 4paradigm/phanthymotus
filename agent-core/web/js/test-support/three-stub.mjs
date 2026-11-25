/**
 * three-stub.mjs — node 测试环境里 `three` 的替身。
 *
 * 浏览器由 index.html 的 importmap 把 three 指到 CDN；node 的测试进程没有
 * importmap，注册表扫描测试（renderer-registry.test.mjs）要 import mapping /
 * pointcloud / skeleton 这三个 import three 的渲染器，就得给裸说明符一个去处。
 * 测试从不真正渲染，所以桩是递归魔法代理：new THREE.X()、THREE.MathUtils.xxx()
 * 都返回同一个桩，链式属性访问也不炸。
 */

const stub = new Proxy(function () {}, {
  get(_target, prop) {
    if (prop === Symbol.toPrimitive) return () => 0;
    if (prop === 'then') return undefined; // 不是 thenable，避免被 await 误伤
    return stub;
  },
  construct() {
    return stub;
  },
  apply() {
    return stub;
  },
});

export default stub;
export const OrbitControls = stub;

// `import * as THREE` 拿到的是命名空间对象，只会暴露这里的具名导出——模块
// 顶层会真正执行的符号必须列出来（目前只有 mapping.js:40 的 new Vector3；
// 其余符号只在 mount/draw 里用到，测试从不调用，但列上以防再挪到顶层）。
export const Vector3 = stub;
export const Euler = stub;
export const Color = stub;
export const Scene = stub;
export const Group = stub;
export const Mesh = stub;
export const PerspectiveCamera = stub;
export const OrthographicCamera = stub;
export const WebGLRenderer = stub;
export const BufferGeometry = stub;
export const BufferAttribute = stub;
export const Points = stub;
export const PointsMaterial = stub;
export const LineBasicMaterial = stub;
export const MeshStandardMaterial = stub;
export const MeshBasicMaterial = stub;
export const AmbientLight = stub;
export const DirectionalLight = stub;
export const HemisphereLight = stub;
export const MathUtils = stub;
