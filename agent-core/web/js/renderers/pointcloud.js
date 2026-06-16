/**
 * pointcloud.js — 3D point cloud renderer using Three.js.
 * Accumulates ~1 second of frames for dense Livox Mid-360 non-repetitive scan coverage.
 * Expects binary WebSocket data: [uint32 numPoints][float32 x,y,z,intensity × N]
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAX_POINTS = 50000;  // 10 frames × 5000 points
const ACCUM_INTERVAL = 1000;  // 1 second accumulation window

export const PointCloudRenderer = {
  name: 'pointcloud',
  canRender: (hint) => hint === 'sensor/pointcloud',

  _el: null,
  _renderer: null,
  _scene: null,
  _camera: null,
  _controls: null,
  _points: null,
  _positions: null,
  _colors: null,
  _raf: null,
  _ro: null,
  _accumPos: null,
  _accumCol: null,
  _accumCount: 0,
  _swapTimer: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-pointcloud';
    this._el.style.cssText = 'width:100%;height:100%;position:relative;overflow:hidden';
    container.appendChild(this._el);

    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;

    // Scene
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x1c1c1e);

    // Camera
    this._camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 200);
    this._camera.position.set(0, 5, 8);
    this._camera.lookAt(0, 0, 0);

    // Renderer
    this._renderer = new THREE.WebGLRenderer({ antialias: false });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._el.appendChild(this._renderer.domElement);

    // Controls
    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.1;

    // Grid
    const grid = new THREE.GridHelper(20, 20, 0x444444, 0x333333);
    this._scene.add(grid);

    // Origin axes
    const axes = new THREE.AxesHelper(1);
    this._scene.add(axes);

    // Points geometry (pre-allocated)
    const geo = new THREE.BufferGeometry();
    this._positions = new Float32Array(MAX_POINTS * 3);
    this._colors = new Float32Array(MAX_POINTS * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(this._positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this._colors, 3));
    geo.setDrawRange(0, 0);

    const mat = new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      sizeAttenuation: true,
    });
    this._points = new THREE.Points(geo, mat);
    this._scene.add(this._points);

    // Accumulation buffer
    this._accumPos = new Float32Array(MAX_POINTS * 3);
    this._accumCol = new Float32Array(MAX_POINTS * 3);
    this._accumCount = 0;

    // Swap timer: every 1s, push accumulated points to display
    this._swapTimer = setInterval(() => this._swap(), ACCUM_INTERVAL);

    // Resize observer
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);

    // Render loop
    const animate = () => {
      this._raf = requestAnimationFrame(animate);
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
    };
    animate();
  },

  _resize() {
    if (!this._el || !this._renderer) return;
    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
  },

  _swap() {
    const count = this._accumCount;
    if (count === 0) return;

    // Copy accumulation to display buffer
    this._positions.set(this._accumPos.subarray(0, count * 3));
    this._colors.set(this._accumCol.subarray(0, count * 3));

    const geo = this._points.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, count);

    // Reset accumulator
    this._accumCount = 0;
  },

  onData(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) return;

    const view = new DataView(buffer);
    const firstUint = view.getUint32(0, true);

    let numPoints, pointOffset, pointStride, hasIntensity;

    if (firstUint < 256) {
      // New PointCloud2 passthrough: [uint32 point_step][uint32 total_points][raw bytes]
      pointStride = firstUint;
      numPoints = view.getUint32(4, true);
      pointOffset = 8;
      hasIntensity = pointStride >= 16;  // intensity at offset 12 if point_step >= 16
    } else {
      // Legacy format: [uint32 N][float32 x,y,z,intensity × N]
      numPoints = firstUint;
      pointOffset = 4;
      pointStride = 16;
      hasIntensity = true;
    }

    const expected = pointOffset + numPoints * pointStride;
    if (buffer.byteLength < expected) {
      // Adjust numPoints to what's actually available
      numPoints = Math.floor((buffer.byteLength - pointOffset) / pointStride);
    }
    if (numPoints <= 0) return;

    const startIdx = this._accumCount;
    const available = MAX_POINTS - startIdx;
    const count = Math.min(numPoints, available);
    if (count <= 0) return;  // buffer full, wait for swap

    const pos = this._accumPos;
    const col = this._accumCol;

    for (let i = 0; i < count; i++) {
      const off = pointOffset + i * pointStride;
      const x = view.getFloat32(off, true);
      const y = view.getFloat32(off + 4, true);
      const z = view.getFloat32(off + 8, true);
      const intensity = hasIntensity ? view.getFloat32(off + 12, true) : 0;

      const idx = (startIdx + i) * 3;
      // Livox frame (inverted mount): z down → Three.js: x right, y up, z forward
      pos[idx]     = y;
      pos[idx + 1] = -z;
      pos[idx + 2] = -x;

      // Jet colormap based on intensity (0-255 typical range)
      const t = Math.min(1, Math.max(0, intensity / 255));
      col[idx]     = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.75) * 4));
      col[idx + 1] = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.5) * 4));
      col[idx + 2] = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.25) * 4));
    }

    this._accumCount = startIdx + count;
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  unmount() {
    if (this._swapTimer) clearInterval(this._swapTimer);
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._controls?.dispose();
    this._points?.geometry?.dispose();
    this._points?.material?.dispose();
    this._renderer?.dispose();
    this._el?.remove();
    this._el = null;
    this._renderer = null;
    this._scene = null;
    this._camera = null;
    this._controls = null;
    this._points = null;
    this._accumPos = null;
    this._accumCol = null;
    this._swapTimer = null;
  },
};
