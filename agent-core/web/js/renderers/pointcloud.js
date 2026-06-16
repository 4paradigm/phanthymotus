/**
 * pointcloud.js — 3D point cloud renderer using Three.js.
 * Sliding window: keeps last 5 frames merged for dense Livox Mid-360 non-repetitive scan coverage.
 * Axis mapping loaded from tool config (configurable lidar orientation).
 * Expects binary WebSocket data: [uint32 point_step][uint32 total_points][raw bytes]
 *   or legacy: [uint32 numPoints][float32 x,y,z,intensity × N]
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAX_POINTS = 50000;
const FRAME_WINDOW = 5;           // sliding window: keep last 5 frames
const MAX_POINTS_PER_FRAME = 10000;

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
  _frames: [],          // ring buffer of { pos, col, count }
  _displayDirty: false,
  _mcpId: null,
  _axisMap: null,       // { xIdx, xSign, yIdx, ySign, zIdx, zSign }

  mount(container, mcpId) {
    this._mcpId = mcpId || null;
    this._frames = [];
    this._displayDirty = false;

    // Set default axis map (matches previous hardcoded: X←y, Y←-z, Z←-x)
    this._buildAxisMap(null);

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

    // Origin axes — rotated so blue (Z) points forward (Livox +x maps to Three.js -Z)
    const axes = new THREE.AxesHelper(1);
    axes.rotation.y = Math.PI;
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

    // Resize observer
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);

    // Render loop
    const animate = () => {
      this._raf = requestAnimationFrame(animate);
      if (this._displayDirty) this._mergeFrames();
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
    };
    animate();

    // Fetch config (async, non-blocking)
    this._loadConfig();
  },

  async _loadConfig() {
    if (!this._mcpId) return;
    try {
      const res = await fetch(`/api/canvas/tool-config/${this._mcpId}/lidar_cloud`);
      const json = await res.json();
      if (json.data) this._buildAxisMap(json.data);
    } catch { /* use defaults */ }
  },

  _buildAxisMap(cfg) {
    const idxOf = s => ({ x: 0, y: 1, z: 2 }[s] ?? 1);
    this._axisMap = {
      xIdx: idxOf(cfg?.axis_x_source ?? 'y'), xSign: cfg?.axis_x_negate ? -1 : 1,
      yIdx: idxOf(cfg?.axis_y_source ?? 'z'), ySign: cfg?.axis_y_negate ? -1 : 1,
      zIdx: idxOf(cfg?.axis_z_source ?? 'x'), zSign: cfg?.axis_z_negate ? -1 : 1,
    };
  },

  _resize() {
    if (!this._el || !this._renderer) return;
    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
  },

  _mergeFrames() {
    this._displayDirty = false;
    let offset = 0;
    for (const frame of this._frames) {
      const n3 = frame.count * 3;
      if (offset + n3 > MAX_POINTS * 3) break;
      this._positions.set(frame.pos.subarray(0, n3), offset);
      this._colors.set(frame.col.subarray(0, n3), offset);
      offset += n3;
    }
    const totalPoints = offset / 3;
    const geo = this._points.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, totalPoints);
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
      hasIntensity = pointStride >= 16;
    } else {
      // Legacy format: [uint32 N][float32 x,y,z,intensity × N]
      numPoints = firstUint;
      pointOffset = 4;
      pointStride = 16;
      hasIntensity = true;
    }

    const expected = pointOffset + numPoints * pointStride;
    if (buffer.byteLength < expected) {
      numPoints = Math.floor((buffer.byteLength - pointOffset) / pointStride);
    }
    if (numPoints <= 0) return;

    const count = Math.min(numPoints, MAX_POINTS_PER_FRAME);
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const am = this._axisMap;

    for (let i = 0; i < count; i++) {
      const off = pointOffset + i * pointStride;
      const raw0 = view.getFloat32(off, true);      // x
      const raw1 = view.getFloat32(off + 4, true);  // y
      const raw2 = view.getFloat32(off + 8, true);  // z
      const intensity = hasIntensity ? view.getFloat32(off + 12, true) : 0;

      const raw = [raw0, raw1, raw2];
      const idx = i * 3;
      pos[idx]     = am.xSign * raw[am.xIdx];
      pos[idx + 1] = am.ySign * raw[am.yIdx];
      pos[idx + 2] = am.zSign * raw[am.zIdx];

      // Jet colormap based on intensity (0-255 typical range)
      const t = Math.min(1, Math.max(0, intensity / 255));
      col[idx]     = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.75) * 4));
      col[idx + 1] = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.5) * 4));
      col[idx + 2] = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.25) * 4));
    }

    // Sliding window: push new frame, keep last FRAME_WINDOW
    this._frames.push({ pos, col, count });
    if (this._frames.length > FRAME_WINDOW) {
      this._frames.shift();
    }
    this._displayDirty = true;
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  unmount() {
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
    this._positions = null;
    this._colors = null;
    this._frames = [];
    this._mcpId = null;
  },
};
