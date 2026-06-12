/**
 * mapping.js — 3D SLAM map renderer using Three.js.
 *
 * Receives full map snapshots at 1Hz from the driver (voxel-deduplicated).
 * Binary protocol (v2, 17-byte header):
 *   [float32 robot_x, robot_y, robot_yaw] (12 bytes)
 *   [uint8 flags]                          (1 byte: bit0=full_map, bit1=has_z)
 *   [uint32 num_points]                    (4 bytes)
 *   Body: [float32 x, y, z] × N           (if has_z, 12 bytes/point)
 *      or [float32 x, y] × N              (if !has_z, 8 bytes/point, z=0)
 *
 * Legacy protocol (16-byte header) still supported for backward compatibility.
 *
 * Renders a rainbow height-colored 3D point cloud with robot position indicator.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAX_POINTS = 80000;

export const MappingRenderer = {
  name: 'mapping',
  canRender: (hint) => hint === 'sensor/mapping',

  _el: null,
  _renderer: null,
  _scene: null,
  _camera: null,
  _controls: null,
  _points: null,
  _positions: null,
  _colors: null,
  _robotMesh: null,
  _raf: null,
  _ro: null,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-mapping';
    this._el.style.cssText = 'width:100%;height:100%;position:relative;overflow:hidden';
    container.appendChild(this._el);

    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;

    // Scene
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x000000);

    // Camera — isometric-ish starting angle
    this._camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 500);
    this._camera.position.set(5, 8, 5);
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

    // Point cloud geometry (pre-allocated)
    const geo = new THREE.BufferGeometry();
    this._positions = new Float32Array(MAX_POINTS * 3);
    this._colors = new Float32Array(MAX_POINTS * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(this._positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this._colors, 3));
    geo.setDrawRange(0, 0);

    const mat = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: true,
      sizeAttenuation: true,
    });
    this._points = new THREE.Points(geo, mat);
    this._scene.add(this._points);

    // Robot indicator — green cone pointing forward
    const coneGeo = new THREE.ConeGeometry(0.15, 0.4, 8);
    coneGeo.rotateX(Math.PI / 2); // point along +Z (forward)
    const coneMat = new THREE.MeshBasicMaterial({ color: 0x4DDB6A });
    this._robotMesh = new THREE.Mesh(coneGeo, coneMat);
    this._scene.add(this._robotMesh);

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

  onData(buffer) {
    if (!(buffer instanceof ArrayBuffer)) return;

    const byteLen = buffer.byteLength;
    const view = new DataView(buffer);

    // Detect protocol version by header size
    let robotX, robotY, robotYaw, flags, numPoints, headerSize, hasZ;

    if (byteLen >= 17) {
      // Try new protocol: check if flags byte makes sense
      const possibleFlags = view.getUint8(12);
      const possibleNum = view.getUint32(13, true);
      const expectedBodyNew = hasZ ? possibleNum * 12 : possibleNum * 8;

      // Heuristic: new protocol has flags with bit0 or bit1 set
      if ((possibleFlags & 0x02) !== 0) {
        // New protocol (has_z flag set)
        robotX = view.getFloat32(0, true);
        robotY = view.getFloat32(4, true);
        robotYaw = view.getFloat32(8, true);
        flags = possibleFlags;
        numPoints = possibleNum;
        headerSize = 17;
        hasZ = (flags & 0x02) !== 0;
      } else if (byteLen >= 16) {
        // Legacy protocol
        robotX = view.getFloat32(0, true);
        robotY = view.getFloat32(4, true);
        robotYaw = view.getFloat32(8, true);
        flags = 0;
        numPoints = view.getUint32(12, true);
        headerSize = 16;
        hasZ = false;
      } else {
        return;
      }
    } else if (byteLen >= 16) {
      // Legacy protocol
      robotX = view.getFloat32(0, true);
      robotY = view.getFloat32(4, true);
      robotYaw = view.getFloat32(8, true);
      flags = 0;
      numPoints = view.getUint32(12, true);
      headerSize = 16;
      hasZ = false;
    } else {
      return;
    }

    const bytesPerPoint = hasZ ? 12 : 8;
    const expectedBody = headerSize + numPoints * bytesPerPoint;
    if (byteLen < expectedBody) return;

    const count = Math.min(numPoints, MAX_POINTS);

    // Parse points and compute Z range for coloring
    const pos = this._positions;
    let zMin = Infinity, zMax = -Infinity;

    for (let i = 0; i < count; i++) {
      const off = headerSize + i * bytesPerPoint;
      const x = view.getFloat32(off, true);
      const y = view.getFloat32(off + 4, true);
      const z = hasZ ? view.getFloat32(off + 8, true) : 0;

      const idx = i * 3;
      // Map coordinate: x→right, z(height)→up, y→depth (Three.js convention)
      pos[idx] = x;
      pos[idx + 1] = z;
      pos[idx + 2] = -y;

      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    }

    // Rainbow height colormap
    const col = this._colors;
    const zRange = zMax - zMin;
    const zScale = zRange > 0.01 ? 1.0 / zRange : 1.0;

    for (let i = 0; i < count; i++) {
      const z = pos[i * 3 + 1]; // y in Three.js = z in map
      const t = hasZ ? (z - zMin) * zScale : 0.5; // normalized [0,1]
      const idx = i * 3;
      // Rainbow: red → yellow → green → cyan → blue → purple
      col[idx] = this._rainbowR(t);
      col[idx + 1] = this._rainbowG(t);
      col[idx + 2] = this._rainbowB(t);
    }

    // Update geometry
    const geo = this._points.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, count);

    // Update robot position
    if (this._robotMesh) {
      this._robotMesh.position.set(robotX, 0.2, -robotY);
      this._robotMesh.rotation.set(0, -robotYaw, 0);
    }
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  // Rainbow colormap helpers (HSV-like, 0=red, 0.25=green, 0.5=cyan, 0.75=blue, 1.0=purple)
  _rainbowR(t) {
    if (t < 0.25) return 1.0;
    if (t < 0.5) return 1.0 - (t - 0.25) * 4;
    if (t < 0.75) return 0.0;
    return (t - 0.75) * 4 * 0.7; // purple tint
  },
  _rainbowG(t) {
    if (t < 0.25) return t * 4;
    if (t < 0.5) return 1.0;
    if (t < 0.75) return 1.0 - (t - 0.5) * 4;
    return 0.0;
  },
  _rainbowB(t) {
    if (t < 0.25) return 0.0;
    if (t < 0.5) return (t - 0.25) * 4;
    if (t < 0.75) return 1.0;
    return 1.0;
  },

  unmount() {
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._controls?.dispose();
    this._points?.geometry?.dispose();
    this._points?.material?.dispose();
    this._robotMesh?.geometry?.dispose();
    this._robotMesh?.material?.dispose();
    this._renderer?.dispose();
    this._el?.remove();
    this._el = null;
    this._renderer = null;
    this._scene = null;
    this._camera = null;
    this._controls = null;
    this._points = null;
    this._robotMesh = null;
    this._positions = null;
    this._colors = null;
  },
};
