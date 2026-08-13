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
 * Renders a rainbow height-colored point cloud with robot position indicator.
 * Supports a top-down 2D projection, the normal 3D view, free browsing and
 * follow/auto-fit controls.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAX_POINTS = 80000;
const MAX_PATH_POSES = 4096;

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
  _pathLine: null,
  _pathPositions: null,
  _goalMesh: null,
  _pathSummary: null,
  _planWs: null,
  _planReconnectTimer: null,
  _viewBtn: null,
  _viewMode: '3d',
  _saved3dPosition: null,
  _saved3dTarget: null,
  _mapBounds: null,
  _pathBounds: null,
  _raf: null,
  _ro: null,
  _followBtn: null,
  _followRobot: true,
  _robotPos: new THREE.Vector3(0, 0.2, 0),

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-mapping';
    this._el.style.cssText = 'width:100%;height:100%;position:relative;overflow:hidden';
    container.appendChild(this._el);
    this._viewMode = '3d';
    this._followRobot = true;
    this._mapBounds = null;
    this._pathBounds = null;

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

    // Controls — supports pan (right-click drag), rotate (left-click), zoom (scroll)
    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.1;
    this._controls.enablePan = true;
    this._controls.screenSpacePanning = true;

    // Disable follow when user interacts
    this._controls.addEventListener('start', () => {
      this._followRobot = false;
      this._updateFollowBtn();
    });

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

    // Robot indicator — green cone pointing along +X
    const coneGeo = new THREE.ConeGeometry(0.15, 0.4, 8);
    coneGeo.rotateZ(-Math.PI / 2); // cone tip points along +X
    const coneMat = new THREE.MeshBasicMaterial({ color: 0x4DDB6A });
    this._robotMesh = new THREE.Mesh(coneGeo, coneMat);
    this._scene.add(this._robotMesh);

    // Nav2 global plan overlay. Both map_view and /plan use the canonical map
    // frame, so composition stays in the dashboard and does not couple the
    // FAST-LIVO2 producer back to Nav2.
    const pathGeo = new THREE.BufferGeometry();
    this._pathPositions = new Float32Array(MAX_PATH_POSES * 3);
    pathGeo.setAttribute('position', new THREE.BufferAttribute(this._pathPositions, 3));
    pathGeo.setDrawRange(0, 0);
    const pathMat = new THREE.LineBasicMaterial({ color: 0x55d675 });
    this._pathLine = new THREE.Line(pathGeo, pathMat);
    this._pathLine.visible = false;
    this._scene.add(this._pathLine);

    const goalGeo = new THREE.SphereGeometry(0.12, 12, 8);
    const goalMat = new THREE.MeshBasicMaterial({ color: 0xffa640 });
    this._goalMesh = new THREE.Mesh(goalGeo, goalMat);
    this._goalMesh.visible = false;
    this._scene.add(this._goalMesh);

    this._pathSummary = document.createElement('div');
    this._pathSummary.style.cssText =
      'position:absolute;left:8px;bottom:7px;z-index:10;padding:3px 7px;' +
      'border-radius:4px;background:rgba(0,0,0,.62);color:#dce7df;font:11px monospace';
    this._pathSummary.textContent = '等待 Nav2 路径';
    this._el.appendChild(this._pathSummary);
    this._connectPlanStream();

    // Follow-robot toggle button
    this._followBtn = document.createElement('button');
    this._followBtn.style.cssText =
      'position:absolute;top:8px;right:8px;z-index:10;' +
      'width:28px;height:28px;border-radius:4px;border:1px solid rgba(255,255,255,0.3);' +
      'background:rgba(77,219,106,0.8);color:#fff;font-size:14px;cursor:pointer;' +
      'display:flex;align-items:center;justify-content:center;padding:0';
    this._followBtn.textContent = '\u2316'; // crosshair character
    this._followBtn.title = 'Follow robot / Free browse';
    this._followBtn.addEventListener('click', () => {
      this._followRobot = !this._followRobot;
      this._updateFollowBtn();
      if (this._followRobot && this._viewMode === '2d') this._fitTopDownView();
    });
    this._el.appendChild(this._followBtn);
    this._updateFollowBtn();

    // Toggle between an overhead planar projection and the original 3D view.
    this._viewBtn = document.createElement('button');
    this._viewBtn.style.cssText =
      'position:absolute;top:8px;right:44px;z-index:10;' +
      'min-width:32px;height:28px;border-radius:4px;border:1px solid rgba(255,255,255,0.3);' +
      'background:rgba(45,94,155,0.82);color:#fff;font:11px monospace;cursor:pointer;padding:0 5px';
    this._viewBtn.addEventListener('click', () => {
      this._setViewMode(this._viewMode === '3d' ? '2d' : '3d');
    });
    this._el.appendChild(this._viewBtn);
    this._updateViewBtn();

    // Resize observer
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);

    // Render loop
    const animate = () => {
      this._raf = requestAnimationFrame(animate);
      // Smooth follow robot
      if (this._followRobot && this._viewMode === '3d') {
        this._controls.target.lerp(this._robotPos, 0.05);
      }
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
    };
    animate();
  },

  _updateFollowBtn() {
    if (!this._followBtn) return;
    this._followBtn.style.background = this._followRobot
      ? 'rgba(77,219,106,0.8)'
      : 'rgba(100,100,100,0.6)';
  },

  _updateViewBtn() {
    if (!this._viewBtn) return;
    this._viewBtn.textContent = this._viewMode.toUpperCase();
    this._viewBtn.title = this._viewMode === '2d'
      ? '当前为二维俯视图，点击切换到 3D'
      : '当前为三维视图，点击切换到 2D';
  },

  _setViewMode(mode) {
    if (!this._camera || !this._controls || mode === this._viewMode) return;
    if (mode === '2d') {
      this._saved3dPosition = this._camera.position.clone();
      this._saved3dTarget = this._controls.target.clone();
      this._viewMode = '2d';
      this._controls.enableRotate = false;
      this._fitTopDownView();
    } else {
      this._viewMode = '3d';
      this._camera.up.set(0, 1, 0);
      this._camera.position.copy(
        this._saved3dPosition || new THREE.Vector3(5, 8, 5)
      );
      this._controls.target.copy(
        this._saved3dTarget || this._robotPos
      );
      this._controls.enableRotate = true;
      this._camera.lookAt(this._controls.target);
      this._camera.updateProjectionMatrix();
      this._controls.update();
    }
    this._updateViewBtn();
  },

  _fitTopDownView() {
    if (this._viewMode !== '2d' || !this._camera || !this._controls || !this._el) return;
    const bounds = [this._mapBounds, this._pathBounds].filter(Boolean);
    const minX = bounds.length ? Math.min(...bounds.map((item) => item.minX)) : this._robotPos.x - 2;
    const maxX = bounds.length ? Math.max(...bounds.map((item) => item.maxX)) : this._robotPos.x + 2;
    const minZ = bounds.length ? Math.min(...bounds.map((item) => item.minZ)) : this._robotPos.z - 2;
    const maxZ = bounds.length ? Math.max(...bounds.map((item) => item.maxZ)) : this._robotPos.z + 2;
    const centerX = (minX + maxX) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const aspect = Math.max(this._el.clientWidth || 400, 1) / Math.max(this._el.clientHeight || 300, 1);
    const verticalSpan = Math.max(maxZ - minZ, (maxX - minX) / aspect, 1.0) * 1.18;
    const distance = Math.min(
      450,
      Math.max(5, verticalSpan / (2 * Math.tan(THREE.MathUtils.degToRad(this._camera.fov / 2))))
    );

    // Map +Y is represented by Three.js -Z, therefore -Z is screen-up.
    this._camera.up.set(0, 0, -1);
    this._camera.position.set(centerX, distance, centerZ);
    this._controls.target.set(centerX, 0, centerZ);
    this._camera.lookAt(this._controls.target);
    this._camera.updateProjectionMatrix();
    this._controls.update();
  },

  _resize() {
    if (!this._el || !this._renderer) return;
    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
    if (this._viewMode === '2d' && this._followRobot) this._fitTopDownView();
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

      // Heuristic: new protocol has flags with bit1 (has_z) set
      if ((possibleFlags & 0x02) !== 0) {
        // New protocol (has_z flag set)
        robotX = view.getFloat32(0, true);
        robotY = view.getFloat32(4, true);
        robotYaw = view.getFloat32(8, true);
        flags = possibleFlags;
        numPoints = possibleNum;
        headerSize = 17;
        hasZ = true;
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
    let mapMinX = Infinity, mapMaxX = -Infinity;
    let mapMinZ = Infinity, mapMaxZ = -Infinity;

    for (let i = 0; i < count; i++) {
      const off = headerSize + i * bytesPerPoint;
      const x = view.getFloat32(off, true);
      const y = view.getFloat32(off + 4, true);
      const z = hasZ ? view.getFloat32(off + 8, true) : 0;

      const idx = i * 3;
      // Map coordinate: x→x(right), z→y(up), -y→z(into screen)
      pos[idx] = x;
      pos[idx + 1] = z;
      pos[idx + 2] = -y;

      mapMinX = Math.min(mapMinX, x);
      mapMaxX = Math.max(mapMaxX, x);
      mapMinZ = Math.min(mapMinZ, -y);
      mapMaxZ = Math.max(mapMaxZ, -y);

      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    }

    // Rainbow height colormap
    const col = this._colors;
    const zRange = zMax - zMin;
    const zScale = zRange > 0.01 ? 1.0 / zRange : 1.0;

    for (let i = 0; i < count; i++) {
      const z = pos[i * 3 + 1]; // y in Three.js = height
      const t = hasZ ? (z - zMin) * zScale : 0.5;
      const idx = i * 3;
      col[idx] = this._rainbowR(t);
      col[idx + 1] = this._rainbowG(t);
      col[idx + 2] = this._rainbowB(t);
    }

    // Update geometry
    const geo = this._points.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, count);
    this._mapBounds = count ? {
      minX: mapMinX, maxX: mapMaxX, minZ: mapMinZ, maxZ: mapMaxZ,
    } : null;

    // Update robot position and orientation
    // Coordinate mapping: robot at (robotX, 0.2, -robotY)
    // Yaw: in the ROS map frame yaw=0 is +X and yaw=pi/2 is +Y.
    // The display maps ROS +Y to Three.js -Z. A positive Three.js rotation
    // around +Y already turns +X toward -Z, so the yaw sign is preserved.
    if (this._robotMesh) {
      this._robotMesh.position.set(robotX, 0.2, -robotY);
      this._robotMesh.rotation.set(0, robotYaw, 0);
      this._robotPos.set(robotX, 0.2, -robotY);
    }
    if (this._viewMode === '2d' && this._followRobot) this._fitTopDownView();
  },

  onDataSilent(buffer) {
    this.onData(buffer);
  },

  _connectPlanStream() {
    if (!this._el || this._planWs) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/bus/plan`);
    ws.binaryType = 'arraybuffer';
    this._planWs = ws;
    ws.onmessage = (event) => {
      if (this._planWs !== ws) return;
      this._onPlanData(event.data);
    };
    ws.onclose = () => {
      if (this._planWs !== ws) return;
      this._planWs = null;
      if (this._el) {
        this._planReconnectTimer = setTimeout(() => {
          this._planReconnectTimer = null;
          this._connectPlanStream();
        }, 5000);
      }
    };
    ws.onerror = () => {};
  },

  _onPlanData(raw) {
    let data;
    try {
      const text = raw instanceof ArrayBuffer
        ? new TextDecoder().decode(raw)
        : String(raw);
      data = JSON.parse(text);
    } catch {
      return;
    }
    if (data?.type === 'ping' || data?.type === 'meta' || data?.type === 'error') return;
    if (!Array.isArray(data?.poses)) return;

    if (data.frame_id !== 'map') {
      this._clearPlan(`路径 frame=${data.frame_id || '—'}，需要 map`);
      return;
    }

    const poses = data.poses.filter((pose) =>
      Number.isFinite(Number(pose?.x)) && Number.isFinite(Number(pose?.y))
    );
    if (!poses.length) {
      this._clearPlan('当前没有规划路径');
      return;
    }

    let length = 0;
    for (let i = 1; i < poses.length; i++) {
      length += Math.hypot(
        Number(poses[i].x) - Number(poses[i - 1].x),
        Number(poses[i].y) - Number(poses[i - 1].y),
      );
    }

    const stride = Math.max(1, Math.ceil(poses.length / MAX_PATH_POSES));
    const sampled = [];
    for (let i = 0; i < poses.length; i += stride) sampled.push(poses[i]);
    if (sampled[sampled.length - 1] !== poses[poses.length - 1]) {
      sampled.push(poses[poses.length - 1]);
    }

    const count = Math.min(sampled.length, MAX_PATH_POSES);
    let pathMinX = Infinity, pathMaxX = -Infinity;
    let pathMinZ = Infinity, pathMaxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const pose = sampled[i];
      const index = i * 3;
      this._pathPositions[index] = Number(pose.x);
      this._pathPositions[index + 1] = 0.12;
      this._pathPositions[index + 2] = -Number(pose.y);
      pathMinX = Math.min(pathMinX, Number(pose.x));
      pathMaxX = Math.max(pathMaxX, Number(pose.x));
      pathMinZ = Math.min(pathMinZ, -Number(pose.y));
      pathMaxZ = Math.max(pathMaxZ, -Number(pose.y));
    }
    const attribute = this._pathLine?.geometry?.attributes?.position;
    if (attribute) attribute.needsUpdate = true;
    this._pathLine?.geometry?.setDrawRange(0, count);
    if (this._pathLine) this._pathLine.visible = count >= 2;
    this._pathBounds = count ? {
      minX: pathMinX, maxX: pathMaxX, minZ: pathMinZ, maxZ: pathMaxZ,
    } : null;

    const goal = poses[poses.length - 1];
    if (this._goalMesh) {
      this._goalMesh.position.set(Number(goal.x), 0.12, -Number(goal.y));
      this._goalMesh.visible = true;
    }
    if (this._pathSummary) {
      this._pathSummary.textContent = `PATH  ${poses.length} poses  ${length.toFixed(2)} m`;
    }
    if (this._viewMode === '2d' && this._followRobot) this._fitTopDownView();
  },

  _clearPlan(message) {
    this._pathLine?.geometry?.setDrawRange(0, 0);
    if (this._pathLine) this._pathLine.visible = false;
    if (this._goalMesh) this._goalMesh.visible = false;
    this._pathBounds = null;
    if (this._pathSummary) this._pathSummary.textContent = message;
    if (this._viewMode === '2d' && this._followRobot) this._fitTopDownView();
  },

  // Rainbow colormap: 0=red, 0.25=yellow, 0.5=green, 0.75=cyan/blue, 1.0=purple
  _rainbowR(t) {
    if (t < 0.25) return 1.0;
    if (t < 0.5) return 1.0 - (t - 0.25) * 4;
    if (t < 0.75) return 0.0;
    return (t - 0.75) * 4 * 0.7;
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
    if (this._planReconnectTimer) clearTimeout(this._planReconnectTimer);
    this._planReconnectTimer = null;
    const planWs = this._planWs;
    this._planWs = null;
    planWs?.close();
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._controls?.dispose();
    this._points?.geometry?.dispose();
    this._points?.material?.dispose();
    this._robotMesh?.geometry?.dispose();
    this._robotMesh?.material?.dispose();
    this._pathLine?.geometry?.dispose();
    this._pathLine?.material?.dispose();
    this._goalMesh?.geometry?.dispose();
    this._goalMesh?.material?.dispose();
    this._renderer?.dispose();
    this._followBtn?.remove();
    this._viewBtn?.remove();
    this._el?.remove();
    this._el = null;
    this._renderer = null;
    this._scene = null;
    this._camera = null;
    this._controls = null;
    this._points = null;
    this._robotMesh = null;
    this._pathLine = null;
    this._pathPositions = null;
    this._goalMesh = null;
    this._pathSummary = null;
    this._planWs = null;
    this._viewBtn = null;
    this._saved3dPosition = null;
    this._saved3dTarget = null;
    this._mapBounds = null;
    this._pathBounds = null;
    this._positions = null;
    this._colors = null;
    this._followBtn = null;
  },
};
