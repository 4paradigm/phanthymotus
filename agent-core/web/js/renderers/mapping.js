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
 * Supports free browsing (pan/rotate/zoom) and follow-robot mode toggle.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { decodeSlamPreview, currentOdomMarker, MAX_TRAJECTORY_POINTS, writeTrajectoryPositions } from './slam-preview-codec.js';

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
  _followBtn: null,
  _followRobot: true,
  _robotPos: new THREE.Vector3(0, 0.2, 0),

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
    });
    this._el.appendChild(this._followBtn);
    this._updateFollowBtn();

    // Resize observer
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);

    // Render loop
    const animate = () => {
      this._raf = requestAnimationFrame(animate);
      // Smooth follow robot
      if (this._followRobot) {
        if (this._camera.isOrthographicCamera) {
          this._camera.position.addScaledVector(this._followDelta.subVectors(this._robotPos, this._controls.target), 0.05);
        }
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

    // Update robot position and orientation
    // Coordinate mapping: robot at (robotX, 0.2, -robotY)
    // Yaw: in original frame yaw=0 is +X, yaw=pi/2 is +Y
    // In Three.js: +X stays +X, +Y maps to -Z → rotation around Y axis = -yaw
    if (this._robotMesh) {
      this._robotMesh.position.set(robotX, 0.2, -robotY);
      this._robotMesh.rotation.set(0, -robotYaw, 0);
      this._robotPos.set(robotX, 0.2, -robotY);
    }
  },

  onDataSilent(buffer) {
    this.onData(buffer);
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
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._controls?.dispose();
    this._points?.geometry?.dispose();
    this._points?.material?.dispose();
    this._robotMesh?.geometry?.dispose();
    this._robotMesh?.material?.dispose();
    this._renderer?.dispose();
    this._followBtn?.remove();
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
    this._followBtn = null;
  },
};

// Keyframe poses in the binary map payload are retained for older readers.
// Only the separately timestamped odometry estimate can drive the live marker.
export const SlamPreviewRenderer = {
  ...MappingRenderer,
  name: 'slam-preview',
  canRender: (hint) => hint === 'sensor/slam-preview',
  mount(container, {toolbar, scope = 'live'} = {}) {
    MappingRenderer.mount.call(this, container);
    this._scope = scope;
    this._el.classList.add('renderer-slam-preview');
    this._views = new Map();
    this._mode = null;
    this._overview = true;
    this._followRobot = false;
    this._robotMesh.visible = false;
    this._robotPos = new THREE.Vector3();
    this._followDelta = new THREE.Vector3();
    this._points.frustumCulled = false;
    const path = new THREE.BufferGeometry();
    path.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_TRAJECTORY_POINTS * 3), 3));
    path.setDrawRange(0, 0);
    this._trajectory = new THREE.Line(path, new THREE.LineBasicMaterial({color: 0xffffff, depthTest: false}));
    this._trajectory.frustumCulled = false;
    this._trajectory.renderOrder = 1;
    this._scene.add(this._trajectory);
    this._tagGroup = new THREE.Group();
    this._scene.add(this._tagGroup);
    this._buttons = document.createElement('span');
    this._buttons.className = 'monitor-card-modes slam-view-modes';
    toolbar?.replaceChildren(this._buttons);
    const button = (label, title, action) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'monitor-card-mode-btn';
      btn.textContent = label;
      btn.title = title;
      btn.onclick = action;
      this._buttons.append(btn);
      return btn;
    };
    this._modeButtons = {};
    for (const mode of ['2D', '3D']) {
      this._modeButtons[mode] = button(mode, mode === '2D' ? '俯视地图' : '三维地图',
        () => this.setViewMode(mode));
    }
    this._overviewBtn = button('全图', '显示完整地图', () => {
      this._followRobot = false;
      this._overview = true;
      this._updateFollowBtn();
      this._fitOverview();
    });
    this._followBtn.className = 'monitor-card-mode-btn';
    this._followBtn.removeAttribute('style');
    this._followBtn.title = '跟随里程计估计位置';
    this._buttons.append(this._followBtn);
    this._followBtn.style.display = 'none';
    this._followBtn.addEventListener('click', () => { this._overview = false; });
    let mode;
    try { mode = localStorage.getItem('slam-view-' + scope); } catch {}
    if (mode !== '2D' && mode !== '3D') mode = scope === 'history' ? '2D' : '3D';
    this.setViewMode(mode, false);
    const canvas = this._renderer.domElement;
    let start;
    canvas.addEventListener('pointerdown', event => {
      if ((!this._onPick && !this._onTagSelect) || event.button !== 0 || !event.isPrimary) return;
      start = {id: event.pointerId, px: event.clientX, py: event.clientY};
      if (this._onPick) {
        start.point = this._mapPoint(event);
        canvas.setPointerCapture(event.pointerId);
      }
    });
    canvas.addEventListener('pointercancel', () => { start = null; });
    canvas.addEventListener('pointerup', event => {
      if (!start || start.id !== event.pointerId) return;
      const {px, py, point} = start;
      start = null;
      const dragged = Math.hypot(event.clientX - px, event.clientY - py) >= 5;
      if (point && this._onPick) {
        if (dragged) {
          const end = this._mapPoint(event);
          point.yaw = Math.atan2(end.y - point.y, end.x - point.x);
        }
        this._onPick(point);
      } else if (!dragged && this._onTagSelect) {
        const ray = new THREE.Raycaster();
        this._camera.updateMatrixWorld();
        this._tagGroup.updateMatrixWorld(true);
        ray.setFromCamera(this._pointerCoords(event), this._camera);
        const hit = ray.intersectObjects(this._tagGroup.children, false)[0];
        if (hit) this._onTagSelect(hit.object.userData.tag);
      }
    });
  },
  setViewMode(mode, persist = true) {
    if (mode === this._mode) return;
    if (this._mode) {
      this._views.set(this._mode, {camera: this._camera, target: this._controls.target.clone(),
        overview: this._overview, follow: this._followRobot});
    }
    this._mode = mode;
    this._controls.dispose();
    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 300;
    const flat = mode === '2D';
    const saved = this._views.get(mode);
    if (saved) {
      this._camera = saved.camera;
    } else if (flat) {
      this._camera = new THREE.OrthographicCamera(-5 * w / h, 5 * w / h, 5, -5, 0.01, 500);
      this._camera.up.set(0, 0, -1);
      this._camera.position.set(0, 50, 0);
    } else {
      this._camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 500);
      this._camera.position.set(5, 8, 5);
    }
    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    if (saved) this._controls.target.copy(saved.target);
    this._controls.update();
    this._controls.enableRotate = !flat;
    if (flat) {
      this._controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
      this._controls.touches.ONE = THREE.TOUCH.PAN;
    }
    this._controls.enabled = !this._onPick;
    this._controls.addEventListener('start', () => {
      this._overview = this._followRobot = false;
      this._updateFollowBtn();
    });
    this._points.material.size = flat ? 2 : 0.03;
    this._overview = saved?.overview ?? true;
    this._followRobot = !!saved?.follow && this._robotMesh.visible;
    this._updateFollowBtn();
    this._resize();
    for (const [key, btn] of Object.entries(this._modeButtons)) {
      btn.classList.toggle('active', mode === key);
      btn.setAttribute('aria-pressed', String(mode === key));
    }
    if (persist) {
      try { localStorage.setItem('slam-view-' + this._scope, mode); } catch {}
    }
  },
  setPickHandler(handler) {
    this._onPick = handler;
    if (handler && !this._camera.isOrthographicCamera) this.setViewMode('2D', false);
    this._controls.enabled = !handler;
    this._modeButtons['3D'].disabled = !!handler;
    this._renderer.domElement.style.cursor = handler ? 'crosshair' : '';
  },
  setTagSelectHandler(handler) {
    this._onTagSelect = handler;
  },
  _pointerCoords(event) {
    const rect = this._renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
      1 - (event.clientY - rect.top) / rect.height * 2);
  },
  _mapPoint(event) {
    const ndc = this._pointerCoords(event);
    this._camera.updateMatrixWorld();
    const p = new THREE.Vector3(ndc.x, ndc.y, 0).unproject(this._camera);
    return {x: p.x, y: -p.z};
  },
  setTags(tags) {
    for (const mesh of [...this._tagGroup.children]) {
      this._tagGroup.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    for (const tag of tags) {
      const mesh = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.45, 4),
        new THREE.MeshBasicMaterial({color: 0xffba46, depthTest: false}));
      if (Number.isFinite(tag.yaw)) mesh.geometry.rotateZ(-Math.PI / 2);
      mesh.position.set(tag.x, 0.12, -tag.y);
      mesh.rotation.y = tag.yaw || 0;
      mesh.renderOrder = 2;
      mesh.userData.tag = tag;
      this._tagGroup.add(mesh);
    }
  },
  _resize() {
    MappingRenderer._resize.call(this);
    if (this._camera.isOrthographicCamera) {
      this._camera.right = this._camera.top * this._camera.aspect;
      this._camera.left = -this._camera.right;
      this._camera.updateProjectionMatrix();
    }
    if (this._overview) this._fitOverview();
  },
  onData(buffer) {
    const decoded = decodeSlamPreview(buffer);
    if (!decoded) return;
    MappingRenderer.onData.call(this, decoded.payload);
    const path = this._trajectory.geometry;
    const attr = path.attributes.position;
    path.setDrawRange(0, writeTrajectoryPositions(decoded.meta.trajectory_xyz, attr.array));
    attr.needsUpdate = true;
    const marker = currentOdomMarker(decoded.meta, Date.now());
    clearTimeout(this._markerTimer);
    this._robotMesh.visible = !!marker;
    this._followBtn.hidden = !marker;
    this._followBtn.style.display = marker ? 'flex' : 'none';
    this._overviewBtn.title = decoded.meta.state === 'historical_snapshot'
      ? '查看完整地图快照' : '查看当前子图全貌';
    if (marker) {
      const [x, y, z] = marker.xyz;
      this._robotPos.set(x, z, -y);
      this._robotMesh.position.copy(this._robotPos);
      this._robotMesh.rotation.set(0, marker.yaw, 0);
      this._markerTimer = setTimeout(() => this._hideMarker(), Math.max(0, marker.expires_ms - Date.now()));
    } else {
      this._hideMarker();
    }
    if (this._overview) this._fitOverview();
  },
  _hideMarker() {
    this._robotMesh.visible = false;
    this._followRobot = false;
    this._followBtn.style.display = 'none';
    this._updateFollowBtn();
  },
  unmount() {
    clearTimeout(this._markerTimer);
    this._trajectory.geometry.dispose();
    this._trajectory.material.dispose();
    this._trajectory = null;
    this.setTags([]);
    this._onPick = this._onTagSelect = null;
    this._views.clear();
    this._buttons.remove();
    this._overviewBtn = null;
    MappingRenderer.unmount.call(this);
  },
  _fitOverview() {
    const count = this._points.geometry.drawRange.count;
    const box = new THREE.Box3().setFromArray(this._positions.subarray(0, count * 3));
    const path = this._trajectory.geometry;
    if (path.drawRange.count) {
      box.union(new THREE.Box3().setFromArray(path.attributes.position.array.subarray(0, path.drawRange.count * 3)));
    }
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    if (this._camera.isOrthographicCamera) {
      const w = this._el.clientWidth || 400;
      const h = this._el.clientHeight || 300;
      const half = Math.max(0.5, (box.max.z - box.min.z) / 2, (box.max.x - box.min.x) * h / (2 * w)) * 1.1;
      Object.assign(this._camera, {left: -half * w / h, right: half * w / h, top: half, bottom: -half,
        far: Math.max(500, box.max.y - box.min.y + 100), zoom: 1});
      this._controls.target.copy(sphere.center);
      this._camera.position.set(sphere.center.x, box.max.y + 50, sphere.center.z);
      this._camera.updateProjectionMatrix();
      this._controls.update();
      return;
    }
    const vertical = THREE.MathUtils.degToRad(this._camera.fov / 2);
    const angle = Math.min(vertical, Math.atan(Math.tan(vertical) * this._camera.aspect));
    const distance = Math.max(1, sphere.radius) * 1.1 / Math.sin(angle);
    const direction = this._camera.position.clone().sub(this._controls.target).normalize();
    this._controls.target.copy(sphere.center);
    this._camera.position.copy(sphere.center).addScaledVector(direction, distance);
    this._camera.far = Math.max(500, distance + sphere.radius * 2);
    this._camera.updateProjectionMatrix();
    this._controls.update();
  },
};
