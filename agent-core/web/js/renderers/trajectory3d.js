/** Live Go1 swing-foot paths. ROS X-forward/Y-left/Z-up → Three X-right/Y-up/Z-toward-viewer. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const FEET = ['FR', 'FL', 'RR', 'RL'];
const COLORS = { FR: 0xff6b6b, FL: 0x62b6ff, RR: 0xffcc66, RL: 0x72df9a };

function point(value) {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) return null;
  return new THREE.Vector3(-value[1], value[2], -value[0]);
}

export const Trajectory3DRenderer = {
  name: 'trajectory3d',
  canRender: hint => hint === 'sensor/trajectory3d',

  mount(container) {
    this._frame = 'body';
    this._data = null;
    this._objects = [];
    this._autoFit = true;
    this._el = document.createElement('div');
    this._el.style.cssText = 'width:100%;height:100%;min-height:160px;position:relative;overflow:hidden;background:#1c1c1e';
    container.appendChild(this._el);
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x1c1c1e);
    const width = this._el.clientWidth || 400;
    const height = this._el.clientHeight || 250;
    this._camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 100);
    this._camera.position.set(0.8, 0.8, 1.1);
    this._camera.lookAt(0, 0, 0);
    this._renderer = new THREE.WebGLRenderer({ antialias: true });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.setSize(width, height);
    this._el.appendChild(this._renderer.domElement);
    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.addEventListener('start', () => { this._autoFit = false; });
    this._scene.add(new THREE.GridHelper(2, 20, 0x444444, 0x333333));
    this._scene.add(new THREE.AxesHelper(0.25));

    const panel = document.createElement('div');
    panel.style.cssText = 'position:absolute;top:8px;left:8px;color:white;background:#161619dd;padding:6px 8px;border-radius:8px;font:11px system-ui;z-index:1';
    panel.innerHTML = '<div style="margin-bottom:5px">摆动腿轨迹 · X前 Y左 Z上</div>';
    for (const [frame, label] of [['body', '机身'], ['world', '世界估计']]) {
      const button = document.createElement('button');
      button.textContent = label;
      button.dataset.frame = frame;
      button.style.cssText = 'margin-right:4px;padding:2px 5px;cursor:pointer';
      button.addEventListener('click', () => {
        this._frame = frame;
        this._autoFit = true;
        for (const b of panel.querySelectorAll('button')) b.style.fontWeight = b === button ? 'bold' : 'normal';
        this._draw();
      });
      if (frame === 'body') button.style.fontWeight = 'bold';
      panel.appendChild(button);
    }
    this._status = document.createElement('div');
    this._status.style.marginTop = '5px';
    panel.appendChild(this._status);
    const legend = document.createElement('div');
    legend.style.marginTop = '4px';
    for (const name of FEET) {
      const item = document.createElement('span');
      item.textContent = `${name} `;
      item.style.color = `#${COLORS[name].toString(16).padStart(6, '0')}`;
      legend.appendChild(item);
    }
    panel.appendChild(legend);
    this._el.appendChild(panel);

    this._ro = new ResizeObserver(() => {
      if (!this._el || !this._renderer) return;
      const w = this._el.clientWidth || 400;
      const h = this._el.clientHeight || 250;
      this._camera.aspect = w / h;
      this._camera.updateProjectionMatrix();
      this._renderer.setSize(w, h);
    });
    this._ro.observe(this._el);
    const animate = () => {
      this._raf = requestAnimationFrame(animate);
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
    };
    animate();
  },

  onData(buffer) {
    try {
      const data = JSON.parse(new TextDecoder().decode(buffer));
      if (!data.feet || typeof data.feet !== 'object') return;
      this._data = data;
      this._draw();
    } catch { /* malformed sample: retain last valid display */ }
  },

  _clearPaths() {
    for (const object of this._objects) {
      this._scene.remove(object);
      object.geometry.dispose();
      object.material.dispose();
    }
    this._objects = [];
  },

  _draw() {
    if (!this._scene) return;
    this._clearPaths();
    const data = this._data;
    if (!data) return;
    const key = this._frame === 'world' ? 'world_xyz_m' : 'body_xyz_m';
    let visible = 0;
    const allPoints = [];
    for (const name of FEET) {
      const foot = data.feet[name] || {};
      for (const [field, opacity] of [['last_completed', 0.35], ['active', 1]]) {
        const values = Array.isArray(foot[field]) ? foot[field] : [];
        // Null world points break the line; never bridge over unavailable pose.
        let segment = [];
        const flush = () => {
          if (segment.length < 2) { segment = []; return; }
          const geo = new THREE.BufferGeometry().setFromPoints(segment);
          const mat = new THREE.LineBasicMaterial({ color: COLORS[name], transparent: true, opacity });
          const line = new THREE.Line(geo, mat);
          this._scene.add(line);
          this._objects.push(line);
          visible += segment.length;
          segment = [];
        };
        for (const sample of values) {
          const xyz = point(sample?.[key]);
          if (xyz) { segment.push(xyz); allPoints.push(xyz); } else flush();
        }
        flush();
        if (field === 'active' && segment.length === 0 && values.length === 1) {
          const xyz = point(values[0]?.[key]);
          if (xyz) {
            const geo = new THREE.SphereGeometry(0.012, 8, 6);
            const mat = new THREE.MeshBasicMaterial({ color: COLORS[name] });
            const marker = new THREE.Mesh(geo, mat);
            marker.position.copy(xyz);
            allPoints.push(xyz);
            this._scene.add(marker);
            this._objects.push(marker);
            visible += 1;
          }
        }
      }
    }
    if (this._autoFit && allPoints.length) {
      const box = new THREE.Box3().setFromPoints(allPoints);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const distance = Math.max(0.45, size.length() * 1.8);
      this._controls.target.copy(center);
      this._camera.position.copy(center).add(new THREE.Vector3(distance, distance * 0.8, distance));
      this._controls.update();
    }
    this._status.textContent = `${data.status || '等待数据'} · ${this._frame === 'world' ? '里程计估计' : '机身坐标'} · ${visible} 点`;
  },

  onDataSilent(buffer) { this.onData(buffer); },

  unmount() {
    this._ro?.disconnect();
    if (this._raf) cancelAnimationFrame(this._raf);
    this._clearPaths();
    this._controls?.dispose();
    this._renderer?.dispose();
    this._el?.remove();
    this._el = null;
    this._scene = null;
    this._renderer = null;
  },
};
