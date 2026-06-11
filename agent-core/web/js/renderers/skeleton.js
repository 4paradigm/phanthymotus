/**
 * skeleton.js — 3D skeleton renderer for Unitree G1 humanoid robot.
 * Fetches URDF model from driver via MCP tool, parses kinematic chain,
 * and renders a real-time 3D skeleton driven by joint state data.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Motor index → URDF joint name mapping (G1 43 DOF: 29 body + 14 hand)
const MOTOR_INDEX_MAP = [
  // 0-5: left leg
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
  'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  // 6-11: right leg
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
  // 12-14: waist
  'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
  // 15-21: left arm (shoulder, elbow, wrist)
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
  'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  // 22-28: right arm
  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
  'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
  // 29-35: left hand (thumb ×3, middle ×2, index ×2)
  'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
  'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
  'left_hand_index_0_joint', 'left_hand_index_1_joint',
  // 36-42: right hand (thumb ×3, middle ×2, index ×2)
  'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
  'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
  'right_hand_index_0_joint', 'right_hand_index_1_joint',
];

export const SkeletonRenderer = {
  name: 'skeleton',
  canRender: (hint) => hint === 'sensor/skeleton',

  _el: null,
  _scene: null,
  _camera: null,
  _threeRenderer: null,
  _controls: null,
  _raf: null,
  _ro: null,
  _joints: {},       // jointName → THREE.Object3D
  _links: {},        // linkName → THREE.Object3D
  _rootGroup: null,  // Z-up → Y-up coordinate transform
  _pelvisGroup: null, // IMU orientation applied here
  _loaded: false,
  _mcpId: null,

  mount(container, mcpId) {
    this._mcpId = mcpId;
    this._joints = {};
    this._links = {};
    this._loaded = false;
    this._el = document.createElement('div');
    this._el.className = 'renderer-skeleton';
    container.appendChild(this._el);

    this._initThree();
    this._loadModel(mcpId);
    this._animate();

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this._el);
  },

  _initThree() {
    const w = this._el.clientWidth || 400;
    const h = this._el.clientHeight || 400;

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x1c1c1e);

    this._camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 50);
    this._camera.position.set(1.2, 0.8, 1.5);
    this._camera.lookAt(0, 0.4, 0);

    this._threeRenderer = new THREE.WebGLRenderer({ antialias: true });
    this._threeRenderer.setSize(w, h);
    this._threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._el.appendChild(this._threeRenderer.domElement);

    this._controls = new OrbitControls(this._camera, this._threeRenderer.domElement);
    this._controls.target.set(0, 0.4, 0);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.1;
    this._controls.update();

    // Lighting
    this._scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 3, 2);
    this._scene.add(dirLight);

    // Ground grid
    const grid = new THREE.GridHelper(2, 20, 0x444444, 0x333333);
    grid.position.y = -0.65;
    this._scene.add(grid);
  },

  async _loadModel(mcpId) {
    if (!mcpId || mcpId === 'dashboard') {
      this._buildFallbackSkeleton();
      return;
    }
    try {
      const res = await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: 'model', arguments: {} }),
      });
      const json = await res.json();
      // API returns { code, data } where data is the content array
      const content = json.data?.[0]?.text ?? json.result?.content?.[0]?.text;
      if (!content) { this._buildFallbackSkeleton(); return; }
      const data = JSON.parse(content);
      if (data.urdf) {
        this._parseAndBuild(data.urdf);
      } else {
        this._buildFallbackSkeleton();
      }
    } catch (e) {
      console.warn('[skeleton] Failed to load URDF model:', e);
      this._buildFallbackSkeleton();
    }
  },

  _parseAndBuild(urdfXml) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(urdfXml, 'text/xml');

    // Parse links and joints
    const linkNodes = doc.querySelectorAll('link');
    const jointNodes = doc.querySelectorAll('joint');

    // Build link map
    const linkMap = {};
    for (const ln of linkNodes) {
      const name = ln.getAttribute('name');
      linkMap[name] = { name, children: [] };
    }

    // Parse joints and build hierarchy
    const joints = [];
    for (const jn of jointNodes) {
      const name = jn.getAttribute('name');
      const type = jn.getAttribute('type');
      const parentLink = jn.querySelector('parent')?.getAttribute('link');
      const childLink = jn.querySelector('child')?.getAttribute('link');
      const originEl = jn.querySelector('origin');
      const axisEl = jn.querySelector('axis');

      let origin = { x: 0, y: 0, z: 0 };
      let rpy = { x: 0, y: 0, z: 0 };
      if (originEl) {
        const xyz = (originEl.getAttribute('xyz') || '0 0 0').split(' ').map(Number);
        const rpyArr = (originEl.getAttribute('rpy') || '0 0 0').split(' ').map(Number);
        origin = { x: xyz[0], y: xyz[1], z: xyz[2] };
        rpy = { x: rpyArr[0], y: rpyArr[1], z: rpyArr[2] };
      }

      let axis = { x: 0, y: 0, z: 1 };
      if (axisEl) {
        const a = (axisEl.getAttribute('xyz') || '0 0 1').split(' ').map(Number);
        axis = { x: a[0], y: a[1], z: a[2] };
      }

      joints.push({ name, type, parentLink, childLink, origin, rpy, axis });
      if (linkMap[parentLink]) {
        linkMap[parentLink].children.push({ joint: name, childLink });
      }
    }

    // Build Three.js hierarchy from pelvis root
    const rootGroup = new THREE.Group();
    // URDF uses Z-up, Three.js uses Y-up — rotate the root
    rootGroup.rotation.x = -Math.PI / 2;
    this._scene.add(rootGroup);
    this._rootGroup = rootGroup;

    // Pelvis orientation group (IMU quaternion applied here)
    const pelvisGroup = new THREE.Group();
    rootGroup.add(pelvisGroup);
    this._pelvisGroup = pelvisGroup;

    const linkObj = {};
    linkObj['pelvis'] = pelvisGroup;
    this._links['pelvis'] = pelvisGroup;

    // BFS to build hierarchy
    const queue = ['pelvis'];
    const visited = new Set(['pelvis']);

    while (queue.length > 0) {
      const current = queue.shift();
      const parentObj = linkObj[current];

      for (const j of joints) {
        if (j.parentLink !== current || visited.has(j.childLink)) continue;
        visited.add(j.childLink);

        // Create joint group at the origin offset
        const jointGroup = new THREE.Group();
        jointGroup.position.set(j.origin.x, j.origin.y, j.origin.z);

        // Apply base RPY rotation (static offset)
        const euler = new THREE.Euler(j.rpy.x, j.rpy.y, j.rpy.z, 'XYZ');
        jointGroup.rotation.copy(euler);

        parentObj.add(jointGroup);

        // Create child link group (rotation applied here for revolute joints)
        const childGroup = new THREE.Group();
        childGroup.userData = { axis: new THREE.Vector3(j.axis.x, j.axis.y, j.axis.z) };
        jointGroup.add(childGroup);

        // Store references
        if (j.type === 'revolute') {
          this._joints[j.name] = childGroup;
        }
        linkObj[j.childLink] = childGroup;
        this._links[j.childLink] = childGroup;

        // Add visual geometry (bone segment) — only for revolute joints
        if (j.type === 'revolute') {
          this._addBoneVisual(jointGroup, j);
        }

        queue.push(j.childLink);
      }
    }

    // Add joint spheres
    for (const [name, obj] of Object.entries(this._joints)) {
      const isHand = name.includes('hand');
      const radius = isHand ? 0.006 : 0.012;
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 8, 8),
        new THREE.MeshPhongMaterial({ color: 0x00ccff, emissive: 0x003344 })
      );
      obj.add(sphere);
    }

    this._loaded = true;
  },

  _addBoneVisual(jointGroup, joint) {
    // Draw a cylinder from parent link origin to this joint's origin
    const len = Math.sqrt(
      joint.origin.x ** 2 + joint.origin.y ** 2 + joint.origin.z ** 2
    );
    if (len < 0.005) return; // too short, skip

    const isHand = joint.name.includes('hand') || joint.name.includes('thumb') ||
                   joint.name.includes('index') || joint.name.includes('middle');
    const radius = isHand ? 0.004 : 0.008;
    const color = isHand ? 0x668899 : 0xaabbcc;

    const geom = new THREE.CylinderGeometry(radius, radius, len, 6);
    geom.translate(0, len / 2, 0); // bottom at origin, top at (0, len, 0)

    const bone = new THREE.Mesh(geom, new THREE.MeshPhongMaterial({ color }));

    // Rotate default Y-axis to point toward joint origin direction
    const dir = new THREE.Vector3(joint.origin.x, joint.origin.y, joint.origin.z).normalize();
    bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    // Add to parent link (static linkage, doesn't rotate with joint)
    jointGroup.parent?.add(bone);
  },

  _buildFallbackSkeleton() {
    // Humanoid stick figure based on G1 dimensions when URDF not available
    const boneMat = new THREE.MeshPhongMaterial({ color: 0xaabbcc });
    const jointMat = new THREE.MeshPhongMaterial({ color: 0x00ccff, emissive: 0x003344 });
    const handMat = new THREE.MeshPhongMaterial({ color: 0x668899 });

    const root = new THREE.Group();
    this._scene.add(root);

    const addJoint = (pos, r = 0.012) => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(r, 8, 8), jointMat);
      s.position.copy(pos);
      root.add(s);
      return s;
    };
    const addBone = (from, to, r = 0.008) => {
      const dir = new THREE.Vector3().subVectors(to, from);
      const len = dir.length();
      if (len < 0.005) return;
      const geom = new THREE.CylinderGeometry(r, r, len, 6);
      geom.translate(0, len / 2, 0);
      const bone = new THREE.Mesh(geom, r < 0.006 ? handMat : boneMat);
      bone.position.copy(from);
      bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      root.add(bone);
    };

    // Key positions (Y-up, approximate G1 proportions)
    const pelvis = new THREE.Vector3(0, 0.75, 0);
    const waist = new THREE.Vector3(0, 0.80, 0);
    const torso = new THREE.Vector3(0, 1.05, 0);
    const head = new THREE.Vector3(0, 1.15, 0);

    // Left leg
    const lHip = new THREE.Vector3(0.065, 0.65, 0);
    const lKnee = new THREE.Vector3(0.065, 0.40, 0);
    const lAnkle = new THREE.Vector3(0.065, 0.10, 0);
    const lFoot = new THREE.Vector3(0.065, 0.08, 0);

    // Right leg
    const rHip = new THREE.Vector3(-0.065, 0.65, 0);
    const rKnee = new THREE.Vector3(-0.065, 0.40, 0);
    const rAnkle = new THREE.Vector3(-0.065, 0.10, 0);
    const rFoot = new THREE.Vector3(-0.065, 0.08, 0);

    // Left arm
    const lShoulder = new THREE.Vector3(0.10, 1.05, 0);
    const lElbow = new THREE.Vector3(0.10, 0.87, 0);
    const lWrist = new THREE.Vector3(0.10, 0.72, 0);
    const lPalm = new THREE.Vector3(0.10, 0.68, 0);

    // Right arm
    const rShoulder = new THREE.Vector3(-0.10, 1.05, 0);
    const rElbow = new THREE.Vector3(-0.10, 0.87, 0);
    const rWrist = new THREE.Vector3(-0.10, 0.72, 0);
    const rPalm = new THREE.Vector3(-0.10, 0.68, 0);

    // Draw bones
    addBone(pelvis, waist);
    addBone(waist, torso);
    addBone(torso, head);

    // Left leg
    addBone(pelvis, lHip);
    addBone(lHip, lKnee);
    addBone(lKnee, lAnkle);
    addBone(lAnkle, lFoot);

    // Right leg
    addBone(pelvis, rHip);
    addBone(rHip, rKnee);
    addBone(rKnee, rAnkle);
    addBone(rAnkle, rFoot);

    // Left arm
    addBone(torso, lShoulder);
    addBone(lShoulder, lElbow);
    addBone(lElbow, lWrist);
    addBone(lWrist, lPalm, 0.005);

    // Right arm
    addBone(torso, rShoulder);
    addBone(rShoulder, rElbow);
    addBone(rElbow, rWrist);
    addBone(rWrist, rPalm, 0.005);

    // Finger stubs (left)
    const lF1 = new THREE.Vector3(0.10, 0.65, 0.012);
    const lF2 = new THREE.Vector3(0.10, 0.65, 0);
    const lF3 = new THREE.Vector3(0.10, 0.65, -0.012);
    addBone(lPalm, lF1, 0.003); addBone(lPalm, lF2, 0.003); addBone(lPalm, lF3, 0.003);

    // Finger stubs (right)
    const rF1 = new THREE.Vector3(-0.10, 0.65, 0.012);
    const rF2 = new THREE.Vector3(-0.10, 0.65, 0);
    const rF3 = new THREE.Vector3(-0.10, 0.65, -0.012);
    addBone(rPalm, rF1, 0.003); addBone(rPalm, rF2, 0.003); addBone(rPalm, rF3, 0.003);

    // Draw joints at key positions
    addJoint(pelvis, 0.02);
    addJoint(head, 0.04);
    [lHip, lKnee, lAnkle, rHip, rKnee, rAnkle].forEach(p => addJoint(p));
    [waist, torso].forEach(p => addJoint(p, 0.015));
    [lShoulder, lElbow, lWrist, rShoulder, rElbow, rWrist].forEach(p => addJoint(p));
    [lPalm, rPalm].forEach(p => addJoint(p, 0.008));

    this._loaded = true;
  },

  onData(buffer, fmt) {
    if (!this._loaded) return;
    try {
      const text = new TextDecoder().decode(buffer);
      const data = JSON.parse(text);

      // Apply pelvis orientation from IMU quaternion
      if (data.imu_quat && this._pelvisGroup) {
        const [w, x, y, z] = data.imu_quat;
        this._pelvisGroup.quaternion.set(x, y, z, w);
      }

      const joints = data.joints;
      if (!Array.isArray(joints)) return;

      for (const j of joints) {
        const idx = j.idx;
        if (idx >= MOTOR_INDEX_MAP.length) continue;
        const jointName = MOTOR_INDEX_MAP[idx];
        const obj = this._joints[jointName];
        if (!obj) continue;

        const angle = j.q;
        const axis = obj.userData.axis;
        if (!axis) continue;

        // Apply rotation around the joint axis from identity
        obj.quaternion.setFromAxisAngle(axis, angle);
      }
    } catch (e) {
      // Silently ignore parse errors (graceful degradation)
    }
  },

  _animate() {
    if (!this._threeRenderer) return;
    this._raf = requestAnimationFrame(() => this._animate());
    this._controls?.update();
    this._threeRenderer.render(this._scene, this._camera);
  },

  _resize() {
    if (!this._el || !this._threeRenderer) return;
    const w = this._el.clientWidth;
    const h = this._el.clientHeight;
    if (w === 0 || h === 0) return;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._threeRenderer.setSize(w, h);
  },

  unmount() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._ro?.disconnect();
    this._controls?.dispose();
    if (this._threeRenderer) {
      this._threeRenderer.dispose();
      this._threeRenderer.domElement?.remove();
    }
    if (this._scene) {
      this._scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
      });
    }
    this._el?.remove();
    this._scene = null;
    this._camera = null;
    this._threeRenderer = null;
    this._controls = null;
    this._joints = {};
    this._links = {};
    this._el = null;
    this._raf = null;
    this._ro = null;
    this._loaded = false;
  },
};
