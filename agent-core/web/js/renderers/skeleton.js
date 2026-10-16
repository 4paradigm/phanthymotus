/**
 * skeleton.js — 3D skeleton renderer using the selected device's model.
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
  _modelGeneration: 0,
  _modelRequest: null,
  _modelNotice: null,

  mount(container, mcpId) {
    this._mcpId = mcpId;
    this._joints = {};
    this._links = {};
    this._loaded = false;
    this._el = document.createElement('div');
    this._el.className = 'renderer-skeleton';
    this._el.style.position = 'relative';
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
    const generation = ++this._modelGeneration;
    this._modelRequest?.abort();
    this._modelRequest = new AbortController();
    this._clearModel();
    this._modelStatus('正在加载机器人模型…');
    if (!mcpId || mcpId === 'dashboard' || mcpId === 'detail') {
      this._modelStatus('无法确定设备，不能加载机器人模型。请启动卡片后重试。');
      return;
    }
    try {
      const res = await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
        method: 'POST',
        signal: this._modelRequest.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: 'model', arguments: {} }),
      });
      if (!res.ok) throw new Error('Model request failed');
      const json = await res.json();
      if (generation !== this._modelGeneration || !this._el) return;
      // API returns { code, data } where data is the content array
      const content = json.data?.[0]?.text ?? json.result?.content?.[0]?.text;
      if (!content) throw new Error('Model response is empty');
      const data = JSON.parse(content);
      if (data.urdf) {
        this._parseAndBuild(data.urdf);
      } else if (data.type === 'quadruped') {
        this._buildQuadrupedSkeleton(data);
      } else {
        throw new Error('Unsupported robot model');
      }
      this._modelStatus(null);
    } catch (e) {
      if (generation !== this._modelGeneration || !this._el) return;
      console.warn('[skeleton] Failed to load URDF model:', e);
      this._clearModel();
      this._modelStatus('机器人模型加载失败。请检查驱动连接和模型资源，然后重新打开。');
    }
  },

  _clearModel() {
    this._loaded = false;
    if (this._rootGroup) {
      this._scene?.remove(this._rootGroup);
      this._rootGroup.traverse(obj => {
        obj.geometry?.dispose();
        if (Array.isArray(obj.material)) obj.material.forEach(material => material.dispose());
        else obj.material?.dispose();
      });
    }
    this._rootGroup = null;
    this._pelvisGroup = null;
    this._joints = {};
    this._links = {};
  },

  _modelStatus(message) {
    if (!this._el) return;
    if (!this._modelNotice) {
      this._modelNotice = document.createElement('div');
      this._modelNotice.className = 'skeleton-model-status';
      this._modelNotice.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;' +
        'justify-content:center;padding:20px;text-align:center;color:var(--orange, #d77757);pointer-events:none';
      this._el.appendChild(this._modelNotice);
    }
    this._modelNotice.textContent = message || '';
    this._modelNotice.style.display = message ? 'flex' : 'none';
  },

  _parseAndBuild(urdfXml) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(urdfXml, 'text/xml');
    if (doc.querySelector('parsererror') || doc.documentElement?.tagName !== 'robot') {
      throw new Error('Invalid robot URDF');
    }

    // Parse links and joints
    const linkNodes = doc.querySelectorAll('link');
    const jointNodes = doc.querySelectorAll('joint');
    if (!linkNodes.length || !jointNodes.length) throw new Error('Robot motion chain is empty');

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

    // Find root link: a link that is a parent but never a child
    const childLinks = new Set(joints.map(j => j.childLink));
    const rootLinkName = Object.keys(linkMap).find(name => !childLinks.has(name)) || 'pelvis';

    linkObj[rootLinkName] = pelvisGroup;
    this._links[rootLinkName] = pelvisGroup;

    // BFS to build hierarchy
    const queue = [rootLinkName];
    const visited = new Set([rootLinkName]);

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

        // Add visual geometry (bone segment) for revolute joints and significant fixed joints
        if (j.type === 'revolute' || j.type === 'fixed') {
          this._addBoneVisual(jointGroup, j);
        }

        queue.push(j.childLink);
      }
    }

    // Add joint spheres for revolute joints
    for (const [name, obj] of Object.entries(this._joints)) {
      const isHand = name.includes('hand');
      const radius = isHand ? 0.006 : 0.012;
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 8, 8),
        new THREE.MeshPhongMaterial({ color: 0x00ccff, emissive: 0x003344 })
      );
      obj.add(sphere);
    }

    // Add endpoint spheres for leaf links (links with no children, e.g. feet)
    const parentLinks = new Set(joints.map(j => j.childLink));
    for (const [name, obj] of Object.entries(this._links)) {
      if (name === rootLinkName) continue;
      const hasChildren = joints.some(j => j.parentLink === name);
      if (!hasChildren) {
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.01, 8, 8),
          new THREE.MeshPhongMaterial({ color: 0x44ddff, emissive: 0x002233 })
        );
        obj.add(sphere);
      }
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

  _buildQuadrupedSkeleton(data) {
    // Quadruped stick figure (Go2 proportions: ~0.6m long, ~0.3m tall)
    const boneMat = new THREE.MeshPhongMaterial({ color: 0xaabbcc });
    const jointMat = new THREE.MeshPhongMaterial({ color: 0x00ccff, emissive: 0x003344 });
    const bodyMat = new THREE.MeshPhongMaterial({ color: 0x889999 });

    const root = new THREE.Group();
    this._scene.add(root);
    this._rootGroup = root;

    const addJoint = (name, pos, r = 0.012) => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(r, 8, 8), jointMat);
      s.position.copy(pos);
      root.add(s);
      if (name) {
        this._joints[name] = s;
        // Default rotation axis (pitch = X-axis for most quadruped joints)
        s.userData.axis = new THREE.Vector3(1, 0, 0);
      }
      return s;
    };
    const addBone = (from, to, r = 0.008) => {
      const dir = new THREE.Vector3().subVectors(to, from);
      const len = dir.length();
      if (len < 0.005) return;
      const geom = new THREE.CylinderGeometry(r, r, len, 6);
      geom.translate(0, len / 2, 0);
      const bone = new THREE.Mesh(geom, boneMat);
      bone.position.copy(from);
      bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      root.add(bone);
    };

    // Body frame (Y-up, Z-forward in Three.js view)
    const bodyLen = 0.36;  // half-length front-to-back
    const bodyH = 0.30;    // hip height
    const shoulderW = 0.10; // half-width
    const thighLen = 0.20;
    const calfLen = 0.20;

    // Torso (front-back spine)
    const front = new THREE.Vector3(0, bodyH, bodyLen);
    const rear = new THREE.Vector3(0, bodyH, -bodyLen);
    const mid = new THREE.Vector3(0, bodyH, 0);

    // Body spine
    addBone(rear, front, 0.012);

    // Head
    const headPos = new THREE.Vector3(0, bodyH + 0.05, bodyLen + 0.12);
    addBone(front, headPos, 0.008);
    addJoint(null, headPos, 0.03);

    // Tail stub
    const tailPos = new THREE.Vector3(0, bodyH + 0.03, -bodyLen - 0.06);
    addBone(rear, tailPos, 0.005);

    // Legs: FR, FL, RR, RL
    const legDefs = [
      { prefix: 'FR', x:  shoulderW, z:  bodyLen },
      { prefix: 'FL', x: -shoulderW, z:  bodyLen },
      { prefix: 'RR', x:  shoulderW, z: -bodyLen },
      { prefix: 'RL', x: -shoulderW, z: -bodyLen },
    ];

    for (const leg of legDefs) {
      const hip = new THREE.Vector3(leg.x, bodyH, leg.z);
      const knee = new THREE.Vector3(leg.x, bodyH - thighLen, leg.z);
      const foot = new THREE.Vector3(leg.x, bodyH - thighLen - calfLen, leg.z);

      addBone(hip, knee);
      addBone(knee, foot);

      // Hip joint rotates around Y for abduction (hip), X for pitch (thigh/calf)
      const hipJoint = addJoint(`${leg.prefix}_hip`, hip);
      hipJoint.userData.axis = new THREE.Vector3(0, 1, 0); // abduction
      addJoint(`${leg.prefix}_thigh`, knee);  // default X-axis (pitch)
      addJoint(`${leg.prefix}_calf`, foot);

      // Shoulder attachment to spine
      addBone(new THREE.Vector3(0, bodyH, leg.z), hip, 0.006);
    }

    // Body center joint (for IMU orientation)
    addJoint(null, mid, 0.018);

    // Adjust camera for quadruped view
    this._camera.position.set(0.8, 0.6, 1.0);
    this._camera.lookAt(0, 0.2, 0);

    // Move grid down to foot level
    this._scene.children.forEach(c => {
      if (c.isGridHelper) c.position.y = bodyH - thighLen - calfLen - 0.02;
    });

    this._loaded = true;
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
        // Prefer name-based matching (R1 sends joint names), fallback to index map (G1)
        const jointName = j.name || (j.idx < MOTOR_INDEX_MAP.length ? MOTOR_INDEX_MAP[j.idx] : null);
        if (!jointName) continue;
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
    this._modelGeneration += 1;
    this._modelRequest?.abort();
    this._modelRequest = null;
    this._modelNotice = null;
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
    this._rootGroup = null;
    this._pelvisGroup = null;
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
