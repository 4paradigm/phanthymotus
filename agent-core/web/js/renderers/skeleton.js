/**
 * skeleton.js — 3D URDF skeleton renderer for robot joint state.
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
    this._heliosUpperBody = null;
    this._lastData = null;
    this._needsFit = true;
    this._userViewChanged = false;
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
    this._controls.addEventListener('start', () => { this._userViewChanged = true; });
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
    this._grid = grid;
  },

  async _loadModel(mcpId) {
    const mountedElement = this._el;
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
      if (this._el !== mountedElement || !this._scene) return;
      // API returns { code, data } where data is the content array
      const content = json.data?.[0]?.text ?? json.result?.content?.[0]?.text;
      if (!content) { this._buildFallbackSkeleton(); return; }
      const data = JSON.parse(content);
      if (data.urdf) {
        this._parseAndBuild(data.urdf);
        if (this._lastData) this._applyState(this._lastData);
        this._fitModel();
      } else if (data.type === 'quadruped') {
        this._buildQuadrupedSkeleton(data);
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
    // Simplify this model's display only; keep every kinematic frame.
    const isHelios = doc.querySelector('robot')?.getAttribute('name') === 'Rokae_Robot_2.1_AR5_v1.1';
    const hiddenSensorLinks = new Set(isHelios ? [
      'ChestRGBD_link', 'HeadRGBD_link', 'RGB1_link', 'RGB2_link', 'RGB3_link',
    ] : []);
    const symbolicJoints = new Set(isHelios ? [
      'left_fixed', 'right_fixed', 'left_joint_2', 'right_joint_2',
      'head_joint_1', 'head_joint_2',
    ] : []);
    this._heliosUpperBody = null;

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
        // URDF fixed-axis RPY means Rz(yaw) * Ry(pitch) * Rx(roll).
        // Three.js intrinsic ZYX gives that order; XYZ tilts compound origins.
        const euler = new THREE.Euler(j.rpy.x, j.rpy.y, j.rpy.z, 'ZYX');
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
        if (!hiddenSensorLinks.has(j.childLink) && !symbolicJoints.has(j.name)
            && (j.type === 'revolute' || j.type === 'fixed')) {
          this._addBoneVisual(jointGroup, j);
        }

        queue.push(j.childLink);
      }
    }

    // Add joint spheres for revolute joints
    for (const [name, obj] of Object.entries(this._joints)) {
      if (isHelios && ['head_joint_1', 'head_joint_2', 'left_joint_1', 'right_joint_1'].includes(name)) continue;
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
      if (name === rootLinkName || hiddenSensorLinks.has(name)) continue;
      const hasChildren = joints.some(j => j.parentLink === name);
      if (!hasChildren) {
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(0.01, 8, 8),
          new THREE.MeshPhongMaterial({ color: 0x44ddff, emissive: 0x002233 })
        );
        obj.add(sphere);
      }
    }

    if (isHelios) this._buildHeliosUpperBody();
    this._loaded = true;
  },

  _buildHeliosUpperBody() {
    // Symbolic torso/shoulders, separate from the untouched URDF kinematic tree.
    const group = new THREE.Group();
    this._links.body_link_4.add(group);
    const bone = () => {
      const geometry = new THREE.CylinderGeometry(0.008, 0.008, 1, 6);
      geometry.translate(0, 0.5, 0);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ color: 0xaabbcc }));
      group.add(mesh);return mesh;
    };
    const dot = radius => new THREE.Mesh(new THREE.SphereGeometry(radius, 8, 8),
      new THREE.MeshPhongMaterial({ color: 0x00ccff, emissive: 0x003344 }));
    const shoulder = dot(0.015);group.add(shoulder);
    const headFrame = new THREE.Group();group.add(headFrame);
    const neck = bone();group.remove(neck);headFrame.add(neck);
    neck.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1));
    neck.scale.y = 0.10;
    const head = dot(0.04);head.position.z = 0.10;headFrame.add(head);
    this._heliosUpperBody = { group, shoulder, headFrame, head, neck,
      shoulderLine: bone(), spine: bone() };
    this._updateHeliosUpperBody();
  },

  _updateHeliosUpperBody() {
    const view = this._heliosUpperBody;
    if (!view) return;
    this._rootGroup.updateWorldMatrix(true, true);
    const local = link => view.group.worldToLocal(link.getWorldPosition(new THREE.Vector3()));
    const left = local(this._links.left_link_2);
    const right = local(this._links.right_link_2);
    const center = left.clone().add(right).multiplyScalar(0.5);
    const segment = (mesh, from, to) => {
      const direction = to.clone().sub(from);
      const length = direction.length();
      mesh.visible = length > 0.0001;
      mesh.position.copy(from);mesh.scale.y = length;
      if (mesh.visible) mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
    };
    segment(view.shoulderLine, left, right);
    segment(view.spine, new THREE.Vector3(), center);
    view.shoulder.position.copy(center);
    view.headFrame.position.copy(center);
    // Both head angles drive the symbol, relative to the real torso orientation.
    view.headFrame.quaternion.copy(view.group.getWorldQuaternion(new THREE.Quaternion())).invert()
      .multiply(this._links.head_link_2.getWorldQuaternion(new THREE.Quaternion()));
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
    try {
      const text = new TextDecoder().decode(buffer);
      const data = JSON.parse(text);
      this._lastData = data;
      this._applyState(data);
    } catch (e) {
      // Silently ignore parse errors (graceful degradation)
    }
  },

  _applyState(data) {
    if (!this._loaded) return;
    if (data.imu_quat && this._pelvisGroup) {
      const [w, x, y, z] = data.imu_quat;
      this._pelvisGroup.quaternion.set(x, y, z, w);
    }
    if (!Array.isArray(data.joints)) return;
    for (const j of data.joints) {
      const name = j.name || (j.idx < MOTOR_INDEX_MAP.length ? MOTOR_INDEX_MAP[j.idx] : null);
      const obj = this._joints[name];
      if (!obj?.userData.axis || !Number.isFinite(j.q)) continue;
      obj.quaternion.setFromAxisAngle(obj.userData.axis, j.q);
    }
    this._updateHeliosUpperBody();
    if (data.joints.length && this._needsFit) {
      this._fitModel();
      this._needsFit = false;
    }
  },

  _fitModel() {
    if (!this._rootGroup || !this._camera || !this._controls) return;
    this._rootGroup.updateWorldMatrix(true, true);
    const bounds = new THREE.Box3().setFromObject(this._rootGroup);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 0.1);
    const vertical = THREE.MathUtils.degToRad(this._camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * this._camera.aspect);
    const distance = radius / Math.sin(Math.min(vertical, horizontal)) * 1.15;
    this._controls.target.copy(center);
    this._camera.position.copy(center).addScaledVector(new THREE.Vector3(1, 0.5, 1.3).normalize(), distance);
    this._camera.near = Math.max(distance / 1000, 0.001);
    this._camera.far = Math.max(distance * 10, 50);
    this._camera.updateProjectionMatrix();
    this._controls.update();
    if (this._grid) this._grid.position.y = bounds.min.y - 0.03;
  },

  _animate() {
    if (!this._threeRenderer) return;
    this._raf = requestAnimationFrame(() => this._animate());
    this._controls?.update();
    this._threeRenderer.render(this._scene, this._camera);
  },

  _resize() {
    if (!this._el || !this._threeRenderer) return;
    const target = this._el;
    const w = target.clientWidth;
    const h = target.clientHeight;
    if (w === 0 || h === 0) return;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._threeRenderer.setSize(w, h);
    // A wide initial fit is too close after narrowing the panel. Refit until
    // the user takes control; later layout changes must preserve their view.
    if (this._loaded && !this._userViewChanged) this._fitModel();
  },

  unmount() {
    this._heliosUpperBody = null;
    this._lastData = null;
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
    this._grid = null;
    this._rootGroup = null;
    this._pelvisGroup = null;
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
