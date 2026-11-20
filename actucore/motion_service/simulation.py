"""Native MuJoCo sessions for interactive debugging, independent of training.

The viewer receives rendered MuJoCo images and state; stepping and optional
policy inference stay in this process. This backend is identified as native MuJoCo,
not as the mjlab/MuJoCo Warp environment used for formal SuperMimic evaluation.
Profiles and model paths are administrator-owned inputs, never client paths.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping


MAX_SESSIONS = 4
MAX_RECORD_FRAMES = 3000
MAX_GEOMS = 256
MAX_BODIES = 256
MAX_CONTACTS = 32
_CONTROL_MODES = {"free_physics", "pd_hold"}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _vector(value: Any, name: str, size: int, bound: float = 1e6) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} values")
    return [_number(v, name, -bound, bound) for v in value]


def _detail(value):
    if not isinstance(value, str) or value not in ("full", "compact"):
        raise ValueError("detail must be full or compact")
    return value


class SimulationSession:
    """One MuJoCo model/data pair, guarded by a single reentrant lock."""

    def __init__(self, profile: dict, root: Path, *, control_mode: str,
                 controller: Any = None, model_id: str | None = None):
        # Importing the service, showing its cards, or querying catalog does not
        # require physics packages. Creation reports a missing runtime honestly.
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("MuJoCo simulation runtime is not installed") from exc
        self.mj, self.np = mujoco, np
        self.id = uuid.uuid4().hex
        self.profile = copy.deepcopy(profile)
        self.profile_id = profile["id"]
        self.model_id = model_id
        self.mode = "policy" if controller is not None else control_mode
        self.controller = controller
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.state = "paused"
        self.error = None
        self.steps = 0
        self._shutdown = False
        self._thread = None
        self._closed_snapshot = None
        self._closed_contact_count = 0
        self._renderer = None
        self._render_lock = threading.Lock()
        self._force_remaining = 0
        self.recording = False
        self.frames: list[dict] = []
        self.record_id = None
        self._record_file = None
        self._record_root = root / "recordings"

        path = Path(profile.get("model_path", ""))
        if not path.is_absolute() or not path.is_file() or path.suffix.lower() not in {".xml", ".mjb"}:
            raise ValueError("profile model_path must name an existing absolute MJCF .xml or .mjb file")
        from .contracts import load_model_fresh
        self.model = load_model_fresh(path)
        if profile.get("compiled_model_sha256"):
            from .contracts import model_fingerprint
            from .store import file_sha256
            if (profile.get("mujoco_version") != mujoco.__version__
                    or model_fingerprint(self.model) != profile["compiled_model_sha256"]
                    or file_sha256(path) != profile.get("model_sha256")):
                raise ValueError("simulation model assets or MuJoCo version changed; rebuild the robot profile")
        self.data = mujoco.MjData(self.model)
        if controller is not None and (abs(self.model.opt.gravity[0]) > 1e-8
                or abs(self.model.opt.gravity[1]) > 1e-8 or self.model.opt.gravity[2] >= 0):
            raise ValueError("SuperMimic policy requires world gravity along negative Z")
        self.hz = _number(profile.get("control_hz", 50), "control_hz", 1, 1000)
        self.dt = 1.0 / self.hz
        if "physics_dt" in profile:
            physics_dt = _number(profile["physics_dt"], "physics_dt", 1e-5, 0.1)
            if not math.isclose(physics_dt, self.model.opt.timestep, rel_tol=1e-6):
                raise ValueError("profile physics_dt does not match MJCF timestep")
        ratio = self.dt / self.model.opt.timestep
        if not math.isclose(ratio, round(ratio), rel_tol=1e-6) or not 1 <= round(ratio) <= 1000:
            raise ValueError("control period must be an integer multiple of MJCF timestep")
        self.substeps = round(ratio)
        if profile.get("quaternion_order", "wxyz") != "wxyz":
            raise ValueError("native MuJoCo sessions require wxyz quaternions")
        names = profile.get("joint_names", [])
        if not isinstance(names, list) or not names or len(names) > 128 or len(set(names)) != len(names):
            raise ValueError("profile joint_names must contain unique joint names")
        if not all(isinstance(n, str) and n for n in names):
            raise ValueError("profile joint_names must contain nonempty strings")
        self.joint_names = names[:]
        self.joint_ids = [self._resolve_name(mujoco.mjtObj.mjOBJ_JOINT, n, self.model.njnt) for n in names]
        if len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError("multiple profile names resolve to the same model joint")
        scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        all_scalar = [i for i in range(self.model.njnt) if int(self.model.jnt_type[i]) in scalar]
        if set(self.joint_ids) != set(all_scalar):
            raise ValueError("profile joints must match all scalar joints in the MJCF model")
        if any(int(self.model.jnt_type[i]) not in scalar for i in self.joint_ids):
            raise ValueError("controlled joints must be hinge or slide joints")
        unsupported = [i for i in range(self.model.njnt)
                       if int(self.model.jnt_type[i]) not in scalar | {int(mujoco.mjtJoint.mjJNT_FREE)}]
        if unsupported:
            raise ValueError("ball joints are not supported by this robot profile")
        self.qadr = self.model.jnt_qposadr[self.joint_ids].copy()
        self.vadr = self.model.jnt_dofadr[self.joint_ids].copy()
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) > 1:
            raise ValueError("only one free robot root is supported")
        self.free_joint = free[0] if free else None
        root_name = profile.get("root_body_name")
        self.root_body = (self._resolve_name(mujoco.mjtObj.mjOBJ_BODY, root_name, self.model.nbody)
                          if root_name else int(self.model.jnt_bodyid[free[0]]) if free else 1)
        if not 0 < self.root_body < self.model.nbody:
            raise ValueError("profile root_body_name is absent from the MJCF model")
        full_names = {i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                      for i in range(1, self.model.nbody)}
        short_names = [name.rsplit("/", 1)[-1] for name in full_names.values()]
        self.body_names = {i: name.rsplit("/", 1)[-1] if short_names.count(name.rsplit("/", 1)[-1]) == 1 else name
                           for i, name in full_names.items()}
        self.full_body_names = full_names
        self.default_q = np.asarray(_vector(profile.get("default_joint_pos", [0.0] * len(names)),
                                            "default_joint_pos", len(names)))
        self.kp = np.asarray(_vector(profile.get("kp", [0.0] * len(names)), "kp", len(names), 1e5))
        self.kd = np.asarray(_vector(profile.get("kd", [0.0] * len(names)), "kd", len(names), 1e5))
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("PD gains must be nonnegative")
        limits = profile.get("joint_limits", {})
        self.lower = np.asarray(_vector(limits.get("lower", [-1e6] * len(names)), "lower", len(names)))
        self.upper = np.asarray(_vector(limits.get("upper", [1e6] * len(names)), "upper", len(names)))
        if np.any(self.lower >= self.upper) or np.any(self.default_q < self.lower) or np.any(self.default_q > self.upper):
            raise ValueError("invalid joint limits or default position")
        for index, jid in enumerate(self.joint_ids):
            if self.model.jnt_limited[jid]:
                low, high = self.model.jnt_range[jid]
                if self.default_q[index] < low or self.default_q[index] > high:
                    raise ValueError("default joint position exceeds MJCF limits")
                self.lower[index] = max(self.lower[index], low)
                self.upper[index] = min(self.upper[index], high)
        self.actuators, self.gains, self.position_servos, self.force_limited_position_servos = (
            self._actuator_map() if self.mode != "free_physics" else ([], [], [], []))
        if self.mode == "free_physics":
            # ctrl=0 does not turn off a position servo: it commands zero angle.
            # Each session owns its model, so disabling actuation here cannot
            # change another session or the training environment.
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        if self.mode != "free_physics" and not np.any(self.kp):
            raise ValueError("PD control requires nonzero profile kp")
        self._reset_locked()

    def _resolve_name(self, object_type, name, count):
        """mjlab's scene export prefixes entity names; ambiguity is an error."""
        exact = self.mj.mj_name2id(self.model, object_type, name)
        if exact >= 0:
            return exact
        matches = [i for i in range(count)
                   if (self.mj.mj_id2name(self.model, object_type, i) or "").rsplit("/", 1)[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"profile name {name!r} is absent or ambiguous in the MJCF model")
        return matches[0]

    def _actuator_map(self):
        """Support motors and verified native position servos without double PD.

        mjlab's G1 uses BuiltinPositionActuator: fixed gain kp, affine bias
        [0, -kp, -kd], unit joint gear. It consumes position directly. A pure
        motor instead consumes the explicitly computed PD force divided by its
        transmission gain. Other actuator contracts are refused.
        """
        mj, model = self.mj, self.model
        ids, gains, position_servos, force_limited_position_servos = [], [], [], []
        for index, jid in enumerate(self.joint_ids):
            matches = [a for a in range(model.nu)
                       if model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT
                       and model.actuator_trnid[a, 0] == jid]
            if len(matches) != 1:
                raise ValueError("PD control requires exactly one motor per profile joint")
            aid = matches[0]
            if (model.actuator_dyntype[aid] != mj.mjtDyn.mjDYN_NONE
                    or model.actuator_gaintype[aid] != mj.mjtGain.mjGAIN_FIXED
                    or self.np.any(model.actuator_gear[aid, 1:])):
                raise ValueError("PD control requires fixed-gain scalar joint motors")
            position = model.actuator_biastype[aid] == mj.mjtBias.mjBIAS_AFFINE
            if position:
                if (not math.isclose(model.actuator_gear[aid, 0], 1.0, abs_tol=1e-8)
                        or not math.isclose(model.actuator_gainprm[aid, 0], self.kp[index], rel_tol=1e-6)
                        or not self.np.allclose(model.actuator_biasprm[aid, :3],
                                              [0, -self.kp[index], -self.kd[index]], rtol=1e-6, atol=1e-8)):
                    raise ValueError("native position servo gains do not match profile kp/kd")
            elif model.actuator_biastype[aid] != mj.mjtBias.mjBIAS_NONE:
                raise ValueError("unsupported actuator bias contract")
            gain = float(model.actuator_gainprm[aid, 0] * model.actuator_gear[aid, 0])
            if not math.isfinite(gain) or abs(gain) < 1e-8:
                raise ValueError("invalid motor gain")
            ids.append(aid)
            gains.append(gain)
            position_servos.append(position)
            # mjlab's BuiltinPositionActuator deliberately permits desired
            # positions beyond the mechanical range: MuJoCo limits the force
            # and constrains the actual joint. Rejecting those references (or
            # clipping them to joint limits) changes the trained PD controller.
            # Only use that contract for verified servos with finite force
            # limits and enabled physical joint constraints. Generic motors
            # and servos without that protection retain the target guard.
            force_range = model.actuator_forcerange[aid]
            force_limited_position_servos.append(bool(
                position and model.actuator_forcelimited[aid]
                and self.np.isfinite(force_range).all() and force_range[0] < force_range[1]
                and model.jnt_limited[jid] and self.np.isfinite(model.jnt_range[jid]).all()
                and not model.opt.disableflags & (int(mj.mjtDisableBit.mjDSBL_LIMIT)
                                                   | int(mj.mjtDisableBit.mjDSBL_CONSTRAINT))))
        if len(ids) != model.nu:
            raise ValueError("extra actuators are not supported in PD/policy mode")
        return (ids, self.np.asarray(gains), self.np.asarray(position_servos),
                self.np.asarray(force_limited_position_servos))

    def _reset_locked(self):
        mj, np = self.mj, self.np
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.default_q
        self.steps = 0
        self._force_remaining = 0
        if self.controller is not None:
            self.controller.reset()
            initial = getattr(self.controller, "initial_state", None) or {}
            if not isinstance(initial, dict):
                raise ValueError("controller initial_state must be a dictionary")
            if "joint_pos" in initial:
                q = np.asarray(_vector(initial["joint_pos"], "initial joint_pos", len(self.joint_names)))
                if np.any(q < self.lower) or np.any(q > self.upper):
                    raise ValueError("controller initial joint position exceeds limits")
                self.data.qpos[self.qadr] = q
            if "joint_vel" in initial:
                self.data.qvel[self.vadr] = _vector(initial["joint_vel"], "initial joint_vel", len(self.joint_names))
            if self.free_joint is not None:
                qadr, vadr = int(self.model.jnt_qposadr[self.free_joint]), int(self.model.jnt_dofadr[self.free_joint])
                if "root_pos" in initial:
                    self.data.qpos[qadr:qadr + 3] = _vector(initial["root_pos"], "initial root_pos", 3)
                if "root_quat" in initial:
                    quat = np.asarray(_vector(initial["root_quat"], "initial root_quat", 4))
                    norm = float(np.linalg.norm(quat))
                    if not math.isclose(norm, 1.0, rel_tol=1e-3):
                        raise ValueError("controller initial quaternion must be normalized")
                    self.data.qpos[qadr + 3:qadr + 7] = quat / norm
                if "root_lin_vel_w" in initial:
                    self.data.qvel[vadr:vadr + 3] = _vector(initial["root_lin_vel_w"], "initial root_lin_vel_w", 3)
                if "root_ang_vel_w" in initial:
                    # Free-joint rotational velocities use the local frame.
                    rot = np.zeros(9)
                    mj.mju_quat2Mat(rot, self.data.qpos[qadr + 3:qadr + 7])
                    self.data.qvel[vadr + 3:vadr + 6] = rot.reshape(3, 3).T @ np.asarray(
                        _vector(initial["root_ang_vel_w"], "initial root_ang_vel_w", 3))
            elif any(key in initial for key in ("root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_w")):
                raise ValueError("controller root state requires a free robot root")
        for index, aid in enumerate(self.actuators):
            if self.position_servos[index]:
                self.data.ctrl[aid] = self.data.qpos[self.qadr[index]]
        mj.mj_forward(self.model, self.data)
        self.error = None

    def _observation(self):
        mj, np, model, data = self.mj, self.np, self.model, self.data
        velocity = np.zeros(6)
        mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, self.root_body, velocity, 0)
        root_vel = velocity.copy()
        bodies = []
        for i in range(1, min(model.nbody, MAX_BODIES + 1)):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, i, velocity, 0)
            bodies.append({"id": i, "name": self.body_names[i], "model_name": self.full_body_names[i],
                           "parent": int(model.body_parentid[i]), "pos": data.xpos[i].tolist(),
                           "quat": data.xquat[i].tolist(), "lin_vel_w": velocity[3:].tolist(),
                           "ang_vel_w": velocity[:3].tolist()})
        return {"time": float(data.time), "step": self.steps, "dt": self.dt,
                "joint_pos": data.qpos[self.qadr].tolist(), "joint_vel": data.qvel[self.vadr].tolist(),
                "root_pose": {"pos": data.xpos[self.root_body].tolist(), "quat": data.xquat[self.root_body].tolist()},
                "root_lin_vel_w": root_vel[3:].tolist(), "root_ang_vel_w": root_vel[:3].tolist(),
                "root_ang_vel_b": (data.xmat[self.root_body].reshape(3, 3).T @ root_vel[:3]).tolist(),
                "bodies": bodies}

    def _step_locked(self):
        np, model, data = self.np, self.model, self.data
        target = self.default_q
        if self.controller is not None:
            target = np.asarray(_vector(self.controller.step(self._observation()), "policy target", len(self.joint_names)))
            if np.any(((target < self.lower) | (target > self.upper))
                      & ~self.force_limited_position_servos):
                raise ValueError("policy target exceeds robot joint limits")
        for _ in range(self.substeps):
            if self.mode != "free_physics":
                torque = self.kp * (target - data.qpos[self.qadr]) - self.kd * data.qvel[self.vadr]
                commands = np.where(self.position_servos, target, torque / self.gains)
                for index, aid in enumerate(self.actuators):
                    value = commands[index]
                    if model.actuator_ctrllimited[aid]:
                        value = np.clip(value, *model.actuator_ctrlrange[aid])
                    data.ctrl[aid] = value
            warnings = [self.mj.mjtWarning.mjWARN_BADQPOS, self.mj.mjtWarning.mjWARN_BADQVEL,
                        self.mj.mjtWarning.mjWARN_BADQACC]
            before = [int(data.warning[w].number) for w in warnings]
            self.mj.mj_step(model, data)
            if any(int(data.warning[w].number) > count for w, count in zip(warnings, before)):
                raise RuntimeError("MuJoCo reported an unstable physics state")
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("MuJoCo produced nonfinite state")
        self.mj.mj_forward(model, data)
        self.steps += 1
        if self._force_remaining:
            self._force_remaining -= 1
            if not self._force_remaining:
                data.xfrc_applied[:] = 0
        if self.recording:
            self._capture_locked()
        if self.controller is not None and getattr(self.controller, "reference_complete", False):
            self.state = "completed"
            self.data.xfrc_applied[:] = 0
            self._force_remaining = 0
            self._stop_record_locked()

    def _capture_locked(self):
        if len(self.frames) >= MAX_RECORD_FRAMES:
            self._stop_record_locked()
            return
        frame = {"time": float(self.data.time), "step": self.steps,
                 "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist()}
        self.frames.append(frame)
        self._record_file.write(json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n")
        self._record_file.flush()
        if len(self.frames) == MAX_RECORD_FRAMES:
            self._stop_record_locked()

    def _stop_record_locked(self):
        self.recording = False
        if self._record_file is not None:
            self._record_file.close()
            self._record_file = None

    def _snapshot_locked(self, detail="full"):
        if self.state == "closed":
            closed = copy.deepcopy(self._closed_snapshot)
            if detail == "compact":
                return {**{key: closed[key] for key in (
                    "time", "step", "dt", "root_pose", "joint_names", "joint_pos", "truncated")},
                        "contact_count": self._closed_contact_count,
                        "bodies": [{"id": body["id"], "name": body["name"]}
                                   for body in closed.get("bodies", [])]}
            return closed
        mj, model, data = self.mj, self.model, self.data
        if detail == "compact":
            # Render metadata does not need world-space transforms for every
            # collision/visual geom or every body velocity. Read only the fields
            # consumed by the workbench beside its JPEG image.
            return {"time": float(data.time), "step": self.steps, "dt": self.dt,
                    "root_pose": {"pos": data.xpos[self.root_body].tolist(),
                                  "quat": data.xquat[self.root_body].tolist()},
                    "joint_names": self.joint_names[:],
                    "joint_pos": data.qpos[self.qadr].tolist(),
                    "contact_count": int(data.ncon),
                    "bodies": [{"id": i, "name": self.body_names[i]}
                               for i in range(1, min(model.nbody, MAX_BODIES + 1))],
                    "truncated": model.nbody > MAX_BODIES + 1}
        observation = self._observation()
        observation.update({"qpos": data.qpos.tolist(), "joint_names": self.joint_names[:]})
        names = {int(mj.mjtGeom.mjGEOM_PLANE): "plane", int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
                 int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mj.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
                 int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mj.mjtGeom.mjGEOM_BOX): "box",
                 int(mj.mjtGeom.mjGEOM_MESH): "mesh", int(mj.mjtGeom.mjGEOM_HFIELD): "hfield"}
        observation["geoms"] = [
            {"id": i, "type": names.get(int(model.geom_type[i]), "unknown"),
             "body": int(model.geom_bodyid[i]), "size": model.geom_size[i].tolist(),
             "pos": data.geom_xpos[i].tolist(), "mat": data.geom_xmat[i].tolist(),
             "rgba": model.geom_rgba[i].tolist()}
            for i in range(min(model.ngeom, MAX_GEOMS))]
        observation["contacts"] = [
            {"pos": data.contact[i].pos.tolist(), "normal": data.contact[i].frame[:3].tolist(),
             "distance": float(data.contact[i].dist)} for i in range(min(data.ncon, MAX_CONTACTS))]
        observation["truncated"] = bool(model.ngeom > MAX_GEOMS or model.nbody > MAX_BODIES + 1 or data.ncon > MAX_CONTACTS)
        return observation

    def reply(self, *, snapshot_detail="full", **extra):
        with self.lock:
            return {"session": {"id": self.id, "profile_id": self.profile_id, "model_id": self.model_id,
                                "state": self.state, "engine": "native_mujoco", "control_mode": self.mode,
                                "origin": getattr(self, "origin", "api"),
                                "time": float(self._closed_snapshot["time"] if self.state == "closed" else self.data.time),
                                "recording": self.recording, "recorded_frames": len(self.frames),
                                "record_id": self.record_id, "error": self.error,
                                "reference_complete": bool(getattr(self.controller, "reference_complete", False)),
                                "reference_frame": getattr(self.controller, "reference_frame", None),
                                "reference_frames": getattr(self.controller, "reference_frames", None)},
                    "snapshot": self._snapshot_locked(snapshot_detail), **extra}

    def _loop(self):
        while True:
            self.wake.clear()
            started = time.monotonic()
            with self.lock:
                if self._shutdown:
                    return
                if self.state == "running":
                    try:
                        self._step_locked()
                    except Exception as exc:
                        self.state = "error"
                        self.error = str(exc)
                        self.data.xfrc_applied[:] = 0
                        self._stop_record_locked()
            self.wake.wait(max(0.001, self.dt - (time.monotonic() - started)))

    def _active(self):
        if self.state == "closed":
            raise ValueError("simulation session is closed")

    def action(self, action: str, args: dict):
        if action == "simulation_render":
            return self.render(args)
        if action == "simulation_close":
            self.close()
            return self.reply()
        with self.lock:
            if action == "simulation_inspect":
                return self.reply(snapshot_detail=_detail(args.get("detail", "full")))
            self._active()
            if self.state == "completed" and action in {
                "simulation_run", "simulation_step", "simulation_record", "simulation_perturb"
            }:
                raise ValueError("reset the completed simulation before running")
            if action == "simulation_run":
                if self.state == "error":
                    raise ValueError("reset the failed simulation before running")
                self.state = "running"
                if self._thread is None:
                    self._thread = threading.Thread(target=self._loop, name=f"mujoco-{self.id[:8]}", daemon=True)
                    self._thread.start()
                self.wake.set()
            elif action == "simulation_pause":
                if self.state == "running":
                    self.state = "paused"
            elif action == "simulation_step":
                if self.state != "paused":
                    raise ValueError("single stepping requires a paused session")
                steps = _integer(args.get("steps", 1), "steps", 1, 100)
                try:
                    for _ in range(steps):
                        self._step_locked()
                        if self.state == "completed":
                            break
                except Exception as exc:
                    self.state = "error"
                    self.error = str(exc)
                    self.data.xfrc_applied[:] = 0
                    self._stop_record_locked()
                    raise
            elif action == "simulation_reset":
                self.state = "paused"
                self._stop_record_locked()
                try:
                    self._reset_locked()
                except Exception as exc:
                    self.state = "error"
                    self.error = str(exc)
                    raise
            elif action == "simulation_perturb":
                body = _integer(args.get("body_id", self.root_body), "body_id", 1, self.model.nbody - 1)
                force = _vector(args.get("force", [0.0, 0.0, 0.0]), "force", 3, 1000)
                torque = _vector(args.get("torque", [0.0, 0.0, 0.0]), "torque", 3, 1000)
                duration = _number(args.get("duration_s", 0.1), "duration_s", self.dt, 2)
                self.data.xfrc_applied[:] = 0
                self.data.xfrc_applied[body] = force + torque
                self._force_remaining = math.ceil(duration / self.dt)
            elif action == "simulation_record":
                if self.recording:
                    raise ValueError("session is already recording")
                self._record_root.mkdir(parents=True, exist_ok=True)
                self.record_id = uuid.uuid4().hex
                self._record_file = (self._record_root / f"{self.record_id}.jsonl").open("x", encoding="utf-8")
                self._record_file.write(json.dumps({"protocol": "motus.simulation-recording/1",
                    "profile_id": self.profile_id, "model_id": self.model_id,
                    "engine": "native_mujoco", "control_hz": self.hz}) + "\n")
                self.frames = []
                self.recording = True
                self._capture_locked()
            elif action == "simulation_record_stop":
                self._stop_record_locked()
            elif action == "simulation_replay":
                return self._replay_locked(args)
            else:
                raise ValueError(f"unsupported simulation action: {action}")
            return self.reply()

    def render(self, args: dict):
        """Render a consistent snapshot without holding up the physics loop.

        OpenGL belongs to the renderer's dedicated thread. Live and replay
        frames use a separate model/data pair and never advance this session.
        One outstanding frame per session bounds work from concurrent clients.
        """
        width = _integer(args.get("width", 960), "width", 160, 960)
        height = _integer(args.get("height", 540), "height", 120, 640)
        jpeg_quality = _integer(args.get("jpeg_quality", 88), "jpeg_quality", 40, 95)
        detail = _detail(args.get("detail", "full"))
        camera = {
            "azimuth": _number(args.get("azimuth", 135), "azimuth", -3600, 3600),
            "elevation": _number(args.get("elevation", -20), "elevation", -89, 89),
            "distance": _number(args.get("distance", 3), "distance", .25, 30),
        }
        if "lookat" in args:
            camera["lookat"] = _vector(args["lookat"], "lookat", 3, 10000)
        if not self._render_lock.acquire(blocking=False):
            raise RuntimeError("frame rendering is busy; retry after the current frame")
        try:
            with self.lock:
                self._active()
                if "offset" in args:
                    offset = _integer(args["offset"], "offset", 0, MAX_RECORD_FRAMES - 1)
                    if offset >= len(self.frames):
                        raise ValueError("recording frame does not exist")
                    reply = self._replay_locked({"offset": offset, "limit": 1}, snapshot_detail=detail)
                    snapshot = reply.pop("frames")[0]
                    frame = copy.deepcopy(self.frames[offset])
                    reply["snapshot"] = snapshot
                else:
                    reply = self.reply(snapshot_detail=detail)
                    snapshot = reply["snapshot"]
                    frame = {"qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
                             "time": float(self.data.time)}
                camera.setdefault("lookat", snapshot["root_pose"]["pos"][:])
                if self._renderer is None:
                    from .rendering import FrameRenderer
                    self._renderer = FrameRenderer(self.model)
                renderer = self._renderer
            rendered = renderer.render(frame, camera, width=width, height=height,
                                       timeout=5, jpeg_quality=jpeg_quality)
            reply["image"] = {"mime": rendered["mime_type"], "width": rendered["width"],
                              "height": rendered["height"],
                              "data": base64.b64encode(rendered["data"]).decode("ascii")}
            return reply
        finally:
            self._render_lock.release()

    def _replay_locked(self, args, *, snapshot_detail="full"):
        offset = _integer(args.get("offset", 0), "offset", 0, MAX_RECORD_FRAMES)
        limit = _integer(args.get("limit", 1), "limit", 1, 10)
        # A separate data object keeps replay from changing the live session,
        # including its velocity, contact state, actuator state, and clock.
        live, live_steps = self.data, self.steps
        frames = []
        try:
            self.data = self.mj.MjData(self.model)
            for frame in self.frames[offset:offset + limit]:
                self.data.qpos[:] = frame["qpos"]
                self.data.qvel[:] = frame["qvel"]
                self.data.time = frame["time"]
                self.steps = frame["step"]
                self.mj.mj_forward(self.model, self.data)
                frames.append(self._snapshot_locked(snapshot_detail))
        finally:
            self.data, self.steps = live, live_steps
        end = offset + len(frames)
        return self.reply(snapshot_detail=snapshot_detail, frames=frames, total=len(self.frames),
                          next_offset=end if end < len(self.frames) else None)

    def close(self):
        with self.lock:
            if self.state == "closed":
                return
            self._closed_snapshot = self._snapshot_locked()
            self._closed_contact_count = int(self.data.ncon)
            self._shutdown = True
            self.state = "closed"
            self._stop_record_locked()
            self.wake.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("simulation thread did not stop")
        # Finish a pending frame before releasing the model. Never hold the
        # physics lock while waiting for the OpenGL owner thread to close.
        with self._render_lock:
            renderer, self._renderer = self._renderer, None
            if renderer is not None:
                renderer.close()
        with self.lock:
            try:
                if self.controller is not None and callable(getattr(self.controller, "close", None)):
                    self.controller.close()
            finally:
                self.controller = None
                self.data = None
                self.model = None


class SimulationManager:
    """Bounded session registry, owned and closed by the motion service."""

    def __init__(self, profiles: Mapping[str, dict], root: Path,
                 model_resolver: Callable | None = None, dataset_resolver: Callable | None = None):
        self.profiles = profiles
        self.root = Path(root)
        self.model_resolver = model_resolver
        self.dataset_resolver = dataset_resolver
        self.sessions: dict[str, SimulationSession] = {}
        self.lock = threading.RLock()
        self._closed = False
        self._controls = {}
        self._requests = {}
        self._lease_seconds = 20.0
        self._clock = time.monotonic
        self._lease_wake = threading.Event()
        self._lease_thread = None

    @staticmethod
    def _identity(value, name):
        if not isinstance(value, str) or not 1 <= len(value) <= 96 or not all(
                c.isascii() and (c.isalnum() or c in "_-.:") for c in value):
            raise ValueError(f"{name} must be a bounded identifier")
        return value

    def _start_leases(self):
        if self._lease_thread is None:
            self._lease_thread = threading.Thread(target=self._lease_loop, name="simulation-control", daemon=True)
            self._lease_thread.start()

    def _lease_loop(self):
        while not self._lease_wake.wait(.5):
            with self.lock:
                if self._closed:
                    return
                for sid, control in self._controls.items():
                    self._expire(self.sessions.get(sid), control)

    def _pause_owner(self, session):
        if session is not None and session.state != "closed":
            session.action("simulation_pause", {})
            if session.recording:
                session.action("simulation_record_stop", {})

    def _expire(self, session, control):
        if control["owner"] and control["deadline"] <= self._clock():
            self._pause_owner(session)
            control.update(owner=None, deadline=0, epoch=control["epoch"] + 1)

    def _control(self, session, view_id=None):
        value = self._controls.get(session.id)
        if value is None:
            return None  # Legacy sessions remain operable during a rolling upgrade.
        self._expire(session, value)
        return {"owner_view_id": value["owner"], "epoch": value["epoch"],
                "expires_in_s": round(max(0, value["deadline"] - self._clock()), 3),
                "lease_seconds": self._lease_seconds,
                "is_controller": bool(session.state != "closed" and view_id and value["owner"] == view_id)}

    def _reply(self, session, result, view_id=None):
        control = self._control(session, view_id)
        if control is not None:
            result["control"] = control
            if "session" in result:
                result["session"]["control"] = dict(control)
        return result

    def _require_control(self, session, args):
        control = self._controls.get(session.id)
        if control is None:
            if args.get("view_id"):
                raise ValueError("请先接管此仿真会话的控制权")
            return
        self._expire(session, control)
        epoch = args.get("control_epoch")
        if (not control["owner"] or args.get("view_id") != control["owner"]
                or type(epoch) is not int or epoch != control["epoch"]):
            raise ValueError("仿真控制权已变化，请刷新状态后接管控制")

    def call(self, action: str, args: dict):
        if not isinstance(args, dict):
            raise ValueError("simulation arguments must be an object")
        view = args.get("view_id")
        if view is not None:
            self._identity(view, "view_id")
        if action == "simulation_list":
            with self.lock:
                result = []
                for session in self.sessions.values():
                    if session.state != "closed":
                        self._control(session, view)
                        result.append(self._reply(session, session.reply(snapshot_detail="compact"), view)["session"])
                return {"sessions": result}
        if action == "simulation_create":
            return self._create(args)
        if not isinstance(args.get("session_id"), str):
            raise ValueError("session_id must be a string")
        with self.lock:
            session = self.sessions.get(args["session_id"])
            if session is None:
                raise ValueError("unknown simulation session")
            self._control(session, view)
            if action == "simulation_claim":
                self._identity(view, "view_id")
                if session.state == "closed":
                    raise ValueError("simulation session is closed")
                takeover = args.get("takeover", False)
                if type(takeover) is not bool:
                    raise ValueError("takeover must be a boolean")
                control = self._controls.get(session.id)
                if control is None:
                    control = {"owner": None, "epoch": 0, "deadline": 0}
                epoch = args.get("control_epoch")
                if type(epoch) is not int or epoch != control["epoch"]:
                    raise ValueError("仿真控制权已变化，请刷新状态")
                if control["owner"] and control["owner"] != view and not takeover:
                    raise ValueError("此仿真正在由另一个窗口控制，请选择接管控制")
                self._controls[session.id] = control
                if control["owner"] != view:
                    self._pause_owner(session)
                    control.update(owner=view, epoch=control["epoch"] + 1)
                control["deadline"] = self._clock() + self._lease_seconds
                self._start_leases()
                return self._reply(session, session.reply(), view)
            if action in {"simulation_heartbeat", "simulation_release"}:
                self._require_control(session, args)
                control = self._controls.get(session.id)
                if control is None:
                    raise ValueError("请先接管此仿真会话的控制权")
                if action == "simulation_release":
                    self._pause_owner(session)
                    control.update(owner=None, deadline=0, epoch=control["epoch"] + 1)
                else:
                    control["deadline"] = self._clock() + self._lease_seconds
                return self._reply(session, session.reply(snapshot_detail="compact"), view)
            readonly = action in {"simulation_render", "simulation_inspect", "simulation_replay"}
            if not readonly:
                self._require_control(session, args)
                # Hold the registry lock through authorization and mutation: an
                # old request cannot race a successful takeover or release.
                result = session.action(action, args)
                return self._reply(session, result, view)
        # Rendering may block on OpenGL; it never holds up ownership changes.
        result = session.action(action, args)
        with self.lock:
            return self._reply(session, result, view)

    def _create(self, args: dict):
        profile_id = args.get("profile_id")
        if not isinstance(profile_id, str) or profile_id not in self.profiles:
            raise ValueError("unknown robot profile")
        profile = copy.deepcopy(self.profiles[profile_id])
        profile["id"] = profile_id
        mode = args.get("control_mode", "free_physics")
        if not isinstance(mode, str) or mode not in _CONTROL_MODES:
            raise ValueError("control_mode must be free_physics or pd_hold")
        origin = args.get("origin", "api")
        if not isinstance(origin, str) or origin not in {"api", "motion", "simulation"}:
            raise ValueError("invalid simulation origin")
        model_id = args.get("model_id") or None
        if model_id is not None and not isinstance(model_id, str):
            raise ValueError("model_id must be a string")
        # Dataset rebinding changes a baked motion policy's contract. The
        # model resolver owns its reference clip; it cannot silently accept a
        # different dataset merely because the viewer passed one.
        if args.get("dataset_id"):
            raise ValueError("interactive sessions use the model's bundled reference motion")
        with self.lock:
            if self._closed:
                raise ValueError("simulation service is closing")
            view = args.get("view_id")
            request_id = args.get("request_id")
            fingerprint = (profile_id, model_id, mode, origin)
            request_key = None
            if view is not None:
                self._identity(view, "view_id")
                self._identity(request_id, "request_id")
                request_key = (view, request_id)
                previous = self._requests.get(request_key)
                if previous:
                    sid, original = previous
                    if original != fingerprint:
                        raise ValueError("创建请求已用于另一模型或机器人，请重新发起")
                    session = self.sessions.get(sid)
                    if session is None:
                        raise ValueError("此创建请求的会话已结束，请重新发起")
                    return self._reply(session, session.reply(), view)
            active = [s for s in self.sessions.values() if s.state != "closed"]
            if len(active) >= MAX_SESSIONS:
                raise ValueError("maximum active simulation sessions reached")
            # Retain a small number of terminal states for idempotent close.
            if len(self.sessions) >= 32:
                self.sessions = {sid: s for sid, s in self.sessions.items() if s.state != "closed"}
                self._controls = {sid: value for sid, value in self._controls.items() if sid in self.sessions}
            controller = None
            if model_id is not None:
                if self.model_resolver is None:
                    raise ValueError("policy execution is not configured for this simulation service")
                controller = self.model_resolver(model_id, profile)
                if not callable(getattr(controller, "step", None)) or not callable(getattr(controller, "reset", None)):
                    raise ValueError("model resolver must return a validated policy controller")
            try:
                session = SimulationSession(profile, self.root, control_mode=mode,
                                            controller=controller, model_id=model_id)
            except Exception:
                if controller is not None and callable(getattr(controller, "close", None)):
                    controller.close()
                raise
            session.origin = origin
            self.sessions[session.id] = session
            if view is not None:
                self._controls[session.id] = {"owner": view, "epoch": 1,
                                              "deadline": self._clock() + self._lease_seconds}
                self._requests[request_key] = (session.id, fingerprint)
                # Keep request tombstones longer than session summaries so a
                # delayed network retry never creates a second live session.
                while len(self._requests) > 1024:
                    self._requests.pop(next(iter(self._requests)))
                self._start_leases()
            return self._reply(session, session.reply(), view)

    def close(self):
        with self.lock:
            self._closed = True
            self._lease_wake.set()
            sessions = list(self.sessions.values())
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=2)
        first_error = None
        for session in sessions:
            try:
                session.close()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
