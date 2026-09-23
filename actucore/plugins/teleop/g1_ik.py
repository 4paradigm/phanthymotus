"""Pinocchio/CasADi inverse kinematics for the fixed G1_23 V1 profile.

This is a focused port of Unitree's Apache-2.0 ``G1_23_ArmIK`` algorithm in
``xr_teleoperate``.  It deliberately excludes TeleVuer/Vuer and visualization.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

LOCKED_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
)


class G123PinocchioIk:
    """Ten-joint dual-arm IK with adapted PR152 pose cost and torque model."""

    def __init__(self, urdf_path: str | Path, *, palm_frames=None, locked_joints=None, pr152_objective=False):
        try:
            import casadi
            import pinocchio as pin
            from pinocchio import casadi as cpin
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "G1_23 IK requires the conda-forge ABI set "
                "pinocchio==3.1.0, casadi==3.6.7, numpy==1.26.4"
            ) from exc

        path = Path(urdf_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"G1_23 body23 URDF is missing: {path}")

        try:
            model = pin.buildModelFromUrdf(str(path))
            missing = [name for name in LOCKED_JOINT_NAMES if not model.existJointName(name)]
            if missing:
                raise RuntimeError(f"G1_23 URDF is missing locked joints: {missing}")
            lock_ids = [model.getJointId(name) for name in LOCKED_JOINT_NAMES]
            q0 = pin.neutral(model)
            if locked_joints is not None:
                if set(locked_joints) != set(LOCKED_JOINT_NAMES):
                    raise ValueError("g1_locked_joint_mapping")
                for name, value in locked_joints.items():
                    joint = model.joints[model.getJointId(name)]
                    if type(value) not in (int,float) or not np.isfinite(value) or not model.lowerPositionLimit[joint.idx_q] <= value <= model.upperPositionLimit[joint.idx_q]:
                        raise ValueError("g1_locked_joint_limit")
                    q0[joint.idx_q] = value
            reduced = pin.buildReducedModel(model, lock_ids, q0)
            from .kinematics import transform
            palms = [pin.SE3(transform(palm_frames[side])) if palm_frames else pin.SE3(np.eye(3), np.array([0.20,0.,0.])) for side in ("left","right")]
            reduced.addFrame(pin.Frame(
                "L_ee",
                reduced.getJointId("left_wrist_roll_joint"),
                palms[0],
                pin.FrameType.OP_FRAME,
            ))
            reduced.addFrame(pin.Frame(
                "R_ee",
                reduced.getJointId("right_wrist_roll_joint"),
                palms[1],
                pin.FrameType.OP_FRAME,
            ))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"failed to load G1_23 IK model: {exc}") from exc

        actual_names = tuple(str(name) for name in list(reduced.names)[1:])
        if reduced.nq != 10 or reduced.nv != 10 or actual_names != ARM_JOINT_NAMES:
            raise RuntimeError(
                "G1_23 reduced model contract mismatch: "
                f"nq={reduced.nq}, nv={reduced.nv}, joints={actual_names}"
            )

        self._pin = pin
        self._casadi = casadi
        self._model = reduced
        self._data = reduced.createData()
        self._lock = threading.Lock()
        self._history: deque[np.ndarray] = deque(maxlen=4)
        self._last_q = np.zeros(10)
        self._ready = False
        self._last_rejection = None
        self._last_solution = None
        self._strict_position = not pr152_objective
        self._warmup_ms: float | None = None

        cmodel = cpin.Model(reduced)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", reduced.nq, 1)
        ctf_l = casadi.SX.sym("tf_l", 4, 4)
        ctf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        left_frame = reduced.getFrameId("L_ee")
        right_frame = reduced.getFrameId("R_ee")
        self._left_frame = left_frame
        self._right_frame = right_frame
        translation_error = casadi.Function(
            "g1_23_translation_error",
            [cq, ctf_l, ctf_r],
            [casadi.vertcat(
                cdata.oMf[left_frame].translation - ctf_l[:3, 3],
                cdata.oMf[right_frame].translation - ctf_r[:3, 3],
            )],
        )
        def rotation_log(rotation):
            # log3's symbolic acos derivative can be NaN near identity on ARM64.
            # asin(s)/s has this smooth local series; keep log3 elsewhere.
            small = casadi.trace(rotation) > 3.0 - 1e-6
            skew = casadi.vertcat(rotation[2,1]-rotation[1,2],
                                  rotation[0,2]-rotation[2,0],
                                  rotation[1,0]-rotation[0,1]) / 2
            sin2 = casadi.sumsqr(skew)
            safe_rotation = casadi.if_else(small, casadi.DM([[0,-1,0],[1,0,0],[0,0,1]]), rotation)
            return casadi.if_else(small, skew*(1+sin2/6+3*sin2**2/40), cpin.log3(safe_rotation))

        rotation_error = casadi.Function(
            "g1_23_rotation_error",
            [cq, ctf_l, ctf_r],
            [casadi.vertcat(
                rotation_log(cdata.oMf[left_frame].rotation @ ctf_l[:3, :3].T),
                rotation_log(cdata.oMf[right_frame].rotation @ ctf_r[:3, :3].T),
            )],
        )

        opti = casadi.Opti()
        var_q = opti.variable(reduced.nq)
        var_q_last = opti.parameter(reduced.nq)
        param_left = opti.parameter(4, 4)
        param_right = opti.parameter(4, 4)
        opti.subject_to(opti.bounded(
            reduced.lowerPositionLimit,
            var_q,
            reduced.upperPositionLimit,
        ))
        opti.minimize(
            50.0 * casadi.sumsqr(translation_error(var_q, param_left, param_right))
            + 0.5 * casadi.sumsqr(rotation_error(var_q, param_left, param_right))
            + (0.02 if pr152_objective else 0.0) * casadi.sumsqr(var_q)
            + 0.1 * casadi.sumsqr(var_q - var_q_last)
        )
        # PR152 uses a bounded least-squares pose objective: operator poses need
        # not be exactly reachable by a five-DOF arm. Keep the legacy relative
        # mode's strict position contract without adding it to PR152 behavior.
        if self._strict_position:
            position_error = translation_error(var_q, param_left, param_right)
            for start in (0, 3):
                opti.subject_to(casadi.sumsqr(position_error[start:start+3]) <= .0149**2)
        opti.solver("ipopt", {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.max_cpu_time": 0.065,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        })
        self._opti = opti
        self._var_q = var_q
        self._var_q_last = var_q_last
        self._param_left = param_left
        self._param_right = param_right

    def ready(self) -> bool:
        """Return true only after one real IPOPT solve has succeeded."""

        with self._lock:
            return self._ready

    def reset(self, current_joint_positions) -> None:
        """Discard filter/seed state at every safe session boundary."""

        current = self._validated_vector(current_joint_positions, "current q")
        with self._lock:
            self._history.clear()
            self._last_q = current.copy()

    def current_targets(self, current_joint_positions) -> tuple[np.ndarray, np.ndarray]:
        """Return measured left/right EE transforms for a zero-motion self-test."""

        current = self._validated_vector(current_joint_positions, "current q")
        with self._lock:
            left, right = self._current_targets_locked(current)
            return left.copy(), right.copy()

    def warm_up(self, current_joint_positions, current_joint_velocities) -> dict:
        """Cold-load IPOPT by solving the measured current end-effector pose."""

        current_q = self._validated_vector(current_joint_positions, "current q")
        current_dq = self._validated_vector(current_joint_velocities, "current dq")
        started = time.monotonic()
        with self._lock:
            self._ready = False
            self._warmup_ms = None
            left, right = self._current_targets_locked(current_q)
            self._history.clear()
            self._last_q = current_q.copy()
        try:
            solved_q, _ = self.solve(left, right, current_q, current_dq)
        except Exception:
            with self._lock:
                self._ready = False
            raise
        elapsed_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        if np.max(np.abs(np.asarray(solved_q) - current_q)) > 0.5:
            raise RuntimeError("G1_23 IK warm-up diverged from measured posture")
        with self._lock:
            self._ready = True
            self._warmup_ms = elapsed_ms
            return {"ready": True, "warmup_ms": round(elapsed_ms, 3)}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ready": self._ready,
                "warmup_ms": (
                    None if self._warmup_ms is None else round(self._warmup_ms, 3)
                ),
                "history_depth": len(self._history),
                "last_rejection": copy.deepcopy(self._last_rejection),
                "last_solution": copy.deepcopy(self._last_solution),
            }

    def solve(self, left, right, current_q, current_dq):
        left = self._validated_transform(left, "left target")
        right = self._validated_transform(right, "right target")
        current_q = self._validated_vector(current_q, "current q")
        self._validated_vector(current_dq, "current dq")
        with self._lock:
            self._opti.set_initial(self._var_q, current_q)
            self._opti.set_value(self._var_q_last, current_q)
            self._opti.set_value(self._param_left, left)
            self._opti.set_value(self._param_right, right)
            try:
                solution = self._opti.solve()
            except RuntimeError as exc:
                try:
                    stats = self._opti.stats()
                except RuntimeError:
                    stats = {}
                self._last_rejection = {
                    "code": "ik_not_converged",
                    "monotonic_ns": time.monotonic_ns(),
                    "measured_q": current_q.tolist(),
                    "targets": [left.tolist(), right.tolist()],
                    "solver_status": stats.get("return_status", "unavailable"),
                    "iterations": stats.get("iter_count"),
                }
                raise ValueError("ik_not_converged") from exc
            raw_q = np.asarray(solution.value(self._var_q), dtype=float).reshape(10)
            if not np.all(np.isfinite(raw_q)):
                raise RuntimeError("G1_23 IK returned non-finite joints")
            # Reachability belongs to the solved pose, not filter tracking lag.
            actual = self._current_targets_locked(raw_q)
            position_errors = [float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                               for a, b in zip(actual, (left, right))]
            if self._strict_position and any(error > .015 for error in position_errors):
                self._last_rejection = {
                    "code": "ik_target_unreachable",
                    "monotonic_ns": time.monotonic_ns(),
                    "measured_q": current_q.tolist(),
                    "targets": [left.tolist(), right.tolist()],
                    "raw_q": raw_q.tolist(),
                    "position_error_m": position_errors,
                    "orientation_error_rad": [float(np.linalg.norm(
                        self._pin.log3(b[:3, :3].T @ a[:3, :3])))
                        for a, b in zip(actual, (left, right))],
                }
                raise ValueError("ik_target_unreachable")
            self._last_solution = {
                "position_error_m": position_errors,
                "position_policy": "strict_15mm" if self._strict_position else "pr152_weighted_pose",
            }
            self._history.append(raw_q.copy())
            # Upstream WeightedMovingFilter weights newest→oldest as
            # [0.4, 0.3, 0.2, 0.1].  During warm-up, normalize the prefix.
            weights = np.array([0.4, 0.3, 0.2, 0.1])[: len(self._history)]
            newest_first = np.stack(tuple(reversed(self._history)))
            q = np.sum(newest_first * (weights / weights.sum())[:, None], axis=0)
            tauff = self._pin.rnea(
                self._model,
                self._data,
                q,
                np.zeros(10),
                np.zeros(10),
            )
            tauff = np.asarray(tauff, dtype=float).reshape(10)
            if not np.all(np.isfinite(tauff)):
                raise RuntimeError("G1_23 inverse dynamics returned non-finite torques")
            self._last_q = q.copy()
            return q, tauff

    def _current_targets_locked(
        self,
        current_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._pin.framesForwardKinematics(self._model, self._data, current_q)
        left_pose = self._data.oMf[self._left_frame]
        right_pose = self._data.oMf[self._right_frame]
        left = np.eye(4)
        right = np.eye(4)
        left[:3, :3] = np.asarray(left_pose.rotation, dtype=float)
        left[:3, 3] = np.asarray(left_pose.translation, dtype=float).reshape(3)
        right[:3, :3] = np.asarray(right_pose.rotation, dtype=float)
        right[:3, 3] = np.asarray(right_pose.translation, dtype=float).reshape(3)
        return left, right

    @staticmethod
    def _validated_vector(value, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (10,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite ten-joint vector")
        return result

    @staticmethod
    def _validated_transform(value, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (4, 4) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite 4x4 transform")
        if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"{name} must be homogeneous")
        determinant = float(np.linalg.det(result[:3, :3]))
        if not np.isclose(determinant, 1.0, atol=1e-3):
            raise ValueError(f"{name} rotation must have determinant one")
        return result


__all__ = ["ARM_JOINT_NAMES", "G123PinocchioIk", "LOCKED_JOINT_NAMES"]
