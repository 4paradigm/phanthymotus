"""TRON2 7-DoF arm kinematics; NumPy only, without hardware I/O.

Positions are metres, angles are radians, quaternions are [x, y, z, w].
The endpoint is the existing URDF grasper link, NOT a calibrated fingertip TCP.
IK is a seeded local endpoint solver. Its iterations are NOT a commanded path;
joint interpolation to its result does not guarantee a Cartesian straight line.
No collision, velocity, acceleration, or robot calibration checks are performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import xml.etree.ElementTree as ET

import numpy as np


_PARTS = ("proximal_pitch", "proximal_roll", "proximal_yaw", "elbow",
          "wrist_yaw", "wrist_pitch", "wrist_roll")
JOINT_NAMES = tuple(f"{part}_{side}_Joint" for side in ("L", "R") for part in _PARTS)
EEF_FRAMES = {"left": "grasper_L_Link", "right": "grasper_R_Link"}
REFERENCE_LABEL = "URDF grasper link origin; not a calibrated grasp TCP"

# Retain the existing vendor.py/URDF intersection, then intersect the current
# official MoveJ table (reviewed 2026-11-10, SDK guide section 3.6.1):
# https://www.limxdynamics.com/en/documents/856848486581276672
# Use the table's literal radians, not a rounded conversion back from degrees.
# These are command/configuration bounds, not measured hardware/firmware limits.
_LEGACY_URDF_LOWER = (-math.pi, -0.2617993877991494, -3.6651914291880923,
             -2.6179938779914944, -1.7453, -math.pi / 4, -math.pi / 2,
             -math.pi, -3.193952531149623, -1.4835,
             -2.6179938779914944, -1.3962634015954636, -math.pi / 4, -math.pi / 2)
_LEGACY_URDF_UPPER = (2.6005, 3.193952531149623, 1.4835, 0.2618,
             1.3962634015954636, math.pi / 4, math.pi / 2,
             2.6005, 0.2617993877991494, 3.6651914291880923, 0.2618,
             1.7453, math.pi / 4, math.pi / 2)
_CURRENT_MOVEJ_LOWER = (-3.14159, -.261799, -3.66519, -2.61799, -1.74533, -.7505, -.7505,
                        -3.14159, -2.96706, -1.48353, -2.61799, -1.39626, -.7505, -.7505)
_CURRENT_MOVEJ_UPPER = (2.60054, 2.96706, 1.48353, .523599, 1.39626, .7505, .7505,
                        2.60054, .261799, 3.66519, .523599, 1.74533, .7505, .7505)
ARM_LOWER = tuple(max(old, current) for old, current in zip(_LEGACY_URDF_LOWER, _CURRENT_MOVEJ_LOWER))
ARM_UPPER = tuple(min(old, current) for old, current in zip(_LEGACY_URDF_UPPER, _CURRENT_MOVEJ_UPPER))
POSITION_TOLERANCE_M = 1e-4
ROTATION_TOLERANCE_RAD = 1e-4
MAX_ITERATIONS = 200


def _vector(value, size: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite real numbers") from exc
    if raw.shape != (size,) or any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat
    ):
        raise ValueError(f"{name} must contain exactly {size} real numbers (no bool/string)")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _exp(rotation_vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotation_vector))
    if not math.isfinite(theta):
        raise ValueError("rotation vector magnitude is not finite")
    if theta < 1e-8:
        k = _skew(rotation_vector)
        return np.eye(3) + k + 0.5 * k @ k
    k = _skew(rotation_vector / theta)
    return np.eye(3) + math.sin(theta) * k + (1. - math.cos(theta)) * k @ k


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Stable matrix to canonical xyzw, including rotations near pi."""
    trace = float(np.trace(rotation))
    if trace > 0.:
        s = 2. * math.sqrt(max(0., trace + 1.))
        q = np.array([(rotation[2, 1] - rotation[1, 2]) / s,
                      (rotation[0, 2] - rotation[2, 0]) / s,
                      (rotation[1, 0] - rotation[0, 1]) / s, s / 4.])
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2. * math.sqrt(max(0., 1. + rotation[i, i] - rotation[j, j] - rotation[k, k]))
        q = np.zeros(4)
        q[i] = s / 4.
        q[j] = (rotation[j, i] + rotation[i, j]) / s
        q[k] = (rotation[k, i] + rotation[i, k]) / s
        q[3] = (rotation[k, j] - rotation[j, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[3] < 0. else q


def _rotation(quaternion) -> np.ndarray:
    q = _vector(quaternion, 4, "target_quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or abs(norm - 1.) > 1e-6:
        raise ValueError("target_quaternion_xyzw must be a unit quaternion (tolerance 1e-6)")
    x, y, z, w = q / norm
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def _log(rotation: np.ndarray) -> np.ndarray:
    q = _quaternion(rotation)
    length = float(np.linalg.norm(q[:3]))
    if length < 1e-10:
        return 2. * q[:3]
    return q[:3] * (2. * math.atan2(length, q[3]) / length)


def _xml_vector(text: str, name: str) -> np.ndarray:
    try:
        return _vector([float(x) for x in text.split()], 3, name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid URDF {name}") from exc


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class Kinematics:
    """FK, seeded damped-least-squares IK, and explicit base/tool increments.

    Arm names are ``left`` and ``right`` in the URDF/vendor identity mapping.
    The caller must verify that mapping against the operator's camera convention.
    """

    def __init__(self, urdf):
        """Read both arm chains from the platform's supplied URDF."""
        try:
            root = ET.fromstring(urdf)
        except (ET.ParseError, TypeError) as exc:
            raise ValueError(f"Cannot read kinematic URDF: {exc}") from exc
        by_child = {}
        links = {element.get("name") for element in root.findall("link")}
        for element in root.findall("joint"):
            kind, name = element.get("type"), element.get("name")
            parent_node, child_node = element.find("parent"), element.find("child")
            if not name or parent_node is None or child_node is None:
                raise ValueError("URDF joint is missing name, parent, or child")
            parent, child = parent_node.get("link"), child_node.get("link")
            if parent not in links or child not in links or child in by_child:
                raise ValueError(f"Invalid URDF link topology at {name}")
            origin_node = element.find("origin")
            xyz = _xml_vector(origin_node.get("xyz", "0 0 0") if origin_node is not None else "0 0 0", "origin xyz")
            rpy = _xml_vector(origin_node.get("rpy", "0 0 0") if origin_node is not None else "0 0 0", "origin rpy")
            origin = np.eye(4)
            origin[:3, 3] = xyz
            origin[:3, :3] = (_exp(np.array([0., 0., rpy[2]])) @
                               _exp(np.array([0., rpy[1], 0.])) @
                               _exp(np.array([rpy[0], 0., 0.])))
            axis_node, limit = element.find("axis"), element.find("limit")
            axis = _xml_vector(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", "axis")
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12 or not math.isfinite(norm):
                raise ValueError(f"Invalid URDF axis at {name}")
            axis /= norm
            lower = upper = 0.
            if kind == "revolute":
                try:
                    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                except (TypeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"Missing URDF limits at {name}") from exc
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError(f"Invalid URDF limits at {name}")
            by_child[child] = _Joint(name, parent, child, kind, origin, axis, lower, upper)
        self._chains = {}
        self._limits = {}
        self._measurement_limits = {}
        for arm, tip in EEF_FRAMES.items():
            chain, child, visited = [], tip, set()
            while child != "base_Link":
                if child in visited or child not in by_child:
                    raise ValueError(f"No valid base_Link -> {tip} chain")
                visited.add(child)
                joint = by_child[child]
                if joint.kind not in ("fixed", "revolute"):
                    raise ValueError(f"Unsupported joint type in arm chain: {joint.kind}")
                chain.append(joint)
                child = joint.parent
            chain.reverse()
            moving = [joint for joint in chain if joint.kind == "revolute"]
            offset = 0 if arm == "left" else 7
            if tuple(j.name for j in moving) != JOINT_NAMES[offset:offset + 7]:
                raise ValueError(f"Unexpected 7-DoF joint mapping for {arm}")
            lower = np.maximum(ARM_LOWER[offset:offset+7], [j.lower for j in moving])
            upper = np.minimum(ARM_UPPER[offset:offset+7], [j.upper for j in moving])
            if np.any(lower >= upper):
                raise ValueError(f"Empty joint-limit intersection for {arm}")
            self._chains[arm] = tuple(chain)
            self._limits[arm] = (lower, upper)
            # Read actual/historical configurations throughout the model's
            # mechanical range even when today's command interface is narrower.
            self._measurement_limits[arm] = (np.array([j.lower for j in moving]),
                                             np.array([j.upper for j in moving]))

    def _configuration(self, arm, q7, *, command=False):
        if not isinstance(arm, str) or arm not in self._chains:
            raise ValueError("arm must be 'left' or 'right'")
        q = _vector(q7, 7, "joints_rad")
        lower, upper = (self._limits if command else self._measurement_limits)[arm]
        if np.any(q < lower) or np.any(q > upper):
            limits = "legacy vendor/URDF/current MoveJ command limit intersection" if command else "URDF mechanical limits"
            raise ValueError(f"{arm} joints_rad exceeds the {limits}")
        return q

    def command_seed(self, arm, measured, tolerance_rad):
        """Use a settled encoder reading as an IK seed within command bounds.

        FK and safety checks keep using the original encoder vector. Only the
        optimizer's initial guess can be adjusted, within a proven tolerance.
        """
        measured = self._configuration(arm, measured)
        lower, upper = self._limits[arm]
        seed = np.clip(measured, lower, upper)
        changes = []
        for index in np.flatnonzero(seed != measured):
            joint = JOINT_NAMES[int(index) + (7 if arm == "right" else 0)]
            delta = float(seed[index] - measured[index])
            if abs(delta) > tolerance_rad:
                raise ValueError(f"{joint} measured outside command limit by {abs(delta):.6g} rad; "
                                 f"verified seed tolerance {tolerance_rad:.6g} rad")
            changes.append({"joint": joint, "measured_rad": float(measured[index]),
                            "seed_rad": float(seed[index]), "correction_rad": delta})
        return seed, changes

    def _fk(self, arm, q):
        transform, origins, axes, index = np.eye(4), [], [], 0
        for joint in self._chains[arm]:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ joint.axis)
                motion = np.eye(4)
                motion[:3, :3] = _exp(joint.axis * q[index])
                transform = transform @ motion
                index += 1
        position = transform[:3, 3]
        jacobian = np.column_stack([np.r_[np.cross(a, position - p), a]
                                    for p, a in zip(origins, axes)])
        return transform, jacobian

    @staticmethod
    def _pose(arm, transform):
        return {"position_m": transform[:3, 3].tolist(),
                "quaternion_xyzw": _quaternion(transform[:3, :3]).tolist(),
                "frame": "base_Link", "eef_frame": EEF_FRAMES[arm],
                "reference_label": REFERENCE_LABEL,
                # TRON2 grasper geometry: the fingers extend along local -Z;
                # the two jaws move oppositely along the local Y axis.
                "finger_direction_base": (-transform[:3, 2]).tolist(),
                "jaw_opening_axis_base": transform[:3, 1].tolist()}

    def forward(self, arm, q7):
        """Read a pose within URDF mechanical limits, not a command admission."""
        q = self._configuration(arm, q7)
        return self._pose(arm, self._fk(arm, q)[0])

    def delta(self, arm, q7, translation_m, rotation_vector_rad, frame):
        """Apply increments about the current endpoint origin.

        ``base``: p'=p+dp, R'=Exp(dr)R. ``tool``: p'=p+Rdp,
        R'=R Exp(dr), where R is the PRE-increment tool orientation.
        This defines independent translation and rotation, not an SE(3) twist.
        q7 is feedback checked against URDF mechanical limits. A returned pose
        has not yet been checked for IK reachability or command-limit admission.
        """
        q = self._configuration(arm, q7)
        translation = _vector(translation_m, 3, "translation_m")
        rotation = _exp(_vector(rotation_vector_rad, 3, "rotation_vector_rad"))
        transform = self._fk(arm, q)[0]
        old_rotation = transform[:3, :3].copy()
        if frame == "base":
            transform[:3, 3] += translation
            transform[:3, :3] = rotation @ old_rotation
        elif frame == "tool":
            transform[:3, 3] += old_rotation @ translation
            transform[:3, :3] = old_rotation @ rotation
        else:
            raise ValueError("delta frame must be 'base' or 'tool'")
        if not np.all(np.isfinite(transform)):
            raise ValueError("delta pose is not finite")
        return self._pose(arm, transform)

    def solve(self, arm, target_position, target_quaternion, seed7):
        """Solve locally from seed7; reject invalid input or failure to converge.

        No random/global restart or wrapping to another joint branch is used.
        Iteration steps are bounded, but this does not certify an executable or
        collision-free path. Seed and output must satisfy command limits; an
        out-of-command-range seed is rejected, never silently clipped to recover.
        The caller must validate the full motion separately.
        """
        q = self._configuration(arm, seed7, command=True)
        target_p = _vector(target_position, 3, "target_position_m")
        target_r = _rotation(target_quaternion)
        lower, upper = self._limits[arm]
        weights = np.array([1., 1., 1., .2, .2, .2])

        def residual(configuration):
            transform, jacobian = self._fk(arm, configuration)
            error = np.r_[target_p - transform[:3, 3],
                          _log(target_r @ transform[:3, :3].T)]
            return error, jacobian

        error, jacobian = residual(q)
        for iteration in range(MAX_ITERATIONS + 1):
            position_error = float(np.linalg.norm(error[:3]))
            rotation_error = float(np.linalg.norm(error[3:]))
            if position_error <= POSITION_TOLERANCE_M and rotation_error <= ROTATION_TOLERANCE_RAD:
                return {"joints_rad": q.tolist(), "position_error_m": position_error,
                        "rotation_error_rad": rotation_error, "iterations": iteration,
                        "frame": "base_Link", "eef_frame": EEF_FRAMES[arm],
                        "reference_label": REFERENCE_LABEL}
            if iteration == MAX_ITERATIONS:
                break
            weighted_j = weights[:, None] * jacobian
            weighted_e = weights * error
            cost = float(weighted_e @ weighted_e)
            accepted = False
            # Local trust region, damping, and backtracking preserve the seed
            # neighbourhood without silently accepting an unconverged solution.
            for damping in (0.002, 0.01, 0.05):
                # At a bound, clipping an outward step repeatedly loses that
                # column's contribution while the other joints still assume
                # it moved. Re-solve the same Cartesian request on free axes.
                free = np.ones(len(q), dtype=bool)
                step = np.zeros_like(q)
                while np.any(free):
                    jac = weighted_j[:, free]
                    step[free] = jac.T @ np.linalg.solve(
                        jac @ jac.T + damping*damping*np.eye(6), weighted_e)
                    outward = free & (((q <= lower + 1e-10) & (step < 0)) |
                                      ((q >= upper - 1e-10) & (step > 0)))
                    if not np.any(outward):
                        break
                    step[outward] = 0.
                    free[outward] = False
                step *= min(1., .15 / max(float(np.max(np.abs(step))), 1e-15))
                for scale in (1., .5, .25, .125, .0625):
                    candidate = np.clip(q + scale * step, lower, upper)
                    candidate_error, candidate_j = residual(candidate)
                    weighted_candidate = weights * candidate_error
                    if float(weighted_candidate @ weighted_candidate) < cost - 1e-16:
                        q, error, jacobian = candidate, candidate_error, candidate_j
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break
        margin = np.minimum(q - lower, upper - q)
        index = int(np.argmin(margin))
        joint = JOINT_NAMES[index + (7 if arm == "right" else 0)]
        boundary = "lower" if q[index] - lower[index] <= upper[index] - q[index] else "upper"
        raise ValueError(f"IK did not converge from seed: position_error_m={position_error:.6g}, "
                         f"rotation_error_rad={rotation_error:.6g}, iterations={iteration}; "
                         f"closest command limit at final IK iterate: {joint} {boundary}, "
                         f"margin_rad={margin[index]:.5g}")
