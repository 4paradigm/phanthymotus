"""Reading `motus.odom/1`, from the consumer's side.

**This deliberately re-implements the reader rather than importing one.** The
producing half lives in `phanthymotus-driver/common/odom.py`, in a different
repository and a different container image, so importing it is not available —
but even if it were, the project's rule for cross-repo protocols applies:

    the spec is a document, not a shared library; both sides implement against
    it and each run their own contract tests

(CLAUDE.md, on `motus.vla/1`, and the same reasoning that keeps `motus.control/1`
from having a shared schema package.) A shared module couples the two release
cadences to each other's implementation details; the `schema` field is what
decouples them.

So what is here is small on purpose — three functions and one rule.

── the rule ─────────────────────────────────────────────────────────────────

**`None` is not `0.0`.** An axis a robot does not measure arrives as JSON
`null`, and it must stay `None` all the way to the stuck detector. A robot with
no odometry reporting 0.0 m/s is indistinguishable from a robot that has
stopped, and the stuck detector would then fire on every robot without
odometry, on its first step. That is the failure the whole format is shaped
around, and this file is where a consumer would undo it — one `or 0.0` is all
it takes.

Spec: `phanthymotus-driver/README_dev.md` § "Robot Odometry (motus.odom/1)".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

SCHEMA = "motus.odom/1"

# Must match the producer's order, which is `motus.control/1` twist order. The
# contract test in tests/test_navi_odom.py pins it against the descriptor this
# card negotiates with, so a divergence is caught here rather than as a robot
# comparing its forward speed against its yaw rate.
AXES = ("vx", "vy", "vz", "wx", "wy", "wz")

BODY_FRAME = "body"


def planar_feedback(odom):
    """Use measured axes first, filling missing axes with recent pose deltas."""
    sample = odom or {}
    estimated = sample.get("pose_feedback", (None, None, None))
    motion = tuple(value if value is not None and math.isfinite(value) else fallback
                   for value, fallback in zip((sample.get(axis) for axis in ("vx", "vy", "wz")), estimated))
    return motion if all(v is not None for v in motion) else None


@dataclass(frozen=True)
class PlanarPose:
    stamp_ms: int
    frame: tuple
    x: float
    y: float
    yaw: float

    def relative_to(self, origin):
        dx, dy = self.x - origin.x, self.y - origin.y
        c, s = math.cos(origin.yaw), math.sin(origin.yaw)
        return c * dx + s * dy, -s * dx + c * dy, math.remainder(self.yaw - origin.yaw, math.tau)

    def feedback_since(self, previous, max_gap_ms):
        dt = self.stamp_ms - previous.stamp_ms
        if self.frame != previous.frame or not 0 < dt <= max_gap_ms:
            return None
        motion = tuple(-1000 * v / dt for v in previous.relative_to(self))
        return motion if all(math.isfinite(v) for v in motion) else None


def planar_pose(sample, stamp_ms):
    """Read a body-oriented tracking pose and preserve its continuity identifiers."""
    pose = sample.get("pose")
    if sample.get("schema") != SCHEMA or stamp_ms is None or not isinstance(pose, dict):
        return None
    vendor = sample.get("vendor")
    vendor = vendor if isinstance(vendor, dict) else {}
    slam = vendor.get("motus_slam")
    slam = slam if isinstance(slam, dict) else {}
    if slam.get("status", vendor.get("status", "tracking")) != "tracking":
        return None
    if slam.get("stamp_source", "measurement") != "measurement":
        return None
    parent, child = pose.get("frame_id"), pose.get("child_frame_id")
    if not all(isinstance(frame, str) and frame for frame in (parent, child)):
        return None
    reference = vendor.get("pose_reference")
    if reference not in ("body", "body_axes") and (
            reference is not None or child not in ("base_link", "base_footprint", "body")):
        return None
    pos, quat = pose.get("position"), pose.get("quaternion")
    if not all(isinstance(vec, list) and len(vec) == size and
               all(type(value) in (float, int) and math.isfinite(value) for value in vec)
               for vec, size in ((pos, 3), (quat, 4))):
        return None
    norm = math.hypot(*quat)
    if not .99 <= norm <= 1.01:
        return None
    x, y, z, w = (value / norm for value in quat)
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    frame = (parent, child, reference, slam.get("segment_id", vendor.get("segment_id")),
             vendor.get("calibration_id"))
    return PlanarPose(stamp_ms, frame, pos[0], pos[1], yaw)


def axis_of(sample: dict, name: str):
    """One axis out of a sample, or `None` if unmeasured, absent or unusable.

    Returns `None` — never 0.0 — for every kind of absence, including a sample
    in the wrong frame. A world-frame reading is real odometry, but comparing
    it against a body-frame command needs the robot's heading and a rotation;
    treating it as body-frame silently gives a wrong answer whenever the robot
    is not facing along world x.
    """
    if not isinstance(sample, dict):
        return None
    if sample.get("schema") != SCHEMA:
        return None
    if sample.get("frame") != BODY_FRAME:
        return None
    try:
        index = AXES.index(name)
    except ValueError:
        return None
    values = sample.get("twist") or []
    if index >= len(values):
        return None
    value = values[index]
    if value is None or isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def interface_provides(info: dict, name: str) -> bool:
    """Whether a driver declared, at start, that it measures this axis.

    Read once from the state card's `info()`. The per-sample `None`s say "not
    in this frame"; this says "not ever", which is what lets the card tell an
    operator that it has no stuck protection instead of appearing to have one.
    """
    interface = (info or {}).get("odom_interface") or {}
    if interface.get("schema") != SCHEMA:
        return False
    return name in (interface.get("provides") or [])


def measured_obstacle_pose(sample, camera_vendor):
    """Match the measured pose's own clock and stamp to the depth camera clock."""
    clock = camera_vendor.get("clock_reference")
    vendor = sample.get("vendor")
    slam = vendor.get("motus_slam") if isinstance(vendor, dict) else None
    if (not isinstance(clock, dict) or not clock.get("domain_id")
            or clock.get("clock_id") != "host_realtime" or clock.get("timestamp_basis") != "host_realtime"
            or camera_vendor.get("timestamp_semantics") != "source_capture_mapped"
            or not isinstance(slam, dict) or slam.get("stamp_source") != "measurement"):
        return None
    reference = slam.get("clock_reference")
    if not isinstance(reference, dict) or any(reference.get(k) != clock[k] for k in
                                               ("clock_id", "timestamp_basis", "domain_id")):
        return None
    measurement = slam.get("measurement")
    if not isinstance(measurement, dict) or measurement.get("clock_id") != clock["clock_id"]:
        return None
    mapping = measurement.get("mapping")
    if not isinstance(mapping, dict) or mapping.get("verified") is not True:
        return None
    ns = measurement.get("stamp_ns")
    pose = sample.get("pose")
    if type(ns) is not int or ns <= 0 or not isinstance(pose, dict) or pose.get("stamp_ms") != ns // 10**6:
        return None
    mount = camera_vendor.get("body_mount")
    if (measurement.get("frame_id") != pose.get("frame_id")
            or not slam.get("segment_id") or not measurement.get("clock_epoch")
            or (isinstance(mount, dict) and mount.get("parent_frame") != pose.get("child_frame_id"))):
        return None
    body = planar_pose(sample, ns // 10**6)
    if body is None:
        return None
    continuity = (clock["domain_id"], measurement["clock_epoch"], measurement.get("source_generation"))
    return ns, replace(body, frame=body.frame + continuity)


def obstacle_pose_at(history, ns):
    for index, (stamp, pose) in enumerate(history):
        if stamp == ns:
            return pose
        if stamp > ns:
            if not index:
                return None
            previous, before = history[index-1]
            part = (ns - previous) / (stamp - previous)
            return PlanarPose(ns // 10**6, pose.frame,
                              before.x + part * (pose.x - before.x),
                              before.y + part * (pose.y - before.y),
                              before.yaw + part * math.remainder(pose.yaw - before.yaw, math.tau))
    return None
