"""Decode the Bumi visual-odometry pose without depending on ROS.

Freshness, ordering and segment transitions require stream state and are the
consumer's responsibility. This parser does not turn lost samples into poses.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PoseSample:
    stamp_ms: int
    frame_id: str
    child_frame_id: str
    segment_id: str
    position: tuple
    quaternion: tuple
    covariance: tuple


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("invalid vector length")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in value):
        raise ValueError("vector contains invalid values")
    return tuple(float(v) for v in value)


def decode_bumi_odom(sample):
    """Decode one tracking sample from Bumi's motus.odom/1 visual stream."""
    if not isinstance(sample, dict) or sample.get("schema") != "motus.odom/1":
        raise ValueError("unsupported odometry schema")
    stamp = sample.get("stamp_ms")
    if type(stamp) is not int or stamp <= 0:
        raise ValueError("invalid odometry timestamp")
    pose, vendor = sample.get("pose"), sample.get("vendor")
    if not isinstance(pose, dict) or not isinstance(vendor, dict):
        raise ValueError("pose and quality are required")
    if vendor.get("status") != "tracking":
        raise ValueError("odometry is not tracking")
    frame, child, segment = (pose.get("frame_id"), pose.get("child_frame_id"),
                             vendor.get("segment_id"))
    if any(not isinstance(v, str) or not v.strip() for v in (frame, child, segment)):
        raise ValueError("pose frames and segment are required")
    position = _vector(pose.get("position"), 3)
    quaternion = _vector(pose.get("quaternion"), 4)
    if abs(sum(v * v for v in quaternion) - 1.0) > 0.002:
        raise ValueError("quaternion is not normalized")
    covariance = _vector(pose.get("covariance"), 36)
    return PoseSample(stamp, frame, child, segment, position, quaternion, covariance)
