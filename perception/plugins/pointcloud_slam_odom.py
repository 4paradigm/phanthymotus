"""Decode the Bumi visual-odometry pose without depending on ROS.

Freshness, ordering and segment transitions require stream state and are the
consumer's responsibility. This parser does not turn lost samples into poses.
"""

from dataclasses import dataclass
import math

from .pointcloud_slam_contract import validate_tracking_pose


@dataclass(frozen=True)
class PoseSample:
    stamp_ms: int
    frame_id: str
    child_frame_id: str
    segment_id: str
    position: tuple
    quaternion: tuple
    covariance: tuple
    clock_id: str = None


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


def decode_slam_odom(sample):
    """Decode the common SLAM profile; stream freshness is checked separately."""
    validate_tracking_pose(sample)
    if not isinstance(sample, dict) or sample.get("schema") != "motus.odom/1":
        raise ValueError("unsupported odometry schema")
    units = sample.get("units")
    if not isinstance(units, dict) or units.get("length") != "m":
        raise ValueError("pose length must be in meters")
    stamp = sample.get("stamp_ms")
    if type(stamp) is not int or stamp <= 0:
        raise ValueError("invalid odometry timestamp")
    pose, vendor = sample.get("pose"), sample.get("vendor")
    quality = vendor.get("motus_slam") if isinstance(vendor, dict) else None
    if not isinstance(pose, dict) or not isinstance(quality, dict):
        raise ValueError("pose and SLAM metadata are required")
    if quality.get("status") != "tracking" or quality.get("stamp_source") != "measurement":
        raise ValueError("odometry is not a tracking measurement")
    frame, child = pose.get("frame_id"), pose.get("child_frame_id")
    segment, clock = quality.get("segment_id"), quality.get("clock_id")
    if any(not isinstance(v, str) or not v.strip() for v in (frame, child, segment, clock)):
        raise ValueError("pose frames, segment and clock are required")
    position = _vector(pose.get("position"), 3)
    quaternion = _vector(pose.get("quaternion"), 4)
    if abs(sum(v * v for v in quaternion) - 1.0) > 0.002:
        raise ValueError("quaternion is not normalized")
    covariance = pose.get("covariance")
    if covariance is not None:
        covariance = _vector(covariance, 36)
    return PoseSample(stamp, frame, child, segment, position, quaternion, covariance, clock)
