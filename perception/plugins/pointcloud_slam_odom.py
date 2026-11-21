"""Decode numeric odometry poses without depending on ROS."""

from dataclasses import dataclass, replace
import math

from .pointcloud_slam_contract import validate_tracking_pose
from .rgbd_odometry_contract import SIGMA_LIMIT_M
from .spatial_time import FrameContext


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
    pose_reference: str = None
    observation: FrameContext = None
    precise_stamp_ns: int = None
    observation_parents: tuple = ()
    prediction_anchor_ns: int = None
    external_continuity: bytes = None
    chassis_origin: tuple = None
    visual_origin: tuple = None
    external_origin: tuple = None

    @property
    def declared_origin(self):
        if self.external_origin is not None:
            return 'external_fusion', self.external_origin
        if self.chassis_origin is not None:
            return 'chassis', self.chassis_origin
        if self.visual_origin is not None:
            return 'visual', self.visual_origin
        return None

    @property
    def stamp_ns(self):
        if self.external_origin is not None and self.precise_stamp_ns is not None:
            return self.precise_stamp_ns
        return (self.observation.observation_stamp_ns if self.observation is not None else
                self.precise_stamp_ns if self.precise_stamp_ns is not None else self.stamp_ms*1_000_000)


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("invalid vector length")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in value):
        raise ValueError("vector contains invalid values")
    return tuple(float(v) for v in value)


def decode_legacy_odom(sample):
    """Decode one tracking sample from the legacy motus.odom/1 visual profile."""
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
    """Decode finite pose numbers without certifying producer metadata."""
    validate_tracking_pose(sample)
    pose = sample['pose']
    vendor = sample.get('vendor')
    vendor = vendor if isinstance(vendor, dict) else {}
    quality = vendor.get('motus_slam')
    quality = quality if isinstance(quality, dict) else {}
    stamp = pose.get('stamp_ms', sample.get('stamp_ms'))
    position = _vector(pose.get('position'), 3)
    quaternion = _vector(pose.get('quaternion'), 4)
    # Scaling first also keeps large finite quaternion components computable.
    scale = max(abs(v) for v in quaternion)
    scaled = tuple(v / scale for v in quaternion)
    norm = math.hypot(*scaled)
    quaternion = tuple(v / norm for v in scaled)
    covariance = pose.get('covariance')
    if covariance is not None:
        covariance = _vector(covariance, 36)
    precise = quality.get('stamp_ns')
    if precise is not None and type(precise) is not int:
        raise ValueError('precise pose timestamp must be integer nanoseconds')
    observation = quality.get('observation')
    observation = FrameContext.from_dict(observation) if isinstance(observation, dict) else None
    origin = (quality.get('segment_id', vendor.get('segment_id', '')),
              pose.get('frame_id', ''), pose.get('child_frame_id', ''))
    reference = quality.get('pose_reference', vendor.get('pose_reference'))
    external = (quality.get('pose_source') == 'external_fusion' and reference == 'body' and
                all(isinstance(value, str) and value for value in origin))
    fusion = vendor.get('external_fusion')
    if external and isinstance(fusion, dict) and fusion.get('update') == 'external':
        observed = fusion.get('observed')
        if isinstance(observed, list) and len(observed) == 6 and not all(observed[i] is True for i in (0, 1, 5)):
            raise ValueError('external motion does not constrain planar position and heading')
    return PoseSample(stamp, pose.get('frame_id', ''), pose.get('child_frame_id', ''),
                      origin[0], position, quaternion, covariance, quality.get('clock_id'),
                      reference, observation, precise, external_origin=origin if external else None)


def decode_slam_prediction(sample, anchor):
    """Read a bounded registration guess in an already measured odometry gauge."""
    if anchor is None or not isinstance(sample, dict) or sample.get('pose') is not None:
        raise ValueError('IMU prediction requires a measured anchor and an absent tracking pose')
    vendor = sample.get('vendor') or {}
    if not isinstance(vendor, dict):
        raise ValueError('invalid IMU prediction vendor')
    fusion, quality = vendor.get('fusion'), vendor.get('motus_slam')
    if not isinstance(fusion, dict) or not isinstance(quality, dict):
        raise ValueError('missing IMU prediction evidence')
    stamp, origin = fusion.get('prediction_stamp_ns'), fusion.get('prediction_anchor_ns')
    if (vendor.get('source') != 'rgbd_eskf' or quality.get('status') not in ('coasting', 'recovering') or
            fusion.get('mapping_allowed') is not False or fusion.get('translation_integrated') is not True or
            type(origin) is not int or type(stamp) is not int or stamp != quality.get('stamp_ns') or
            origin != anchor.stamp_ns or not 0 < stamp-origin <= 2_000_000_000):
        raise ValueError('invalid IMU prediction window')
    pose = decode_slam_odom(dict(sample, pose=fusion.get('prediction')))
    if (pose.segment_id, pose.frame_id, pose.child_frame_id, pose.clock_id) != (
            anchor.segment_id, anchor.frame_id, anchor.child_frame_id, anchor.clock_id) or pose.stamp_ns != stamp:
        raise ValueError('IMU prediction origin or capture time changed')
    if not pose.covariance or any(not 0 < pose.covariance[i] <= SIGMA_LIMIT_M**2 for i in (0, 7, 14)):
        raise ValueError('IMU prediction position covariance is out of bounds')
    return replace(pose, prediction_anchor_ns=origin)
