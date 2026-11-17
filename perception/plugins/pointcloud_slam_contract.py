"""Pure-Python spatial SLAM wire contracts (motus.slam_input/1 and motus.slam_frame/1).

Wire layouts are compatible with the Driver's spatial_slam formats. Consumer
admission intentionally checks numeric structure rather than producer evidence.
Physical conventions are the data provider's responsibility. No ROS, robot
SDK or NumPy dependency is needed for these parsers.
"""
from __future__ import annotations

import math

ODOM_SCHEMA = 'motus.odom/1'
POSE_PROFILE = 'motus_slam/1'
INPUT_SCHEMA = 'motus.slam_input/1'
FRAME_SCHEMA = 'motus.slam_frame/1'
CLOUD_FORMAT = 'sensor/pointcloud2'
POSE_FORMAT = 'state/odom'
CLOUD_SEMANTICS = ('depth_frame', 'scan')
OBSERVATION_MODELS = ('pinhole', 'lidar')
TIMING_MODELS = ('instantaneous', 'deskewed')


class SpatialSlamError(ValueError):
    """A producer claim that cannot safely be used for SLAM."""



def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def validate_slam_capability(interface):
    """Require an explicit SLAM pose capability; velocity/pose_drift are not proof."""
    if not isinstance(interface, dict) or interface.get('schema') != ODOM_SCHEMA:
        raise SpatialSlamError('odom_interface.schema must be motus.odom/1')
    capability = interface.get('slam_pose')
    if (not isinstance(capability, dict) or capability.get('available') is not True
            or capability.get('profile') != POSE_PROFILE):
        raise SpatialSlamError('odom_interface.slam_pose requires motus_slam/1 availability')
    return interface


def validate_tracking_pose(sample):
    """Require a numeric pose; alignment and quality remain producer responsibilities."""
    if not isinstance(sample, dict) or sample.get('schema') != ODOM_SCHEMA:
        raise SpatialSlamError('invalid motus.odom/1 sample')
    pose = sample.get('pose')
    if not isinstance(pose, dict):
        raise SpatialSlamError('numeric pose is required')
    stamp = pose.get('stamp_ms', sample.get('stamp_ms'))
    if type(stamp) is not int:
        raise SpatialSlamError('SLAM stamp_ms must be integer time')
    xyz, quat = pose.get('position'), pose.get('quaternion')
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or not all(_number(v) for v in xyz):
        raise SpatialSlamError('pose.position must be finite xyz')
    if not isinstance(quat, (list, tuple)) or len(quat) != 4 or not all(_number(v) for v in quat):
        raise SpatialSlamError('pose.quaternion must be finite xyzw')
    if not math.hypot(*quat):
        raise SpatialSlamError('pose.quaternion must have nonzero norm')
    covariance = pose.get('covariance')
    if covariance is not None and (not isinstance(covariance, (list, tuple))
            or len(covariance) != 36 or not all(_number(v) for v in covariance)):
        raise SpatialSlamError('pose.covariance must have 36 finite values or null')
    return sample


def validate_input_port(port):
    """Validate wire routing; descriptive source/time/geometry labels are optional."""
    if not isinstance(port, dict) or port.get('schema') != INPUT_SCHEMA:
        raise SpatialSlamError('input.schema must be motus.slam_input/1')
    if port.get('format') not in (POSE_FORMAT,CLOUD_FORMAT):
        raise SpatialSlamError('input format must be native pose or PointCloud2')
    return port


def validate_mount(mount, *, parent_frame=None, child_frame=None, require_calibrated=False):
    """Check numeric 4x4 data; physical conventions belong to the producer."""
    if not isinstance(mount, dict):
        raise SpatialSlamError('extrinsic must be an object')
    if mount.get('mode') == 'dynamic' and mount.get('matrix') is None:
        return mount
    rows = mount.get('matrix')
    if (not isinstance(rows, (list, tuple)) or len(rows) != 4 or
            any(not isinstance(row, (list, tuple)) or len(row) != 4 or
                not all(_number(x) for x in row) for row in rows)):
        raise SpatialSlamError('extrinsic.matrix must be finite numeric 4x4')
    return mount


def validate_frame_identity(packet, *, clock_id, strict=True,
                            require_digest=False, relation_id=None):
    """Read optional sidecar labels without treating them as admission evidence."""
    if not isinstance(packet, dict) or packet.get('schema') != FRAME_SCHEMA:
        raise SpatialSlamError('invalid measurement packet')
    if type(packet.get('stamp_ns')) is not int:
        raise SpatialSlamError('invalid measurement stamp_ns')
    return packet.get('clock_id'), packet.get('clock_epoch'), packet.get('source_generation')
