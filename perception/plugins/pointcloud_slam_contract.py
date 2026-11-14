"""Pure-Python spatial SLAM wire contracts (motus.slam_input/1 and motus.slam_frame/1).

Canonical copy: phanthymotus-driver/common/spatial_slam.py.
Runtime mirror: phanthymotus/perception/plugins/pointcloud_slam_contract.py.
Keep both implementations identical and update both contract test suites together.
No ROS, robot SDK, or NumPy dependencies; do not infer calibrated mounts or
verified clocks from plausible numeric values.
"""
from __future__ import annotations

import math
import re

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


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value != 'unknown'


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
    """Validate one tracking motus.odom/1 sample, without inventing unknown data."""
    if not isinstance(sample, dict) or sample.get('schema') != ODOM_SCHEMA:
        raise SpatialSlamError('invalid motus.odom/1 sample')
    units = sample.get('units')
    if not isinstance(units, dict) or units.get('length') != 'm':
        raise SpatialSlamError('SLAM pose length unit must be m')
    stamp = sample.get('stamp_ms')
    if type(stamp) is not int or stamp <= 0:
        raise SpatialSlamError('SLAM stamp_ms must be positive integer measurement time')
    vendor, pose = sample.get('vendor'), sample.get('pose')
    quality = vendor.get('motus_slam') if isinstance(vendor, dict) else None
    if not isinstance(quality, dict) or not isinstance(pose, dict):
        raise SpatialSlamError('pose and vendor.motus_slam are required')
    if quality.get('status') != 'tracking' or quality.get('stamp_source') != 'measurement':
        raise SpatialSlamError('SLAM tracking pose requires measurement stamp_source')
    for key in ('segment_id', 'clock_id'):
        if not _text(quality.get(key)):
            raise SpatialSlamError('vendor.motus_slam.' + key + ' is required')
    for key in ('frame_id', 'child_frame_id'):
        if not _text(pose.get(key)):
            raise SpatialSlamError('pose.' + key + ' is required')
    xyz, quat = pose.get('position'), pose.get('quaternion')
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or not all(_number(v) for v in xyz):
        raise SpatialSlamError('pose.position must be finite xyz meters')
    if not isinstance(quat, (list, tuple)) or len(quat) != 4 or not all(_number(v) for v in quat):
        raise SpatialSlamError('pose.quaternion must be finite xyzw')
    if abs(sum(v*v for v in quat) - 1.) > 0.002:
        raise SpatialSlamError('pose.quaternion must be normalized')
    covariance = pose.get('covariance')
    if covariance is not None and (not isinstance(covariance, (list, tuple))
            or len(covariance) != 36 or not all(_number(v) for v in covariance)):
        raise SpatialSlamError('pose.covariance must have 36 finite values or null')
    return sample


def validate_input_port(port):
    """Check one motus.slam_input/1 declaration; pair-level checks are separate."""
    if not isinstance(port, dict) or port.get('schema') != INPUT_SCHEMA:
        raise SpatialSlamError('input.schema must be motus.slam_input/1')
    fmt, kind = port.get('format'), port.get('kind')
    if (fmt, kind) not in ((POSE_FORMAT, 'pose'), (CLOUD_FORMAT, 'cloud')):
        raise SpatialSlamError('input format/kind must be native pose or PointCloud2')
    for key in ('topic', 'source_id', 'clock_id', 'reference_point'):
        if not _text(port.get(key)):
            raise SpatialSlamError('input.' + key + ' is required')
    if port.get('stamp_source') != 'measurement' or port.get('length_unit') != 'm':
        raise SpatialSlamError('input requires measurement timestamps and meters')
    if kind == 'pose':
        if not _text(port.get('child_frame_id')):
            raise SpatialSlamError('pose input.child_frame_id is required')
    else:
        if not _text(port.get('frame_id')):
            raise SpatialSlamError('cloud input.frame_id is required')
        if port.get('cloud_semantics') not in CLOUD_SEMANTICS:
            raise SpatialSlamError('cloud.cloud_semantics unsupported')
        if port.get('observation_model') not in OBSERVATION_MODELS:
            raise SpatialSlamError('cloud.observation_model unsupported')
        if port.get('timing_model') not in TIMING_MODELS:
            raise SpatialSlamError('cloud.timing_model unsupported')
        if not _text(port.get('frame_metadata_topic')):
            raise SpatialSlamError('cloud.frame_metadata_topic required')
        clock = port.get('clock')
        if not isinstance(clock, dict) or not _text(clock.get('mapping')):
            raise SpatialSlamError('cloud.clock.mapping required')
    return port


def validate_mount(mount, *, parent_frame=None, child_frame=None, require_calibrated=False):
    """Require a stated T_parent_child; a plausible matrix is not calibration."""
    if not isinstance(mount, dict):
        raise SpatialSlamError('extrinsic must be an object')
    for key in ('parent_frame', 'child_frame', 'source', 'version'):
        if not _text(mount.get(key)):
            raise SpatialSlamError('extrinsic.' + key + ' is required')
    if parent_frame is not None and mount['parent_frame'] != parent_frame:
        raise SpatialSlamError('extrinsic.parent_frame mismatch')
    if child_frame is not None and mount['child_frame'] != child_frame:
        raise SpatialSlamError('extrinsic.child_frame mismatch')
    if mount.get('calibrated') is not None and type(mount['calibrated']) is not bool:
        raise SpatialSlamError('extrinsic.calibrated must be boolean')
    if require_calibrated and mount.get('calibrated') is not True:
        raise SpatialSlamError('extrinsic must be calibrated for strict SLAM')
    mode = mount.get('mode', 'static')
    if mode not in ('static', 'dynamic'):
        raise SpatialSlamError('extrinsic.mode must be static or dynamic')
    if mode == 'dynamic':
        if mount['source'] != 'tf' or 'matrix' in mount:
            raise SpatialSlamError('dynamic extrinsic requires TF and no static matrix')
    else:
        rows = mount.get('matrix')
        if not isinstance(rows, (list, tuple)) or len(rows) != 4:
            raise SpatialSlamError('extrinsic.matrix must be rigid 4x4')
        if any(not isinstance(row, (list, tuple)) or len(row) != 4
               or not all(_number(x) for x in row) for row in rows):
            raise SpatialSlamError('extrinsic.matrix must be finite rigid 4x4')
        if any(abs(float(rows[3][j]) - [0., 0., 0., 1.][j]) > 1e-6 for j in range(4)):
            raise SpatialSlamError('extrinsic.matrix last row must be 0,0,0,1')
        rotation = [[float(rows[i][j]) for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[k][i]*rotation[k][j] for k in range(3))
                if abs(dot - (1. if i == j else 0.)) > 1e-6:
                    raise SpatialSlamError('extrinsic.matrix rotation must be orthonormal')
        a, b, c = rotation
        det = (a[0]*(b[1]*c[2]-b[2]*c[1])
               - a[1]*(b[0]*c[2]-b[2]*c[0])
               + a[2]*(b[0]*c[1]-b[1]*c[0]))
        if abs(det - 1.) > 1e-6:
            raise SpatialSlamError('extrinsic.matrix must be a proper rotation')
    return mount


def validate_frame_identity(packet, *, clock_id, strict=True,
                            require_digest=False, relation_id=None):
    """Validate motus.slam_frame/1 and return (clock, epoch, generation).

    This is the same identity accepted by the native Perception stream. The
    hash covers the original ROS PointCloud2 data bytes, not a display packet.
    """
    if not isinstance(packet, dict) or packet.get('schema') != FRAME_SCHEMA:
        raise SpatialSlamError('measurement provenance is required')
    stamp = packet.get('stamp_ns')
    if type(stamp) is not int or stamp <= 0:
        raise SpatialSlamError('invalid measurement stamp_ns')
    for key in ('frame_id', 'clock_id', 'clock_epoch', 'source_generation', 'source_clock'):
        if not _text(packet.get(key)):
            raise SpatialSlamError('measurement ' + key + ' is unknown')
    if packet['clock_id'] != clock_id:
        raise SpatialSlamError('measurement clock_id mismatch')
    source = packet.get('source_stamp')
    if not _number(source) or source < 0 or packet.get('source_unit') not in ('ms', 'ns'):
        raise SpatialSlamError('original measurement timestamp/unit is required')
    mapping = packet.get('mapping')
    if not isinstance(mapping, dict) or any(not _text(mapping.get(k))
                                            for k in ('method', 'version')):
        raise SpatialSlamError('measurement clock mapping is required')
    offset = mapping.get('offset_ns')
    if not _number(offset):
        raise SpatialSlamError('measurement mapping offset is required')
    factor = 1_000_000 if packet['source_unit'] == 'ms' else 1
    if int(source * factor) + offset != stamp:
        raise SpatialSlamError('measurement clock mapping does not reproduce stamp_ns')
    error = mapping.get('uncertainty_ns')
    if error is not None and (not _number(error) or error < 0):
        raise SpatialSlamError('invalid measurement mapping uncertainty')
    if relation_id is not None and (mapping.get('relation_id') != relation_id
                                    or mapping.get('verified') is not True or error is None):
        raise SpatialSlamError('measurement verified clock relation evidence is required')
    if strict and (mapping.get('verified') is not True or error is None):
        raise SpatialSlamError('measurement mapping is not verified')
    if require_digest:
        digest = packet.get('data_sha256')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise SpatialSlamError('measurement payload digest is required')
    return packet['clock_id'], packet['clock_epoch'], packet['source_generation']
