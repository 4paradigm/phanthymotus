"""Camera-start map axes without inventing body mounting or gravity evidence.

The native boundary is simulated from its documented initial_pose RPY value;
these tests do not execute RTAB-Map or establish physical floor alignment.
"""
import json
import struct

import numpy as np
import pytest

from plugins import rgbd_odometry_runtime as runtime
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.pointcloud_slam_preview import encode_mapping
from plugins.pointcloud_slam_transforms import pose_matrix
from plugins.rgbd_body_mount import CaptureMounts
from plugins.rgbd_fusion import InertialFilter, quaternion
from test_rgbd_body_mount import declaration
from test_rgbd_degradation import fused, native  # noqa: F401
from test_rgbd_odometry_status import sessions  # noqa: F401


def initial_pose(parameters):
    # Pinned OdometryROS uses identity when initial_pose is omitted, otherwise
    # parses x y z roll pitch yaw before starting native odometry.
    x, y, z, roll, pitch, yaw = map(float, parameters.get('initial_pose', '0 0 0 0 0 0').split())
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rz @ ry @ rx, [x, y, z]
    return pose


@pytest.mark.parametrize('options', [
    {},
    dict(imu_topic='/imu', short_term_compensation=True),
    dict(imu_topic='/imu', short_term_compensation=True, inertial_association=True),
])
def test_visual_world_basis_places_optical_down_below_camera_and_forward_on_map_xy(options):
    parameters = runtime.estimator_parameters(dict(frame_id='camera_optical', **options), 'run')
    pose = initial_pose(parameters)
    np.testing.assert_allclose(pose[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    assert parameters['frame_id'] == 'camera_optical'
    assert parameters['odom_frame_id'] == 'run_odom'
    assert not parameters.get('wait_imu_to_init', False)
    # Optical right/down/forward and a floor sample one metre below the camera.
    camera_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 2]])
    points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    packet = encode_mapping(points, pose, max_points=4, pose_reference='camera_optical')
    x, y, yaw, flags, count = struct.unpack('<fffBI', packet[:17])
    assert (x, y, flags, count) == (0., 0., 15, 4)  # camera heading is available
    assert yaw == pytest.approx(0., abs=1e-12)
    wire = np.frombuffer(packet[17:], dtype='<f4').reshape(-1, 3)
    # Core mapping renderer displays wire (x, y, z) as Three.js (x, z, -y).
    display = wire[:, [0, 2, 1]] * [1, 1, -1]
    np.testing.assert_allclose(display, [[0, 0, 1], [0, -1, 0], [1, 0, 0], [2, -1, 0]], atol=1e-12)


def test_native_attitude_initialization_keeps_ownership_of_gravity_alignment():
    parameters = runtime.estimator_parameters(dict(frame_id='camera_optical', imu_topic='/imu'), 'run')
    assert 'initial_pose' not in parameters
    assert parameters['wait_imu_to_init'] is True
    assert parameters['Odom/GuessSmoothingDelay'] == '0'
    assert parameters['frame_id'] == 'camera_optical'
    assert not parameters.get('motus_rotation_hint', False)


def test_fusion_fallback_keeps_camera_start_basis_and_unavailable_body_pose(fused):
    make, now, _ = fused
    session = make()
    parameters = runtime.estimator_parameters(session.calibration, session.segment)
    matrix = initial_pose(parameters)
    camera = dict(position=[0., 0., 0.], quaternion=quaternion(matrix[:3, :3]), covariance=[0.]*36)
    filter = InertialFilter(1_490_000_000, camera, [0, 0, 0], [0, -9.80665, 0], [0, 0, 0], [.03, 0, 0])
    filter.propagate(.01, [0, 0, 0], [0, -9.80665, 0])
    compensated = filter.camera_pose()
    np.testing.assert_allclose(compensated['position'], [0, 0, 0], atol=1e-12)
    session.mounts = CaptureMounts(declaration(), 'camera_optical', 'camera-global')
    session.fusion.last_tracking = (1_500_000_000, compensated, camera)
    session.fusion.last_visual = 1_500_000_000
    session.fusion.fault('IMU unavailable')
    now[0] = 1_520_000_000
    session._watchdog()
    assert session.fallback is not None and session.gate.state == 'recovering'
    for k in range(3):
        stamp = 1_600_000_000+k*100_000_000
        now[0] = stamp+20_000_000
        msg, info = native(session, stamp)
        for axis, value in zip(('x', 'y', 'z', 'w'), camera['quaternion']):
            setattr(msg.pose.pose.orientation, axis, value)
        session._result('pose', msg)
        session._result('info', info)
    sample = json.loads(session.publisher.messages[-1])
    assert sample['vendor']['fallback'] == 'visual_only'
    assert sample['vendor']['motus_slam']['segment_id'] == session.segment
    assert sample['vendor']['pose_reference'] == 'camera_optical'
    assert sample['vendor']['body_pose'] is None
    assert sample['vendor']['camera_mount']['status'] == 'unavailable'
    pose = decode_slam_odom(sample)
    assert pose.child_frame_id == 'camera_optical'
    np.testing.assert_allclose(pose_matrix(pose)[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    assert runtime.estimator_parameters(session.calibration, session.segment) == parameters
