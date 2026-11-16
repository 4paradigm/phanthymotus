"""Robot-independent runtime SLAM configuration, no ROS hardware required."""
from copy import deepcopy

import numpy as np
import pytest

from plugins.pointcloud_slam_config import infer_geometry, reconcile_explicit_extrinsic
from test_pointcloud_slam_stream import odom


MOUNT = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
IDENTITY = np.eye(4).tolist()


def sample(calibrated=False, with_mount=True):
    value = odom(1000, segment='run1')
    value['pose']['child_frame_id'] = 'run1_base'
    value['vendor']['motus_slam']['clock_id'] = 'ros'
    if with_mount:
        value['vendor']['motus_slam']['extrinsic'] = {
            'parent_frame': 'run1_base', 'child_frame': 'sensor_optical',
            'mode': 'static', 'matrix': deepcopy(MOUNT),
            'source': 'measured_mount' if calibrated else 'assumed_axis_convention',
            'version': 'v1', 'calibrated': calibrated,
        }
    return value


def test_uncalibrated_sensor_mount_is_accepted_and_provenance_is_preserved():
    settings, provenance = infer_geometry(sample(), 'sensor_optical', 1_000_000_000)
    assert settings['cloud_frame'] == 'sensor_optical'
    assert settings['child_frame'] == '{segment_id}_base'
    assert settings['clock_id'] == 'ros'
    np.testing.assert_allclose(settings['child_from_cloud'], MOUNT)
    assert provenance['extrinsic_source'] == 'producer_declared_uncalibrated'
    assert 'not_independent_clock_proof' in provenance['clock_relation']

    manual_settings, manual_provenance = infer_geometry(
        sample(), 'sensor_optical', 1_000_000_000, manual=MOUNT)
    np.testing.assert_allclose(manual_settings['child_from_cloud'], MOUNT)
    assert manual_provenance['extrinsic_source'] == 'manual_card'

    with pytest.raises(ValueError, match='child_from_cloud'):
        infer_geometry(sample(with_mount=False), 'sensor_optical', 1_000_000_000)


def test_calibrated_upstream_mount_is_accepted_and_conflicts_fail_closed():
    settings, provenance = infer_geometry(sample(calibrated=True), 'sensor_optical', 1_000_000_000)
    np.testing.assert_allclose(settings['child_from_cloud'], MOUNT)
    assert provenance['extrinsic_source'] == 'producer_calibrated'
    with pytest.raises(ValueError, match='conflicts'):
        infer_geometry(sample(calibrated=True), 'sensor_optical', 1_000_000_000,
                       manual=IDENTITY)


def test_identical_frame_can_use_identity_without_unmeasured_mount():
    data = odom(1000, segment='run1')
    config, provenance = infer_geometry(data, 'body', 1_000_000_000)
    np.testing.assert_allclose(config['child_from_cloud'], IDENTITY)
    assert provenance['extrinsic_source'] == 'identical_frame'


@pytest.mark.parametrize('change,match', [
    ('frame', 'extrinsic frame'),
    ('stamp', 'acquisition times'),
    ('mount_mode', 'dynamic mount'),
    ('mount_matrix', 'rigid'),
])
def test_invalid_frame_time_or_mount_is_rejected(change, match):
    value = sample(calibrated=True)
    frame, stamp = 'sensor_optical', 1_000_000_000
    if change == 'frame':
        frame = 'other_sensor'
    if change == 'stamp':
        stamp += 400_000_000
    if change == 'mount_mode':
        value['vendor']['motus_slam']['extrinsic']['mode'] = 'dynamic'
    if change == 'mount_matrix':
        value['vendor']['motus_slam']['extrinsic']['matrix'] = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match=match):
        infer_geometry(value, frame, stamp)


def test_explicit_producer_declaration_never_silently_overwrites_manual_mount():
    reconcile_explicit_extrinsic(None, MOUNT)
    reconcile_explicit_extrinsic(MOUNT, MOUNT)
    with pytest.raises(ValueError, match='conflicts'):
        reconcile_explicit_extrinsic(IDENTITY, MOUNT)


def test_segment_name_must_not_replace_incidental_characters_in_frame():
    # Segment "a" is not a prefix of frame "base_link".
    value = odom(1000, segment='a')
    config, provenance = infer_geometry(value, 'body', 1_000_000_000,
                                        manual=IDENTITY)
    assert config['child_frame'] == 'body'
    assert provenance['segment_id'] == 'a'
    value['pose']['child_frame_id'] = 'base_link'
    config, _ = infer_geometry(value, 'base_link', 1_000_000_000)
    assert config['child_frame'] == 'base_link'


def test_manual_mount_must_match_even_unverified_producer_mount():
    with pytest.raises(ValueError, match='conflicts'):
        infer_geometry(sample(calibrated=False), 'sensor_optical',
                       1_000_000_000, manual=IDENTITY)


def test_runtime_discovery_selects_declared_dynamic_mount_without_static_guess():
    from test_pointcloud_slam_capture_mount import camera, pose
    cfg, provenance = infer_geometry(pose(), 'camera_optical', 1_000_000_000,
                                     camera_info=camera())
    assert cfg['extrinsic_mode'] == 'dynamic' and cfg['child_from_cloud'] is None
    assert provenance['clock_accuracy_validated'] is False
    with pytest.raises(ValueError, match='static'):
        infer_geometry(pose(), 'camera_optical', 1_000_000_000,
                       camera_info=camera(), manual=IDENTITY)
