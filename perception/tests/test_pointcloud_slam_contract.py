"""Cross-repository, ROS-free contract acceptance tests. Keep both copies aligned."""
from copy import deepcopy

import pytest

from plugins.pointcloud_slam_contract import (
    ODOM_SCHEMA, POSE_PROFILE, INPUT_SCHEMA, FRAME_SCHEMA,
    POSE_FORMAT, CLOUD_FORMAT,
    SpatialSlamError, validate_slam_capability, validate_tracking_pose,
    validate_input_port, validate_mount, validate_frame_identity,
)

IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]


def capability():
    return {'schema': 'motus.odom/1', 'slam_pose':
            {'available': True, 'profile': 'motus_slam/1'}}


def sample():
    return {
        'schema': 'motus.odom/1', 'stamp_ms': 1000,
        'units': {'length': 'm'},
        'pose': {'frame_id': 'run_odom', 'child_frame_id': 'base',
                 'position': [1., 2., 0.], 'quaternion': [0., 0., 0., 1.],
                 'covariance': None},
        'vendor': {'motus_slam': {'status': 'tracking', 'segment_id': 'run',
                                 'stamp_source': 'measurement', 'clock_id': 'ros'}},
    }


def port(kind):
    shared = {
        'schema': 'motus.slam_input/1',
        'source_id': 'sensor-1', 'clock_id': 'ros',
        'reference_point': 'base', 'length_unit': 'm',
        'stamp_source': 'measurement',
    }
    if kind == 'pose':
        return {**shared, 'topic': '/odom', 'kind': 'pose',
                'format': 'state/odom', 'child_frame_id': 'base'}
    return {**shared, 'topic': '/cloud', 'kind': 'cloud',
            'format': 'sensor/pointcloud2', 'frame_id': 'camera',
            'cloud_semantics': 'depth_frame', 'observation_model': 'pinhole',
            'timing_model': 'instantaneous', 'frame_metadata_topic': '/cloud/meta',
            'clock': {'mapping': 'device_to_ros', 'verified': False}}


def mount():
    return {'parent_frame': 'base', 'child_frame': 'camera',
            'mode': 'static', 'source': 'measured_mount', 'version': 'revision/1',
            'calibrated': True, 'matrix': deepcopy(IDENTITY)}


def frame():
    return {'schema': 'motus.slam_frame/1', 'stamp_ns': 1000000000,
            'frame_id': 'camera', 'clock_id': 'ros', 'clock_epoch': 'epoch-1',
            'source_generation': 'generation-1', 'source_stamp': 1000,
            'source_unit': 'ms', 'source_clock': 'sensor_clock',
            'mapping': {'method': 'source_to_ros', 'version': '1',
                        'offset_ns': 0, 'verified': True,
                        'uncertainty_ns': 100, 'relation_id': 'calibrated_pair'},
            'data_sha256': 'a' * 64}


def test_constants_and_explicit_capability():
    assert (ODOM_SCHEMA, POSE_PROFILE, INPUT_SCHEMA, FRAME_SCHEMA) == (
        'motus.odom/1', 'motus_slam/1', 'motus.slam_input/1', 'motus.slam_frame/1')
    assert validate_slam_capability(capability()) == capability()
    invalid = deepcopy(capability())
    invalid['slam_pose']['available'] = False
    with pytest.raises(SpatialSlamError, match='slam_pose'):
        validate_slam_capability(invalid)
    with pytest.raises(SpatialSlamError, match='schema'):
        validate_slam_capability({'schema': 'another'})


def test_tracking_pose_and_nullable_covariance():
    valid = sample()
    assert validate_tracking_pose(valid) is valid
    valid['pose']['covariance'] = [1e90]*36
    assert validate_tracking_pose(valid) is valid
    for field,value in [('quaternion',[0,0,0,0]), ('position',[float('nan'),0,0]),
                        ('covariance',[0]*35)]:
        bad = deepcopy(valid); bad['pose'][field] = value
        with pytest.raises(SpatialSlamError): validate_tracking_pose(bad)


def test_input_port_checks_wire_routing_and_retains_other_metadata():
    for kind in ('pose','cloud'):
        item = port(kind); item.update(clock_id='unknown', stamp_source='unknown', length_unit='custom',kind='producer-description')
        assert validate_input_port(item) is item
    invalid = port('cloud'); invalid['format'] = 'sensor/pointcloud'
    with pytest.raises(SpatialSlamError, match='format'): validate_input_port(invalid)


def test_mount_labels_and_calibration_are_producer_metadata():
    value = mount(); value.update(calibrated=False, source='', version='', child_frame='elsewhere')
    assert validate_mount(value, parent_frame='different', child_frame='other', require_calibrated=True) is value
    value['matrix'] = [[1]]
    with pytest.raises(SpatialSlamError): validate_mount(value)


def test_frame_identity_does_not_require_source_clock_or_digest_proofs():
    value = frame(); value.update(data_sha256='not-a-digest', clock_id='other', source_generation='unknown')
    value['mapping'].update(verified=False, offset_ns=99)
    assert validate_frame_identity(value, clock_id='ros', require_digest=True, relation_id='other') == (
        'other','epoch-1','unknown')
    from plugins.pointcloud_slam_input import frame_identity
    assert frame_identity(value, clock_id='ros', payload=True) == ('other','epoch-1','unknown')
    value['stamp_ns'] = float('nan')
    with pytest.raises(SpatialSlamError): validate_frame_identity(value, clock_id='ros')


def test_perception_pose_decoder_checks_shared_profile():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value = sample()
    decoded = decode_slam_odom(value)
    assert decoded.segment_id == 'run' and decoded.clock_id == 'ros'
    bad = deepcopy(value)
    bad['pose']['quaternion'] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match='quaternion'):
        decode_slam_odom(bad)


def test_perception_explicit_input_is_compatible_with_common_declaration():
    from plugins.pointcloud_slam_input import resolve_inputs
    pose, cloud = port('pose'), port('cloud')
    pose['clock'] = {'mapping': 'device_to_ros', 'verified': True}
    cloud['clock']['verified'] = True
    cloud['extrinsic'] = mount()
    original = {'/odom': deepcopy(pose), '/cloud': deepcopy(cloud)}
    bindings = {'/odom': POSE_FORMAT, '/cloud': CLOUD_FORMAT}
    result = resolve_inputs({'/odom': pose, '/cloud': cloud}, bindings)
    assert result['settings']['cloud_frame'] == 'camera'
    assert result['settings']['child_frame'] == 'base'
    assert result['settings']['clock_id'] == 'ros'
    assert result['settings']['child_from_cloud'] == IDENTITY
    assert result['frame_metadata_topic'] == '/cloud/meta'
    assert {'/odom': pose, '/cloud': cloud} == original
    cloud['extrinsic']['calibrated'] = False
    uncalibrated = resolve_inputs({'/odom': pose, '/cloud': cloud}, bindings)
    assert uncalibrated['extrinsic']['calibrated'] is False
    assert uncalibrated['settings']['child_from_cloud'] == IDENTITY


@pytest.mark.parametrize('stamp', [None, True, 1.5, '1000'])
def test_explicit_pose_stamp_cannot_fall_back_to_twist_time(stamp):
    value = sample()
    value['pose']['stamp_ms'] = stamp
    with pytest.raises(SpatialSlamError, match='stamp_ms'):
        validate_tracking_pose(value)


def test_optional_diagnostic_evidence_cannot_override_numeric_pose_time():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value = sample(); value['pose']['stamp_ms'] = 1000
    quality = value['vendor']['motus_slam']
    quality['stamp_ns'] = 1_000_000_123
    quality['measurement'] = {'stamp_ns': 2_000_000_000}
    assert validate_tracking_pose(value) is value
    assert decode_slam_odom(value).stamp_ns == 1_000_000_123
