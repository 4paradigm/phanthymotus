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
    valid['pose']['covariance'] = [0.] * 36
    assert validate_tracking_pose(valid) is valid
    for change in ('lost', 'stale_stamp', 'bad_quaternion', 'wrong_units',
                   'bad_covariance', 'no_frame'):
        bad = deepcopy(valid)
        if change == 'lost':
            bad['vendor']['motus_slam']['status'] = 'lost'
        elif change == 'stale_stamp':
            bad['vendor']['motus_slam']['stamp_source'] = 'received'
        elif change == 'bad_quaternion':
            bad['pose']['quaternion'] = [0., 0., 0., 0.]
        elif change == 'wrong_units':
            bad['units']['length'] = 'cm'
        elif change == 'bad_covariance':
            bad['pose']['covariance'] = [0.] * 35
        elif change == 'no_frame':
            bad['pose']['child_frame_id'] = ''
        with pytest.raises(SpatialSlamError):
            validate_tracking_pose(bad)


def test_input_port_geometry_and_formats():
    for kind in ('pose', 'cloud'):
        item = port(kind)
        assert validate_input_port(item) is item
    invalid = port('cloud')
    invalid['format'] = 'sensor/pointcloud'
    with pytest.raises(SpatialSlamError, match='format/kind'):
        validate_input_port(invalid)
    invalid = port('cloud')
    invalid['cloud_semantics'] = 'accumulated_map'
    with pytest.raises(SpatialSlamError, match='cloud_semantics'):
        validate_input_port(invalid)
    invalid = port('cloud')
    invalid['clock']['mapping'] = ''
    with pytest.raises(SpatialSlamError, match='clock.mapping'):
        validate_input_port(invalid)


def test_rigid_mount_and_calibration_are_distinct():
    value = mount()
    assert validate_mount(value, parent_frame='base', child_frame='camera',
                          require_calibrated=True) is value
    assumed = deepcopy(value)
    assumed['calibrated'] = False
    assert validate_mount(assumed, parent_frame='base', child_frame='camera') is assumed
    with pytest.raises(SpatialSlamError, match='calibrated'):
        validate_mount(assumed, require_calibrated=True)
    wrong_frame = deepcopy(value)
    wrong_frame['child_frame'] = 'lidar'
    with pytest.raises(SpatialSlamError, match='child_frame'):
        validate_mount(wrong_frame, child_frame='camera')
    reflected = deepcopy(value)
    reflected['matrix'][0][0] = -1.
    with pytest.raises(SpatialSlamError, match='proper rotation'):
        validate_mount(reflected)
    skew = deepcopy(value)
    skew['matrix'][0][1] = .1
    with pytest.raises(SpatialSlamError, match='orthonormal'):
        validate_mount(skew)
    dynamic = {k: value[k] for k in ('parent_frame', 'child_frame', 'version')}
    dynamic.update(mode='dynamic', source='tf', calibrated=True)
    assert validate_mount(dynamic) is dynamic
    dynamic['matrix'] = deepcopy(IDENTITY)
    with pytest.raises(SpatialSlamError, match='dynamic'):
        validate_mount(dynamic)


def test_frame_identity_exact_stamp_verified_clock_relation_and_digest():
    value = frame()
    assert validate_frame_identity(value, clock_id='ros', require_digest=True,
                                   relation_id='calibrated_pair') == (
        'ros', 'epoch-1', 'generation-1')
    bad = deepcopy(value)
    bad['mapping']['offset_ns'] = 1
    with pytest.raises(SpatialSlamError, match='reproduce'):
        validate_frame_identity(bad, clock_id='ros')
    bad = deepcopy(value)
    bad['mapping']['verified'] = False
    with pytest.raises(SpatialSlamError, match='verified'):
        validate_frame_identity(bad, clock_id='ros')
    assert validate_frame_identity(bad, clock_id='ros', strict=False) == (
        'ros', 'epoch-1', 'generation-1')
    with pytest.raises(SpatialSlamError, match='relation'):
        validate_frame_identity(bad, clock_id='ros', strict=False,
                                relation_id='calibrated_pair')
    bad = deepcopy(value)
    bad['data_sha256'] = 'NOT_A_HASH'
    with pytest.raises(SpatialSlamError, match='digest'):
        validate_frame_identity(bad, clock_id='ros', require_digest=True)
    bad = deepcopy(value)
    bad['source_generation'] = 'unknown'
    with pytest.raises(SpatialSlamError, match='source_generation'):
        validate_frame_identity(bad, clock_id='ros')


def test_perception_frame_identity_delegates_to_shared_contract():
    from plugins.pointcloud_slam_input import frame_identity
    data = frame()
    assert frame_identity(data, clock_id='ros', payload=True,
                          relation_id='calibrated_pair') == (
        'ros', 'epoch-1', 'generation-1')
    unverified = deepcopy(data)
    unverified['mapping']['verified'] = False
    # There is one normal admission path: an unverified flag is provenance,
    # not a separate strict/experimental mode. Explicit cross-source relation
    # proof remains mandatory when a relation_id is requested.
    assert frame_identity(unverified, clock_id='ros') == (
        'ros', 'epoch-1', 'generation-1')
    with pytest.raises(ValueError, match='relation'):
        frame_identity(unverified, clock_id='ros', relation_id='calibrated_pair')


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
    with pytest.raises(ValueError, match='calibration'):
        resolve_inputs({'/odom': pose, '/cloud': cloud}, bindings)
