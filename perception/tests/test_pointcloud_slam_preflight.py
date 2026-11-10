"""Preflight must report trusted static facts without starting RTAB-Map."""
from copy import deepcopy

from plugins.pointcloud_slam_preflight import evaluate_preflight
from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin


IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]


def inputs():
    base = dict(schema='motus.slam_input/1', source_id='camera',
                stamp_source='measurement', clock_id='mapped_camera',
                reference_point='camera_optical_origin', length_unit='m')
    pose = {**base, 'topic': '/o', 'format': 'state/odom', 'kind': 'pose',
            'child_frame_id': 'base',
            'clock': {'mapping': 'shared_clock', 'verified': True}}
    cloud = {**base, 'topic': '/c', 'format': 'sensor/pointcloud2', 'kind': 'cloud',
             'frame_id': 'optical', 'observation_model': 'pinhole',
             'cloud_semantics': 'depth_frame', 'timing_model': 'instantaneous',
             'frame_metadata_topic': '/c/metadata',
             'clock': {'mapping': 'shared_clock', 'verified': True},
             'extrinsic': {'parent_frame': 'base', 'child_frame': 'optical',
                           'mode': 'static', 'matrix': deepcopy(IDENTITY),
                           'source': 'measured_mount', 'version': 'calibration-v1',
                           'calibrated': True}}
    return {'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1',
                'slam_pose': {'available': True, 'profile': 'motus_slam/1'}}},
            'input_metadata': {'/o': pose, '/c': cloud}}


def test_valid_claims_only_establish_static_readiness():
    result = evaluate_preflight(inputs())
    assert result['state'] == 'static_ready'
    assert result['diagnostics'] == []
    assert result['runtime_validation'] == 'ros_graph_and_frame_pairing_pending'
    assert result['live_frame_pairing_checked'] is False


def test_unverified_clock_is_a_note_not_a_strict_startup_gate():
    args = inputs()
    args['input_metadata']['/c']['clock']['verified'] = False
    args['input_metadata']['/c']['extrinsic']['calibrated'] = False
    result = evaluate_preflight(args)
    assert result['state'] == 'static_ready'
    assert not result['diagnostics']
    assert any('not independently verified' in note for note in result['notes'])
    assert any('not marked calibrated' in note for note in result['notes'])
    assert result['live_frame_pairing_checked'] is False


def test_missing_slam_pose_capability_and_bad_topic_type():
    args = inputs()
    args['odom_interfaces']['/o']['slam_pose']['available'] = False
    result = evaluate_preflight(args, topic_types={
        '/o': ['std_msgs/msg/String'], '/c': ['std_msgs/msg/String']})
    fields = {issue['field'] for issue in result['diagnostics']}
    assert result['state'] == 'not_ready'
    assert {'odom_interface', 'ros_type'} <= fields


def test_manual_mount_conflict_fails_preflight():
    wrong = deepcopy(IDENTITY)
    wrong[0][3] = 0.3
    result = evaluate_preflight(inputs(), manual_extrinsic=wrong)
    assert result['state'] == 'not_ready'
    assert any('extrinsic' in d['field'] for d in result['diagnostics'])


def test_preflight_action_does_not_launch_mapping_or_mutate_state():
    card = PointCloudSlamPlugin({}, executor=None)
    before = (card._node, card._pending, card._project_inputs, card._session_dir)
    response = card.dispatch('pointcloud_slam', {'action': 'preflight', **inputs()})
    after = (card._node, card._pending, card._project_inputs, card._session_dir)
    assert response['state'] == 'static_ready'
    assert before == after == (None, None, None, None)
    tool_schema = card.get_tools()[0]['inputSchema']
    assert tool_schema['x-slam-input-contract'] == 'motus.slam_input/1'
    assert 'preflight' in tool_schema['properties']['action']['enum']


def test_missing_declarations_report_waiting_not_strict_rejection():
    args = inputs()
    args['input_metadata'] = {}
    args['odom_interfaces'] = {}
    result = evaluate_preflight(args)
    assert result['state'] == 'waiting_for_inputs'
    assert not result['diagnostics']
    assert {d['field'] for d in result['pending']} == {
        'input_metadata', 'odom_interface'}
    assert result['live_frame_pairing_checked'] is False


def test_incomplete_frame_at_start_is_pending_not_fabricated():
    args = inputs()
    args['input_metadata']['/c']['frame_id'] = None
    result = evaluate_preflight(args)
    assert result['state'] == 'waiting_for_inputs'
    assert any(d['field'] == 'frame_id' for d in result['pending'])
    assert not result['diagnostics']


def test_declared_incompatible_clock_stays_not_ready():
    args = inputs()
    args['input_metadata']['/c']['clock_id'] = 'another_clock'
    result = evaluate_preflight(args)
    assert result['state'] == 'not_ready'
    assert any(d['field'] == 'clock_id' for d in result['diagnostics'])


def test_declared_mount_conflict_is_not_an_initialization_delay():
    args = inputs()
    args['input_metadata']['/o']['extrinsic'] = deepcopy(
        args['input_metadata']['/c']['extrinsic'])
    args['input_metadata']['/o']['extrinsic']['matrix'][0][3] = 0.15
    result = evaluate_preflight(args)
    assert result['state'] == 'not_ready'
    assert any(d['field'] == 'extrinsic' for d in result['diagnostics'])


def test_unknown_initializing_source_id_does_not_invent_independent_clock():
    args = inputs()
    args['input_metadata']['/o']['source_id'] = 'unknown'
    result = evaluate_preflight(args)
    assert result['state'] == 'waiting_for_inputs'
    assert not result['diagnostics']
    assert {issue['field'] for issue in result['pending']} >= {
        'source_id', 'clock.relation_id'}
