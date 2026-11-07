"""Producer facts gate live fusion, without changing legacy archive meanings."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_pointcloud_slam_stream import odom, stream
from test_pointcloud_slam_plugin import config
from test_pointcloud_slam_ros import cloud
from vision_stubs import _FakeExecutor


def declarations():
    common = dict(schema='motus.slam_input/1', source_id='camera',
                  stamp_source='measurement', clock_id='clock', length_unit='m',
                  reference_point='camera_optical_origin')
    return {'/o': {**common, 'topic': '/o', 'format': 'state/odom', 'kind': 'pose',
                   'child_frame_id': 'body'},
            '/c': {**common, 'topic': '/c', 'format': 'sensor/pointcloud2', 'kind': 'cloud',
                   'frame_id': 'camera', 'observation_model': 'pinhole',
                   'cloud_semantics': 'depth_frame', 'timing_model': 'instantaneous',
                   'frame_metadata_topic': '/c/metadata',
                   'clock': {'mapping': 'shared_camera_mapper', 'verified': False,
                             'uncertainty_ns': None},
                   'extrinsic': {'parent_frame': 'body', 'child_frame': 'camera',
                                 'matrix': np.eye(4).tolist(), 'source': 'axis_convention',
                                 'version': 'test/1', 'calibrated': False}}}


def packet(ms=1000, epoch='e1', generation='g1'):
    return dict(schema='motus.slam_frame/1', stamp_ns=ms * 1_000_000,
                frame_id='camera', clock_id='clock', clock_epoch=epoch,
                source_generation=generation, source_stamp=ms, source_unit='ms',
                source_clock='hardware_clock', mapping=dict(method='arrival_offset',
                    version='1', offset_ns=0, verified=False, uncertainty_ns=None),
                data_sha256=hashlib.sha256(cloud().data).hexdigest())


def measured_odom(ms=1000, segment='a', epoch='e1', generation='g1'):
    sample = odom(ms, segment=segment)
    sample['vendor']['motus_slam']['measurement'] = packet(ms, epoch, generation)
    return sample


def resolved():
    from plugins.pointcloud_slam_input import resolve_inputs
    return resolve_inputs(declarations(), {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
                          experimental=True)


def test_resolver_uses_producer_frames_and_preserves_unverified_evidence():
    value = resolved()
    assert value['settings']['cloud_frame'] == 'camera'
    assert value['settings']['child_frame'] == 'body'
    assert value['settings']['clock_relation_verified'] is False
    assert value['settings']['extrinsic_calibrated'] is False
    assert value['inputs']['/c']['clock']['uncertainty_ns'] is None
    assert value['frame_metadata_topic'] == '/c/metadata'


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('stamp_source', 'unknown', 'stamp_source'),
    ('length_unit', 'mm', 'length_unit'),
    ('cloud_semantics', 'accumulated_map', 'cloud_semantics'),
    ('timing_model', 'unknown', 'timing_model'),
    ('observation_model', 'unknown', 'observation_model'),
    ('extrinsic', None, 'extrinsic'),
    ('clock_id', 'other', 'clock_id'),
])
def test_resolver_rejects_unsupported_cloud_facts(field, value, reason):
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/c'][field] = value
    with pytest.raises(ValueError, match=reason):
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}, experimental=True)


def test_strict_policy_requires_real_clock_and_transform_evidence():
    from plugins.pointcloud_slam_input import resolve_inputs
    with pytest.raises(ValueError, match='verified'):
        resolve_inputs(declarations(), {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


def test_plugin_resolves_inputs_without_manual_matrix_or_frame_settings():
    from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
    cfg = {k: v for k, v in config().items() if k not in (
        'cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud',
        'clock_relation_verified', 'extrinsic_calibrated')}
    plugin = PointCloudSlamPlugin({**cfg, 'input_policy': 'experimental'}, _FakeExecutor())
    result = plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/o', '/c'], 'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
        'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
            'available': True, 'profile': 'motus_slam/1'}}}, 'input_metadata': declarations()})
    assert result['state'] == 'running', result
    info = plugin.dispatch('pointcloud_slam', {'action': 'info'})
    assert info['resolved_inputs']['/c']['frame_id'] == 'camera'
    assert info['input_validation']['extrinsic_calibrated'] is False
    assert plugin._config.get('extrinsic_calibrated') is not True
    plugin.shutdown()


def test_malformed_declared_contract_cannot_fall_back_to_legacy_settings():
    from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
    plugin = PointCloudSlamPlugin(config(), _FakeExecutor())
    result = plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/o', '/c'], 'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
        'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
            'available': True, 'profile': 'motus_slam/1'}}}, 'input_metadata': {'/c': {}}})
    assert result['state'] == 'error'
    assert result['input_diagnostics']
    assert plugin._node is None


def test_strict_stream_waits_for_exact_stamp_metadata_and_bounds_unmatched_clouds():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.,
            data_sha256=packet()['data_sha256'])
    assert not s.store.segments
    s.cloud_metadata(packet(), now=1.)
    assert len(s.store.segments[0].frames) == 1
    for ms in (1100, 1200, 1300):
        s.cloud(ms*1_000_000, 'camera', [[1, 2, 3]], now=1., data_sha256='a'*64)
    assert s.status(now=1.)['unmatched_clouds'] <= s.max_pending


def test_payload_mismatch_never_fuses_even_when_stamp_matches():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.cloud_metadata(packet(), now=1.)
    s.cloud(1_000_000_000, 'camera', [[9, 2, 3]], now=1., data_sha256='a'*64)
    assert not s.store.segments
    assert 'payload' in s.last_error


def test_new_epoch_recovers_with_lower_stamp_and_rejects_retired_packets():
    s = stream(input_contract=resolved())
    for ms, sid, epoch, generation in ((1000, 'a', 'e1', 'g1'), (100, 'b', 'e2', 'g2')):
        s.odom(measured_odom(ms, sid, epoch, generation), now=1.)
        s.cloud_metadata(packet(ms, epoch, generation), now=1.)
        s.cloud(ms*1_000_000, 'camera', [[1, 2, 3]], now=1., data_sha256=packet()['data_sha256'])
    assert [(g['segment_id'], len(g['poses'])) for g in s.graph()] == [('a', 1), ('b', 1)]
    s.odom(measured_odom(1100), now=1.)
    assert s._poses[-1].segment_id == 'b'
    s.cloud_metadata(packet(), now=1.)
    assert len(s.store.segments) == 2


def test_epoch_change_requires_new_segment_and_unknown_provenance_stops_pairing():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.odom(measured_odom(1100, epoch='e2'), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'segment' in s.last_error
    s.odom(odom(1200, segment='b'), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'


def test_ros_adapter_subscribes_sidecar_and_passes_binary_payload_digest():
    from plugins.pointcloud_slam_ros import SlamInputNode
    s = stream(input_contract=resolved())
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.)
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    assert '/c/metadata' in callbacks
    callbacks['/o'](NS(data=json.dumps(measured_odom())))
    callbacks['/c'](cloud())
    assert not s.store.segments
    callbacks['/c/metadata'](NS(data=json.dumps(packet())))
    assert len(s.store.segments[0].frames) == 1


def test_invalid_new_epoch_does_not_retire_current_valid_epoch():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    bad = measured_odom(1100, segment='b', epoch='e2', generation='g2')
    bad['pose']['child_frame_id'] = 'wrong'
    s.odom(bad, now=1.)
    assert s._input_identity == ('clock', 'e1', 'g1')
    assert ('clock', 'e1', 'g1') not in s._retired_identities


def test_metadata_waiting_status_names_missing_evidence():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1., data_sha256=packet()['data_sha256'])
    assert s.status(now=1.)['integration'] == 'waiting_for_cloud_metadata'


def test_axis_convention_transform_cannot_target_a_different_pose_reference():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/c']['extrinsic']['pose_reference'] = 'body_axes'
    metadata['/o']['pose_reference'] = 'body'
    with pytest.raises(ValueError, match='pose_reference'):
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}, experimental=True)


def test_pose_producer_can_supply_actual_sensor_mount_without_depth_assumptions():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    mount = metadata['/c'].pop('extrinsic')
    mount.update(source='calibration', calibrated=True, version='measured-mount/2')
    mount['matrix'][0][3] = .24
    metadata['/o']['extrinsic'] = mount
    result = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}, experimental=True)
    assert result['settings']['child_from_cloud'][0][3] == .24
    assert result['settings']['extrinsic_calibrated'] is True


def test_conflicting_transform_declarations_are_rejected():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/o']['extrinsic'] = deepcopy(metadata['/c']['extrinsic'])
    metadata['/o']['extrinsic']['matrix'][0][3] = .24
    with pytest.raises(ValueError, match='extrinsic'):
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}, experimental=True)


def test_lower_stamp_in_new_epoch_cannot_continue_same_segment():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.odom(measured_odom(100, epoch='e2', generation='g2'), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'segment' in s.last_error


def test_clock_mapping_change_without_new_epoch_is_a_pairing_barrier():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    changed = measured_odom(1100)
    changed['vendor']['motus_slam']['measurement']['mapping']['version'] = '2'
    s.odom(changed, now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'mapping' in s.last_error


def test_mapping_must_reproduce_mapped_acquisition_stamp():
    from plugins.pointcloud_slam_input import frame_identity
    bad = packet()
    bad['mapping']['offset_ns'] = 10
    with pytest.raises(ValueError, match='mapping'):
        frame_identity(bad, clock_id='clock', experimental=True)


def test_restore_uses_resolved_producer_configuration(monkeypatch):
    from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
    from plugins.pointcloud_slam_keyframes import KeyframeStore
    import plugins.pointcloud_slam_archive as archive
    captured = []
    def load(path, settings, **limits):
        captured.append(deepcopy(settings))
        return KeyframeStore(**limits), [], {}
    monkeypatch.setattr(archive, 'load_archive', load)
    cfg = {**config(), 'input_policy': 'experimental', 'cloud_frame': 'stale_manual_frame',
           'restore_archive': True, 'archive_name': 'check'}
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    result = plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/o', '/c'], 'input_bindings': {'/o':'state/odom','/c':'sensor/pointcloud2'},
        'odom_interfaces': {'/o': {'schema':'motus.odom/1','slam_pose': {
            'available':True,'profile':'motus_slam/1'}}}, 'input_metadata': declarations()})
    try:
        assert result['state'] == 'running', result
        assert captured[0]['cloud_frame'] == 'camera'
        assert captured[0]['extrinsic_calibrated'] is False
    finally:
        plugin.shutdown()


def test_changed_pose_mount_version_requires_input_renegotiation():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/o']['extrinsic'] = metadata['/c'].pop('extrinsic')
    metadata['/o']['pose_reference'] = 'body_axes'
    contract = resolve_inputs(metadata, {'/o':'state/odom','/c':'sensor/pointcloud2'}, experimental=True)
    s = stream(input_contract=contract)
    first = measured_odom()
    first['vendor']['motus_slam'].update(extrinsic_version='test/1', pose_reference='body_axes')
    s.odom(first, now=1.)
    assert s.status(now=1.)['odometry'] == 'tracking'
    changed = measured_odom(1100, segment='b')
    changed['vendor']['motus_slam'].update(extrinsic_version='test/2', pose_reference='body_axes')
    s.odom(changed, now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'extrinsic_version' in s.last_error


def test_runtime_pose_frame_must_match_its_declaration():
    contract = resolved()
    contract['inputs']['/o']['frame_id'] = '{segment_id}_odom'
    s = stream(input_contract=contract)
    s.odom(measured_odom(), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'pose frame' in s.last_error
