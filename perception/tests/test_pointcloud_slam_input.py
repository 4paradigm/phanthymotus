"""Producer declarations and capture evidence gate native mapping admission."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from test_pointcloud_slam_stream import odom, stream
from test_pointcloud_slam_ros import cloud


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
                          )


def test_resolver_uses_producer_frames_and_preserves_unverified_evidence():
    value = resolved()
    assert value['settings']['cloud_frame'] == 'camera'
    assert value['settings']['child_frame'] == 'body'
    # Operator-assertion flags are gone; the contract carries producer evidence only.
    assert 'clock_relation_verified' not in value['settings']
    assert 'extrinsic_calibrated' not in value['settings']
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
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


def test_default_resolution_uses_unverified_clock_and_keeps_provenance():
    from plugins.pointcloud_slam_input import resolve_inputs
    result = resolve_inputs(declarations(), {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert result['clock_relation_mode'] == 'shared_source'
    assert result['inputs']['/c']['clock']['verified'] is False
    assert result['extrinsic']['calibrated'] is False
    assert result['settings']['child_from_cloud'] == np.eye(4).tolist()


def test_strict_stream_waits_for_exact_stamp_metadata_and_bounds_unmatched_clouds():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.,
            data_sha256=packet()['data_sha256'])
    assert not s.accepted
    s.cloud_metadata(packet(), now=1.)
    assert len(s.accepted) == 1
    for ms in (1100, 1200, 1300):
        s.cloud(ms*1_000_000, 'camera', [[1, 2, 3]], now=1., data_sha256='a'*64)
    assert s.status(now=1.)['unmatched_clouds'] <= s.max_pending


def test_payload_mismatch_never_fuses_even_when_stamp_matches():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.cloud_metadata(packet(), now=1.)
    s.cloud(1_000_000_000, 'camera', [[9, 2, 3]], now=1., data_sha256='a'*64)
    assert not s.accepted
    assert 'payload' in s.last_error


def test_new_epoch_recovers_with_lower_stamp_and_rejects_retired_packets():
    s = stream(input_contract=resolved())
    for ms, sid, epoch, generation in ((1000, 'a', 'e1', 'g1'), (100, 'b', 'e2', 'g2')):
        s.odom(measured_odom(ms, sid, epoch, generation), now=1.)
        s.cloud_metadata(packet(ms, epoch, generation), now=1.)
        s.cloud(ms*1_000_000, 'camera', [[1, 2, 3]], now=1., data_sha256=packet()['data_sha256'])
    assert [frame.pose.segment_id for frame in s.accepted] == ['a', 'b']
    s.odom(measured_odom(1100), now=1.)
    assert s._poses[-1].segment_id == 'b'
    s.cloud_metadata(packet(), now=1.)
    assert len(s.accepted) == 2


def test_epoch_change_requires_new_segment_and_unknown_provenance_stops_pairing():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.odom(measured_odom(1100, epoch='e2'), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'segment' in s.last_error
    s.odom(odom(1200, segment='b'), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'


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
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


def test_pose_producer_can_supply_actual_sensor_mount_without_depth_assumptions():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    mount = metadata['/c'].pop('extrinsic')
    mount.update(source='calibration', calibrated=True, version='measured-mount/2')
    mount['matrix'][0][3] = .24
    metadata['/o']['extrinsic'] = mount
    result = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert result['settings']['child_from_cloud'][0][3] == .24
    # The evidence stays on the declaration; it is not repackaged as a card flag.
    assert 'extrinsic_calibrated' not in result['settings']
    assert result['extrinsic']['calibrated'] is True


def test_conflicting_transform_declarations_are_rejected():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/o']['extrinsic'] = deepcopy(metadata['/c']['extrinsic'])
    metadata['/o']['extrinsic']['matrix'][0][3] = .24
    with pytest.raises(ValueError, match='extrinsic'):
        resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


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
        frame_identity(bad, clock_id='clock')


def test_changed_pose_mount_version_requires_input_renegotiation():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/o']['extrinsic'] = metadata['/c'].pop('extrinsic')
    metadata['/o']['pose_reference'] = 'body_axes'
    contract = resolve_inputs(metadata, {'/o':'state/odom','/c':'sensor/pointcloud2'})
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


def test_no_per_pose_measurement_needed_for_same_source_tracking():
    s = stream(input_contract=resolved())
    sample = odom(1000)
    s.odom(sample, now=1.)
    assert s.status(now=1.)['odometry'] == 'tracking'
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.,
            data_sha256=packet()['data_sha256'])
    # The source sends a sidecar but the odometry doesn't need to invent one.
    s.cloud_metadata(packet(), now=1.05)
    assert [f.pose.segment_id for f in s.accepted] == ['a']


def test_missing_sidecar_falls_back_to_bounded_odom_pairing():
    s = stream(input_contract=resolved())
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.01,
            data_sha256=packet()['data_sha256'])
    assert not s.accepted
    # An incoming tracking pose is sufficient to drain the sidecar grace.
    s.odom(odom(1200, x=0.2), now=1.30)
    assert len(s.accepted) == 1
    assert s.status(now=1.30)['frames_without_sidecar'] == 1


def test_known_payload_digest_mismatch_never_uses_fallback():
    s = stream(input_contract=resolved())
    s.odom(odom(1000), now=1.)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.01,
            data_sha256='f' * 64)
    s.cloud_metadata(packet(), now=1.02)
    s.odom(odom(1200), now=1.30)
    assert not s.accepted
    assert s.status(now=1.30)['frames_without_sidecar'] == 0
    assert 'payload' in s.last_error


def test_invalid_tracking_sample_blocks_cloud_without_evidence():
    s = stream(input_contract=resolved())
    s.odom(odom(1000), now=1.)
    lost = odom(1100, status='lost')
    s.odom(lost, now=1.1)
    assert s.status(now=1.1)['odometry'] != 'tracking'
    with pytest.raises(ValueError, match='odometry'):
        s.cloud(1_150_000_000, 'camera', [[1, 2, 3]], now=1.15,
                data_sha256=packet()['data_sha256'])
    assert not s.accepted


def test_missing_evidence_after_present_evidence_is_not_downgraded():
    s = stream(input_contract=resolved())
    s.odom(measured_odom(), now=1.)
    s.odom(odom(1100), now=1.1)
    assert s.status(now=1.1)['odometry'] != 'tracking'
    assert 'evidence lost' in s.last_error
