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
    return resolve_inputs(declarations(), {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


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


def test_sidecar_pairing_waits_for_exact_digest_and_bounds_unmatched_clouds():
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


def test_new_bumi_segment_reuses_measured_mount_with_new_frame_names():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    metadata['/o'].update(segment_id='s', frame_id='s_odom',
                          child_frame_id='s_base')
    metadata['/c']['extrinsic']['parent_frame'] = 's_base'
    contract = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert contract['settings']['child_frame'] == '{segment_id}_base'
    assert contract['inputs']['/o']['frame_id_template'] == '{segment_id}_odom'
    assert contract['inputs']['/o']['frame_id'] == 's_odom'
    s = stream(input_contract=contract, child_frame='{segment_id}_base')
    for stamp_ms, segment in [(1000, 's'), (1100, 't')]:
        sample = odom(stamp_ms, segment=segment)
        sample['pose']['frame_id'] = segment + '_odom'
        sample['pose']['child_frame_id'] = segment + '_base'
        s.odom(sample, now=stamp_ms / 1000)
        s.cloud(stamp_ms * 1_000_000, 'camera', [[1, 2, 3]],
                now=stamp_ms / 1000, data_sha256=packet()['data_sha256'])
        s.cloud_metadata(packet(stamp_ms), now=stamp_ms / 1000)
    assert [frame.pose.segment_id for frame in s.accepted] == ['s', 't']
    assert s.status(now=1.1)['odometry'] == 'tracking'


def slew_odom(ms=1000, offset=0):
    value=measured_odom(ms)
    measurement=value['vendor']['motus_slam']['measurement']
    measurement['mapping'].update(method='chassis_clock_request_interval_slew',version='2',
                                  offset_ns=offset,verified=True,uncertainty_ns=50_000_000)
    measurement['stamp_ns']+=offset
    value['stamp_ms']=measurement['stamp_ns']//1_000_000
    return value


@pytest.mark.parametrize('offset',[-1_000_000,1_000_000])
def test_bounded_chassis_slew_does_not_retire_explicit_pose_segment(offset):
    s=stream(input_contract=resolved())
    s.odom(slew_odom(),now=1.)
    s.odom(slew_odom(1100,offset),now=1.1)
    assert s.status(now=1.1)['odometry']=='tracking'
    assert not s.last_error


@pytest.mark.parametrize('change',[
    dict(offset_ns=1_000_001),dict(version='3'),dict(method='other'),
    dict(verified=False),dict(uncertainty_ns=50_000_001),dict(uncertainty_ns=None),
])
def test_slew_does_not_admit_unbounded_or_changed_evidence(change):
    s=stream(input_contract=resolved());s.odom(slew_odom(),now=1.)
    value=slew_odom(1100,100_000);m=value['vendor']['motus_slam']['measurement']
    m['mapping'].update(change)
    m['stamp_ns']=m['source_stamp']*1_000_000+m['mapping']['offset_ns']
    value['stamp_ms']=m['stamp_ns']//1_000_000
    s.odom(value,now=1.1)
    assert s.status(now=1.1)['odometry']!='tracking'


def test_slew_preserves_source_epoch_and_unit_barriers():
    for field,value in [('clock_epoch','new'),('source_clock','other'),('source_unit','ns')]:
        s=stream(input_contract=resolved());s.odom(slew_odom(),now=1.)
        changed=slew_odom(1100,100_000);m=changed['vendor']['motus_slam']['measurement']
        m[field]=value
        if field=='source_unit':m['source_stamp']*=1_000_000
        s.odom(changed,now=1.1)
        assert s.status(now=1.1)['odometry']!='tracking'


def test_late_slew_sample_is_dropped_without_retiring_segment():
    s=stream(input_contract=resolved())
    s.odom(slew_odom(1000,0),now=1.)
    s.odom(slew_odom(1100,100_000),now=1.1)
    s.odom(slew_odom(1050,50_000),now=1.11)
    assert s.last_error=='late odometry sample dropped'
    assert s.status(now=1.11)['odometry']=='tracking'
    s.odom(slew_odom(1200,200_000),now=1.2)
    assert s.status(now=1.2)['odometry']=='tracking'
    assert s._pose_measurement['source_stamp']==1200


def test_same_source_stamp_cannot_be_rewritten_by_slew():
    s=stream(input_contract=resolved())
    s.odom(slew_odom(),now=1.)
    s.odom(slew_odom(1000,1),now=1.01)
    assert s.status(now=1.01)['odometry']!='tracking'


@pytest.mark.parametrize('explicit', [False, True])
def test_twist_receive_time_does_not_change_pose_cloud_association(explicit):
    s = stream(input_contract=resolved() if explicit else None)
    value = measured_odom(1000)
    value['pose']['stamp_ms'] = 1000
    value['stamp_ms'] = 1200
    s.odom(value, now=1.2)
    s.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1.2,
            data_sha256=packet()['data_sha256'])
    if explicit:
        s.cloud_metadata(packet(), now=1.2)
    assert len(s.accepted) == 1
    assert s.accepted[0].pose.stamp_ms == 1000


def test_fresh_twist_cannot_refresh_stale_diagnostic_pose():
    s = stream(max_measurement_age_ms=100, max_future_skew_ms=50)
    value = odom(1200)
    value['pose']['stamp_ms'] = 1000
    s.odom(value, now=1.2, measurement_now_ns=1_200_000_000)
    assert s.status(now=1.2, measurement_now_ns=1_200_000_000)['odometry'] != 'tracking'
    assert 'stale' in s.last_error


def test_explicit_pose_time_cannot_override_diagnostic_source_evidence():
    s = stream(input_contract=resolved())
    value = measured_odom(1000)
    value['pose']['stamp_ms'] = 1100
    s.odom(value, now=1.1)
    assert s.status(now=1.1)['odometry'] != 'tracking'
    assert 'measurement timestamp' in s.last_error


def test_rest_feedback_with_lost_slam_quality_cannot_map():
    s = stream()
    s.odom(odom(1000), now=1.)
    value = odom(1100, status='lost')
    value['twist'] = [0.3, 0, None, None, None, -0.4]
    value['pose'] = None
    s.odom(value, now=1.1)
    with pytest.raises(ValueError, match='odometry stale or unavailable'):
        s.cloud(1_100_000_000, 'camera', [[1, 2, 3]], now=1.1)
    assert not s.accepted
    assert s.status(now=1.1)['odometry'] != 'tracking'


@pytest.mark.parametrize('explicit', [False, True])
def test_rest_republication_does_not_refresh_diagnostic_pose_ingress(explicit):
    s = stream(input_contract=resolved() if explicit else None)
    value = measured_odom(1000)
    value['pose']['stamp_ms'] = 1000
    s.odom(value, now=1.)
    for now, rest_stamp in [(1.1, 1100), (1.9, 1900)]:
        value['stamp_ms'] = rest_stamp
        value['twist'] = [0.3, 0, None, None, None, -0.4]
        s.odom(value, now=now)
        assert s.status(now=now)['odometry'] == 'tracking'
        assert not s.last_error
    assert s.status(now=2.01)['odometry'] == 'stale'
    assert s.status(now=2.01)['last_error'] == 'odometry receipt stale'
