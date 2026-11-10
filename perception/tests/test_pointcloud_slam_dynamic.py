"""Capture-time sensor transforms and explicitly related independent clocks."""

import numpy as np
import pytest

from test_pointcloud_slam_input import declarations, packet, measured_odom
from test_pointcloud_slam_stream import stream
from plugins.pointcloud_slam_input import resolve_inputs

BINDINGS = {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}


def contract(*, dynamic=False, independent=False):
    metadata = declarations()
    if dynamic:
        metadata['/c']['extrinsic'].update(mode='dynamic', source='tf')
        metadata['/c']['extrinsic'].pop('matrix')
    if independent:
        metadata['/o']['source_id'] = 'pose_device'
        for port in metadata.values():
            port['clock'] = dict(mapping='verified_relation', verified=True, relation_id='sync/1')
    return resolve_inputs(metadata, BINDINGS)


def tf_packet(ms=1000, x=0):
    value = packet(ms)
    matrix = np.eye(4)
    matrix[0, 3] = x
    value['extrinsic'] = dict(parent_frame='body', child_frame='camera', source='tf',
        version='test/1', stamp_ns=value['stamp_ns'], clock_id='clock', clock_epoch='e1',
        matrix=matrix.tolist())
    return value


def independent_packet(ms=1000, generation='cloud/1', epoch='e1', *, pose=False):
    value = packet(ms, epoch, generation)
    offset = 5_000_000 if pose else 9_000_000
    value['source_stamp'] = ms-offset/1_000_000
    value['source_clock'] = 'pose_raw' if pose else 'cloud_raw'
    value['mapping'].update(offset_ns=offset, method='pose_sync' if pose else 'cloud_sync',
                            verified=True, uncertainty_ns=42, relation_id='sync/1')
    return value


def independent_odom(ms=1000, segment='a', generation='pose/1', epoch='e1'):
    value = measured_odom(ms, segment, epoch, generation)
    value['vendor']['motus_slam']['measurement'] = independent_packet(ms, generation, epoch, pose=True)
    return value


def fuse(s, value):
    s.cloud_metadata(value, now=1.)
    s.cloud(value['stamp_ns'], 'camera', [[1, 0, 0]], now=1., data_sha256=value['data_sha256'])


def test_dynamic_declaration_has_no_static_transform():
    result = contract(dynamic=True)
    assert result['settings']['child_from_cloud'] is None
    assert result['settings']['extrinsic_mode'] == 'dynamic'
    assert result['extrinsic']['source'] == 'tf'


def test_pending_clouds_use_each_capture_transform_and_copy_evidence():
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1.)
    first, second = tf_packet(1050, 2), tf_packet(1100, 4)
    fuse(s, first)
    fuse(s, second)
    first['extrinsic']['matrix'][0][3] = 99
    first['mapping']['version'] = 'mutated'
    s.odom(measured_odom(1200), now=1.)
    frames = s.accepted
    assert [f.cloud for f in frames] == [[[1, 0, 0]], [[1, 0, 0]]]
    assert [f.child_from_cloud[0, 3] for f in frames] == [2, 4]
    assert frames[0].capture_context['child_from_cloud'][0][3] == 2
    assert frames[0].capture_context['measurement']['mapping']['version'] == '1'
    assert frames[0].capture_context['extrinsic']['stamp_ns'] == 1_050_000_000


@pytest.mark.parametrize(('field', 'bad'), [('stamp_ns', 999), ('clock_id', 'wrong'),
    ('clock_epoch', 'old'), ('parent_frame', 'wrong'), ('child_frame', 'wrong'),
    ('source', 'startup'), ('version', 'old'), ('matrix', [[1]])])
def test_invalid_dynamic_capture_never_fuses(field, bad):
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1.)
    value = tf_packet()
    value['extrinsic'][field] = bad
    fuse(s, value)
    assert not s.accepted
    assert 'extrinsic' in s.last_error


def test_missing_dynamic_capture_never_uses_identity():
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1.)
    fuse(s, packet())
    assert not s.accepted
    assert 'extrinsic' in s.last_error


def test_independent_raw_clocks_and_generations_share_only_verified_normalized_clock():
    result = contract(independent=True)
    assert result['clock_relation_mode'] == 'verified_relation'
    assert result['relation_id'] == 'sync/1'
    assert 'clock_relation_verified' not in result['settings']
    s = stream(input_contract=result)
    s.odom(independent_odom(), now=1.)
    fuse(s, independent_packet())
    assert len(s.accepted) == 1


@pytest.mark.parametrize('field', ['verified', 'relation_id', 'uncertainty_ns'])
def test_independent_frame_requires_complete_relation_proof(field):
    s = stream(input_contract=contract(independent=True))
    s.odom(independent_odom(), now=1.)
    value = independent_packet()
    value['mapping'][field] = None
    fuse(s, value)
    assert not s.accepted
    assert 'relation' in s.last_error or 'mapping' in s.last_error


@pytest.mark.parametrize('port', ['/o', '/c'])
def test_independent_declarations_require_both_verified_relation_proofs(port):
    metadata = declarations()
    metadata['/o']['source_id'] = 'other'
    for value in metadata.values():
        value['clock'] = dict(mapping='sync', verified=True, relation_id='sync/1')
    metadata[port]['clock']['verified'] = False
    with pytest.raises(ValueError, match='relation|verified'):
        resolve_inputs(metadata, BINDINGS)


def test_cloud_restart_requires_new_pose_segment_and_recovers_lower_stamps():
    s = stream(input_contract=contract(independent=True))
    s.odom(independent_odom(), now=1.)
    fuse(s, independent_packet())
    s.cloud_metadata(independent_packet(100, 'cloud/2'), now=1.)
    assert s.status(now=1.)['integration'] == 'waiting_for_new_segment'
    s.odom(independent_odom(100), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    s.odom(independent_odom(100, 'b'), now=1.)
    fuse(s, independent_packet(100, 'cloud/2'))
    assert [frame.pose.segment_id for frame in s.accepted] == ['a', 'b']
    s.cloud_metadata(independent_packet(1100), now=1.)
    assert 'retired' in s.last_error
    assert s._cloud_identity[-1] == 'cloud/2'


def test_pose_restart_keeps_independent_cloud_generation_and_rejects_old_pose():
    s = stream(input_contract=contract(independent=True))
    s.odom(independent_odom(), now=1.)
    fuse(s, independent_packet())
    s.odom(independent_odom(100, 'b', 'pose/2'), now=1.)
    fuse(s, independent_packet(100))
    assert [frame.pose.segment_id for frame in s.accepted] == ['a', 'b']
    s.odom(independent_odom(1100), now=1.)
    assert 'retired' in s.last_error
    assert s._input_identity[-1] == 'pose/2'


@pytest.mark.parametrize('kind', ['generation', 'mapping'])
def test_pose_source_change_cannot_resume_the_old_segment(kind):
    s = stream(input_contract=contract(independent=True))
    s.odom(independent_odom(), now=1.)
    fuse(s, independent_packet())
    changed = independent_odom(1100, generation='pose/2' if kind == 'generation' else 'pose/1')
    if kind == 'mapping':
        changed['vendor']['motus_slam']['measurement']['mapping']['version'] = '2'
    s.odom(changed, now=1.)
    s.odom(independent_odom(1200), now=1.)
    assert s.status(now=1.)['odometry'] != 'tracking'
    assert 'new segment' in s.last_error


def test_invalid_new_cloud_does_not_retire_current_identity():
    s = stream(input_contract=contract(dynamic=True, independent=True), child_from_cloud=None)
    s.odom(independent_odom(), now=1.)
    value = independent_packet()
    value['extrinsic'] = tf_packet()['extrinsic']
    fuse(s, value)
    changed = independent_packet(1100, 'cloud/2')
    changed['extrinsic'] = tf_packet(1100)['extrinsic']
    changed['extrinsic']['version'] = 'wrong'
    s.cloud_metadata(changed, now=1.)
    assert s._cloud_identity[-1] == 'cloud/1'
    assert not s._retired_cloud_identities
    assert s.status(now=1.)['odometry'] == 'tracking'


def test_dynamic_child_placeholder_uses_validated_pose_segment():
    value = contract(dynamic=True)
    value['settings']['child_frame'] = 'body_{segment_id}'
    value['extrinsic']['parent_frame'] = 'body_{segment_id}'
    s = stream(input_contract=value, child_from_cloud=None, child_frame='body_{segment_id}')
    sample = measured_odom()
    sample['pose']['child_frame_id'] = 'body_a'
    s.odom(sample, now=1.)
    evidence = tf_packet()
    evidence['extrinsic']['parent_frame'] = 'body_a'
    fuse(s, evidence)
    assert len(s.accepted) == 1


def test_dynamic_without_contract_cannot_fuse_without_capture_context():
    s = stream(child_from_cloud=None, extrinsic_mode='dynamic')
    s.odom(measured_odom(), now=1.)
    with pytest.raises(ValueError, match='capture context'):
        s.cloud(1_000_000_000, 'camera', [[1, 0, 0]], now=1.)


@pytest.mark.parametrize('missing', ['pose_clock', 'calibration'])
def test_strict_policy_checks_both_clock_proofs_and_dynamic_calibration(missing):
    metadata = declarations()
    metadata['/c']['extrinsic'].update(mode='dynamic', source='tf', calibrated=True)
    metadata['/c']['extrinsic'].pop('matrix')
    for value in metadata.values():
        value['clock'] = dict(mapping='shared', verified=True)
    if missing == 'pose_clock':
        metadata['/o']['clock']['verified'] = False
    else:
        metadata['/c']['extrinsic']['calibrated'] = False
    with pytest.raises(ValueError, match='verified|calibration'):
        resolve_inputs(metadata, BINDINGS)
