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


def test_dynamic_declaration_waits_for_numeric_transform():
    result = contract(dynamic=True)
    assert result['settings']['child_from_cloud'] is None
    s = stream(input_contract=result, child_from_cloud=None)
    s.odom(measured_odom(), now=1)
    s.cloud(1_000_000_000, 'camera', [[1,0,0]], now=2)
    assert not s.accepted
    s.cloud_metadata(tf_packet(x=2), now=1000)
    assert s.accepted[0].child_from_cloud[0,3] == 2


def test_each_capture_uses_numeric_transform_and_owned_evidence():
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1)
    first, second = tf_packet(1050,2), tf_packet(1100,4)
    fuse(s, first); fuse(s, second)
    first['extrinsic']['matrix'][0][3] = 99
    s.odom(measured_odom(1200), now=1000)
    assert [f.child_from_cloud[0,3] for f in s.accepted] == [2,4]
    assert s.accepted[0].capture_context['extrinsic']['matrix'][0][3] == 2


@pytest.mark.parametrize('field,bad', [('stamp_ns',999), ('clock_id','wrong'), ('clock_epoch','old'),
    ('parent_frame','wrong'), ('child_frame','wrong'), ('source','startup'), ('version','old')])
def test_dynamic_numeric_capture_ignores_identity_labels(field,bad):
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1)
    value = tf_packet(x=2); value['extrinsic'][field] = bad
    fuse(s,value)
    assert s.accepted[0].child_from_cloud[0,3] == 2


@pytest.mark.parametrize('matrix', [None, [[1]], [[float('nan')]*4]*4])
def test_dynamic_capture_requires_numeric_transform(matrix):
    s = stream(input_contract=contract(dynamic=True), child_from_cloud=None)
    s.odom(measured_odom(), now=1)
    value = tf_packet(); value['extrinsic']['matrix'] = matrix
    fuse(s,value)
    assert not s.accepted


@pytest.mark.parametrize('field', ['verified','relation_id','uncertainty_ns'])
def test_independent_sources_do_not_require_clock_relation_proof(field):
    s = stream(input_contract=contract(independent=True)); s.odom(independent_odom(), now=1)
    value = independent_packet(); value['mapping'][field] = None
    fuse(s,value)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('port', ['/o','/c'])
def test_independent_declarations_keep_unverified_metadata(port):
    metadata = declarations(); metadata['/o']['source_id'] = 'other'
    metadata[port]['clock'] = dict(mapping='custom', verified=False)
    result = resolve_inputs(metadata,BINDINGS)
    assert result['inputs'][port]['clock']['verified'] is False
