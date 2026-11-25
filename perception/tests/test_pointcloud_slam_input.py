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


def test_resolver_preserves_producer_metadata_without_certifying_it():
    value = resolved()
    assert value['settings']['cloud_frame'] == 'camera'
    assert value['inputs']['/c']['clock']['verified'] is False
    assert value['extrinsic']['calibrated'] is False
    assert value['settings']['child_from_cloud'] == np.eye(4).tolist()


@pytest.mark.parametrize('field,value', [('stamp_source', 'unknown'), ('length_unit', 'mm'),
    ('cloud_semantics', 'accumulated_map'), ('timing_model', 'unknown'),
    ('observation_model', 'unknown'), ('clock_id', 'other'), ('source_id', 'different'),
    ('reference_point', 'unknown'), ('extrinsic', None)])
def test_resolver_accepts_unproven_descriptive_metadata(field, value):
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations(); metadata['/c'][field] = value
    result = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert result['settings']['child_from_cloud'] == np.eye(4).tolist()


@pytest.mark.parametrize('field,value', [('format', 'wrong'), ('schema', 'wrong')])
def test_resolver_checks_wire_routing(field, value):
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations(); metadata['/c'][field] = value
    with pytest.raises(ValueError): resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


@pytest.mark.parametrize('matrix', [[[1]], [[float('nan')]*4]*4])
def test_resolver_checks_numeric_mount_geometry(matrix):
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations(); metadata['/c']['extrinsic']['matrix'] = matrix
    with pytest.raises(ValueError): resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})


@pytest.mark.parametrize('evidence', [None, packet(), dict(packet(), data_sha256='wrong', clock_id='other')])
def test_static_mapping_does_not_wait_for_sidecars_or_digest(evidence):
    s = stream(input_contract=resolved()); s.odom(measured_odom(), now=1)
    if evidence is not None: s.cloud_metadata(evidence, now=2)
    s.cloud(1_000_000_000, 'other-frame', [[1,2,3]], now=1000)
    assert len(s.accepted) == 1


@pytest.mark.parametrize('field,value', [('clock_id', 'other'), ('segment_id', 'other'),
    ('stamp_source', 'unknown'), ('status', 'lost'), ('measurement', None),
    ('extrinsic_version', 'other'), ('pose_reference', 'other')])
def test_metadata_changes_never_retire_later_numeric_pose(field, value):
    s = stream(input_contract=resolved()); s.odom(measured_odom(), now=1)
    changed = measured_odom(1100); changed['vendor']['motus_slam'][field] = value
    s.odom(changed, now=2)
    s.cloud(1_100_000_000, 'camera', [[1,2,3]], now=3)
    s.odom(measured_odom(1000), now=4)
    s.cloud(1_000_000_000, 'camera', [[1,2,3]], now=5)
    assert len(s.accepted) == 2


def test_explicit_pose_numeric_mount_is_applied_despite_different_names():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata = declarations()
    mount = metadata['/c'].pop('extrinsic'); mount.update(parent_frame='other', child_frame='elsewhere')
    mount['matrix'][0][3] = .4
    metadata['/o']['extrinsic'] = mount
    result = resolve_inputs(metadata, {'/o': 'state/odom', '/c': 'sensor/pointcloud2'})
    assert result['settings']['child_from_cloud'][0][3] == .4


def slew_odom(ms=1000, offset=0):
    value=measured_odom(ms)
    measurement=value['vendor']['motus_slam']['measurement']
    measurement['mapping'].update(method='chassis_clock_request_interval_slew',version='2',
                                  offset_ns=offset,verified=True,uncertainty_ns=50_000_000)
    measurement['stamp_ns']+=offset
    value['stamp_ms']=measurement['stamp_ns']//1_000_000
    return value
