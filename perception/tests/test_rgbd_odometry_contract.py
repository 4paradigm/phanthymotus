from copy import deepcopy

import numpy as np
import pytest

from plugins.rgbd_odometry_contract import PairBuffer, make_sample, validate_inputs
from plugins.pointcloud_slam_odom import decode_slam_odom


def declarations():
    shared = dict(schema='motus.camera/1', id='camera-one/rectified',
                  width=640, height=480, K=[400, 0, 320, 0, 400, 240, 0, 0, 1],
                  D=[0]*5, distortion_model='pinhole', source='derived-from-K',
                  vendor=dict(frame_id='camera_optical', clock_id='camera-global',
                              registration='registered_rectified',
                              timestamp_semantics='source_capture_mapped'))
    color = dict(deepcopy(shared), topic='/rgb', format='image/jpeg')
    depth = dict(deepcopy(shared), topic='/depth', format='image/depth-zlib')
    depth['vendor'].update(depth_unit='mm', depth_source='measured')
    return {'/rgb': color, '/depth': depth}


def test_inputs_are_resolved_by_declared_format_not_topic_name():
    value = validate_inputs(declarations(), ['/depth', '/rgb'])
    assert value['color_topic'] == '/rgb'
    assert value['depth_topic'] == '/depth'
    assert value['frame_id'] == 'camera_optical'
    np.testing.assert_allclose(value['K'], [[400, 0, 320], [0, 400, 240], [0, 0, 1]])


@pytest.mark.parametrize('field,value', [
    ('id', 'other-camera'), ('K', [401,0,320,0,400,240,0,0,1]),
    ('D', [.1,0,0,0,0]), ('width', 1280), ('distortion_model', 'unknown'),
])
def test_independent_numeric_geometry_does_not_require_identity_or_labels(field, value):
    data = declarations(); data['/depth'][field] = value
    if field == 'D':
        data['/depth']['distortion_model'] = 'plumb_bob'
    assert validate_inputs(data, ['/rgb','/depth'])['depth_topic'] == '/depth'


@pytest.mark.parametrize('field,value', [
    ('registration', None), ('depth_unit', 'm'), ('depth_source', 'estimated'),
    ('timestamp_semantics', 'received'), ('clock_id', 'other-clock'),
    ('frame_id', 'other-frame'),
])
def test_depth_labels_do_not_gate_numeric_images(field, value):
    data=declarations(); data['/depth']['vendor'][field]=value
    assert validate_inputs(data, ['/rgb','/depth'])['depth_topic'] == '/depth'


def test_legacy_pairing_chooses_nearest_numeric_stamp_and_retains_future_frames():
    pairs=PairBuffer()
    assert pairs.push('color', 1_000_000_000, 'rgb', 1_010_000_000) is None
    assert pairs.push('depth', 2_000_200_000, 'z16', 1_010_000_000) is None
    assert pairs.push('color', 2_001_000_000, 'new-rgb', 2_010_000_000) == (
        (2_001_000_000, 'new-rgb'), (2_000_200_000, 'z16'))


def test_pairing_bounds_memory_without_rejecting_repeated_or_reset_stamps():
    pairs=PairBuffer()
    for i in range(20):pairs.push('color',1_000_000_000+i*1_000_000,b'x',1_100_000_000)
    assert pairs.pending_count <= 5
    for stamp,now in [(1_019_000_000,1_100_000_000),(1,2_000_000_000),
                      (2_000_000_000,1_000_000_000)]:
        assert pairs.push('color',stamp,b'x',now) is None
    with pytest.raises(ValueError):pairs.push('color', 'invalid', b'x', 0)


def pose():
    return dict(frame_id='run1_odom',child_frame_id='camera_optical',
                position=[1.,2.,3.],quaternion=[0.,0.,0.,1.],covariance=None)


def test_valid_output_is_consumable_by_existing_slam_without_fabricated_covariance():
    sample=make_sample('tracking',1_234_567_890,'run1','camera-global',pose())
    out=decode_slam_odom(sample)
    assert out.stamp_ms==1234 and out.segment_id=='run1'
    assert out.child_frame_id=='camera_optical'
    assert out.covariance is None
    assert sample['twist']==[None]*6
    assert sample['vendor']['pose_reference']=='camera_optical'


def test_lost_output_contains_no_frozen_pose():
    sample=make_sample('lost',1_234_567_890,'run1','camera-global',pose(),reason='input stale')
    assert sample['pose'] is None
    assert sample['vendor']['motus_slam']['status']=='lost'
    with pytest.raises(ValueError):decode_slam_odom(sample)


def test_tracking_output_rejects_bad_estimator_pose():
    value=pose();value['quaternion']=[0,0,0,0]
    with pytest.raises(ValueError):make_sample('tracking',1_234_567_890,'run1','camera-global',value)


@pytest.mark.parametrize('field,first,second', [
    ('source_clock', 'hardware_clock', 'global_time'),
    ('mapping_offset_ns', 1000, 2000),
    ('clock_mapping', 'fixed_arrival_offset', 'device_wall'),
    ('clock_mapping_verified', False, True),
    ('clock_mapping_uncertainty_ns', None, 0),
    ('source_epoch', 'epoch-a', 'epoch-b'),
    ('source_id', 'camera-a', 'camera-b'),
])
def test_capture_mapping_labels_do_not_gate_numeric_images(field, first, second):
    data = declarations()
    data['/rgb']['vendor'][field] = first
    data['/depth']['vendor'][field] = second
    assert validate_inputs(data, ['/rgb', '/depth'])['depth_topic'] == '/depth'


def test_shared_unverified_mapping_is_retained_without_body_extrinsic():
    data = declarations()
    for d in data.values():
        d['vendor'].update(source_clock='hardware_clock', mapping_offset_ns=123456,
                           clock_mapping='fixed_arrival_offset', clock_mapping_verified=False,
                           clock_mapping_uncertainty_ns=None, source_id='camera', source_epoch='boot')
    result = validate_inputs(data, ['/rgb', '/depth'])
    assert result['frame_id'] == 'camera_optical'
    assert 'body_mount' not in result
    assert all(d['vendor']['clock_mapping_verified'] is False for d in data.values())


def test_inherited_valid_intrinsics_and_measured_metric_depth_are_accepted():
    data=declarations()
    data['/depth']['source']='inherited'
    data['/depth']['K'][0] += 1e-7
    data['/depth']['vendor'].update(depth_source='measured', depth_semantics='optical_z')
    result=validate_inputs(data,['/rgb','/depth'])
    assert result['depth_source']=='measured'


def test_missing_capture_declaration_uses_explicit_receipt_capability():
    data=declarations()
    for d in data.values():
        d['vendor'].pop('clock_id');d['vendor'].pop('timestamp_semantics')
    assert validate_inputs(data,['/rgb','/depth'])['time_basis']=='receipt_estimated'


def test_registered_depth_projects_into_the_effective_color_intrinsics():
    data=declarations();color=data['/rgb']
    color['width'],color['height']=1280,960
    color['K']=[800,0,640,0,800,480,0,0,1]
    result=validate_inputs(data,['/rgb','/depth'])
    assert result['color_size']==(1280,960)
    np.testing.assert_allclose(result['K'], np.array(data['/rgb']['K']).reshape(3, 3))
    color['K'][2]+=20
    assert validate_inputs(data,['/rgb','/depth'])['color_size']==(1280,960)


def test_one_missing_clock_declaration_selects_receipt_without_false_capture_claim():
    data=declarations();data['/depth']['vendor'].pop('clock_id')
    assert validate_inputs(data,['/rgb','/depth'])['time_basis']=='receipt_estimated'

@pytest.mark.parametrize('source', ['estimated', None])
def test_depth_source_label_does_not_gate_numeric_depth(source):
    data = declarations()
    data['/depth']['vendor'].update(depth_source=source, depth_semantics='optical_z')
    assert validate_inputs(data, ['/rgb', '/depth'])['depth_source'] == source
