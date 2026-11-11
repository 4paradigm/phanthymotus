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
def test_equal_frame_names_do_not_hide_invalid_geometry(field, value):
    data = declarations(); data['/depth'][field] = value
    with pytest.raises(ValueError):
        validate_inputs(data, ['/rgb','/depth'])


@pytest.mark.parametrize('field,value', [
    ('registration', None), ('depth_unit', 'm'), ('depth_source', 'estimated'),
    ('timestamp_semantics', 'received'), ('clock_id', 'other-clock'),
    ('frame_id', 'other-frame'),
])
def test_inputs_require_measured_registered_depth_and_capture_time(field, value):
    data=declarations(); data['/depth']['vendor'][field]=value
    with pytest.raises(ValueError):validate_inputs(data, ['/rgb','/depth'])


def test_pairing_preserves_capture_stamps_and_discards_large_skew():
    pairs=PairBuffer()
    assert pairs.push('color', 1_000_000_000, 'rgb', 1_010_000_000) is None
    pair=pairs.push('depth', 1_000_200_000, 'z16', 1_010_000_000)
    assert pair == ((1_000_000_000, 'rgb'), (1_000_200_000, 'z16'))
    pairs.push('color',1_100_000_000,'old',1_110_000_000)
    assert pairs.push('depth',1_140_000_000,'next',1_150_000_000) is None
    pair=pairs.push('color',1_140_200_000,'new',1_150_000_000)
    assert pair[0][1]=='new'


def test_pairing_bounds_memory_and_rejects_stale_repeated_reset_stamps():
    pairs=PairBuffer()
    for i in range(20):pairs.push('color',1_000_000_000+i*1_000_000,b'x',1_100_000_000)
    assert pairs.pending_count <= 5
    for stamp,now in [(1_019_000_000,1_100_000_000),(1,2_000_000_000),
                      (2_000_000_000,1_000_000_000)]:
        with pytest.raises(ValueError):pairs.push('color',stamp,b'x',now)


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
