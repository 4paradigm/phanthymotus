"""Protocol structure, rather than producer labels, controls IMU admission."""
import numpy as np
import pytest

from plugins.rgbd_imu import ImuAttitude
from plugins.rgbd_fusion import ShortTermFusion
from plugins.rgbd_recovery import AuxiliaryHealthProbe
from test_rgbd_fusion import K, images, pose, source
from test_rgbd_imu import packet


LABELS = [
    dict(stamp_source='measurement', rotation_source='sdk_factory_extrinsics', translation_source='tf'),
    dict(stamp_source='read_completion', rotation_source='configured_fixed_mount', calibration_id=''),
    dict(stamp_source='received', rotation_source='offline_hand_eye', translation_source='survey'),
    dict(stamp_source='receipt_estimated', rotation_source='custom', clock_mapping={'any': 'description'}),
    dict(stamp_source='vendor_clock', source_id='unknown', epoch='', gyro_frame_id='raw-A',
         accel_frame_id='raw-B', camera_frame_id='producer_camera', clock_id='producer_clock',
         rotation_source='operator', translation_source='custom', units={'linear_acceleration': 'provider convention'}),
]


@pytest.mark.parametrize('labels', LABELS)
def test_all_source_and_time_labels_enter_attitude_and_full_fusion(labels):
    c=ShortTermFusion('run','optical','capture-clock',K)
    color,depth=images()
    for i in range(141):
        stamp=1_000_000_000+i*5_000_000
        c.observe_imu(source(stamp, **labels),stamp+1_000_000)
        if i%20==0:
            sample=c.observe_frame(stamp,color,depth,pose(),stamp+1_000_000)
    assert c.state=='tracking'
    np.testing.assert_allclose(sample['pose']['position'],[0.,0.,0.],atol=1e-8)
    assert c.imu.info(stamp)['stamp_source']==labels['stamp_source']


def test_optional_provenance_can_be_absent_in_both_computation_paths():
    p=source(1_000_000_000)
    for key in ('source_id','epoch','clock_id','camera_frame_id','gyro_frame_id','accel_frame_id',
                'rotation_source','translation_source','stamp_source','units'):
        p.pop(key,None)
    a=ImuAttitude('optical','capture-clock')
    assert a.observe(p,1_010_000_000)['quaternion']==[0.,0.,0.,1.]
    f=ShortTermFusion('run','optical','capture-clock',K)
    f.observe_imu(p,1_010_000_000)
    assert len(f.samples)==1


@pytest.mark.parametrize('stamp', [0,1_000_000_000,1_150_000_001,10_000_000_000,900_000_000])
def test_numeric_time_is_consumed_without_age_order_skew_or_interval_policy(stamp):
    a=ImuAttitude('optical','capture-clock')
    a.observe(packet(),1_010_000_000)
    out=a.observe(packet(stamp,accel_stamp_ns=stamp-1_000_000_000),100_000_000_000)
    assert out['stamp_ns']==stamp and a.status(200_000_000_000)=='active'


def test_changing_descriptive_metadata_does_not_reset_computation():
    f=ShortTermFusion('run','optical','capture-clock',K)
    f.observe_imu(source(1_000_000_000),1_010_000_000)
    f.observe_imu(source(1_010_000_000,**LABELS[-1],calibration_id='renamed'),1_020_000_000)
    assert f.imu.state=='active' and len(f.samples)==2
    assert f.imu.info(1_020_000_000)['source_id']=='unknown'


def test_recovery_uses_same_format_rules_without_source_or_capture_dwell():
    probe=AuxiliaryHealthProbe('optical','capture-clock',require_translation=True)
    probe.observe(source(1_000_000_000,**LABELS[-1]),100_000_000_000,10.)
    assert probe.info(200_000_000_000,20.)['status']=='ready'


@pytest.mark.parametrize('field,value', [
    ('stamp_ns',True),('accel_stamp_ns','1000'),
    ('camera_from_gyro',[[1.,0.],[0.,1.]]),
    ('angular_velocity',[1.,2.]),('linear_acceleration',[0.,float('nan'),1.]),
])
def test_malformed_numeric_fields_remain_rejected(field,value):
    a=ImuAttitude('optical','capture-clock')
    p=packet();p[field]=value
    with pytest.raises(ValueError):a.observe(p,1_010_000_000)


def test_receipt_observation_does_not_turn_off_requested_fusion():
    from plugins.rgbd_odometry_runtime import observation_calibration
    c=dict(time_basis='receipt_estimated',clock_id=None,frame_id='optical',
           imu_topic='/imu',short_term_compensation=True,depth_geometry=True,inertial_association=True)
    adapted=observation_calibration(c,'ingress')
    assert adapted['clock_id']=='ingress'
    assert adapted['imu_topic']=='/imu' and adapted['short_term_compensation'] and adapted['depth_geometry']
    from plugins.rgbd_association import association_enabled
    assert association_enabled(adapted)


def test_full_queue_retains_and_computes_the_newest_numeric_samples(monkeypatch):
    import json
    import threading
    from collections import deque
    from test_rgbd_imu import session
    obj,published,runtime,NS,_=session(monkeypatch)
    obj.imu_pending=deque();obj.imu_queue_lock=threading.Lock();obj.imu_overflow=False
    for i in range(101):
        obj._enqueue_imu(NS(data=json.dumps(packet(1_000_000_000+i*5_000_000))))
    assert len(obj.imu_pending)==100
    with obj.lock:obj._drain_imu()
    stamps=[m.header.stamp.sec*10**9+m.header.stamp.nanosec for m in published]
    assert stamps==[1_000_000_000+i*5_000_000 for i in range(1,101)]
    assert obj.imu.state=='active' and obj.gate.state!='lost'
