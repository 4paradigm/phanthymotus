from copy import deepcopy
import pytest
from plugins.spatial_quality import MotionGate, QualityReport
from plugins.spatial_time import TimePolicy
from test_rgbd_odometry_contract import pose
from plugins.rgbd_odometry_contract import make_sample


def sample(stamp=1_000_000_000,x=0,receipt=False):
    context=TimePolicy('ingress').select({} if receipt else dict(stamp_ns=stamp,clock_id='clock',timestamp_semantics='source_capture_mapped'),stamp,stamp)
    p=pose();p['position']=[x,0.,0.]
    return make_sample('tracking',stamp,'run1',context.normalized_clock_id,p,context=context),context


def test_unknown_quality_preserves_valid_legacy_pose_and_low_sharpness_is_diagnostic():
    s,c=sample()
    for quality in (QualityReport(),QualityReport(sharpness=0.,contrast=0.)):
        assert MotionGate().evaluate(None,s,c,quality).action=='accept'


def test_tracking_pose_jump_rejected_with_declared_motion_limit():
    previous,_=sample();current,c=sample(1_100_000_000,2.)
    decision=MotionGate({'reference':'camera_optical','max_speed_mps':1.}).evaluate(previous,current,c,QualityReport())
    assert decision.action=='new_segment'


def test_camera_motion_limit_does_not_use_body_limit():
    previous,_=sample();current,c=sample(1_100_000_000,2.)
    assert MotionGate({'reference':'body','max_speed_mps':1.}).evaluate(previous,current,c,QualityReport()).action=='accept'


def test_receipt_jitter_alone_not_hard_acceleration_failure():
    previous,_=sample(receipt=True);current,c=sample(1_000_000_001,.1,True)
    assert MotionGate({'reference':'camera_optical','max_speed_mps':1.}).evaluate(previous,current,c,QualityReport()).action=='accept'
