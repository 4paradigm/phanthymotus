"""Observation time never upgrades receipt proximity to capture identity."""
from dataclasses import FrozenInstanceError
import json
import pytest
from plugins.spatial_time import FrameContext, TimePolicy, compatible_sources
from plugins.rgbd_odometry_contract import make_sample
from plugins.pointcloud_slam_odom import decode_slam_odom
from plugins.pointcloud_slam_contract import validate_tracking_pose

NOW = 1_794_668_000_123_456_789

def source(**changes):
    value = dict(source_id='camera', source_epoch='boot', stamp_ns=NOW-12_345,
                 clock_id='capture-domain', timestamp_semantics='source_capture_mapped',
                 clock_mapping_verified=False, mapping_offset_ns=100)
    value.update(changes)
    return value


def test_capture_mapping_preserves_integer_stamp():
    c=TimePolicy('local-ingress').select(source(), NOW, 123456)
    assert c.observation_stamp_ns == NOW-12_345
    assert c.original_stamp == NOW-12_345 and c.original_unit == 'ns'
    assert c.time_basis == 'source_capture' and not c.mapping_verified
    assert c.source_capture_id is None
    assert FrameContext.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    with pytest.raises(FrozenInstanceError):c.time_basis='receipt_estimated'


def test_missing_capture_uses_receipt_without_measurement_claim():
    c=TimePolicy('local-ingress').select(source(stamp_ns=None), NOW, 123456)
    assert c.observation_stamp_ns == NOW and c.time_basis == 'receipt_estimated'
    assert c.normalized_clock_id == 'local-ingress'
    assert c.source_capture_id is None and c.uncertainty_ns is None
    assert c.ingress_monotonic_ns == 123456


@pytest.mark.parametrize('field,a,b', [
    ('source_clock','a','b'),('mapping_offset_ns',1,2),('clock_mapping','a','b'),
    ('clock_mapping_verified',True,False),('clock_mapping_uncertainty_ns',1,2),
    ('source_id','a','b'),('source_epoch','a','b'),
])
def test_descriptive_clock_fields_do_not_reject_input(field,a,b):
    compatible_sources(source(stamp_ns=0,**{field:a}),source(stamp_ns=0,**{field:b}))


@pytest.mark.parametrize('stamp', [True,float('nan'),1.5])
def test_malformed_known_capture_time_is_not_rewritten(stamp):
    with pytest.raises(ValueError):TimePolicy('local').select(source(stamp_ns=stamp),NOW,1)


def test_unknown_clock_does_not_replace_supplied_numeric_time():
    c=TimePolicy('local').select(source(stamp_ns=55,clock_id=None,timestamp_semantics=None),NOW,1)
    assert c.original_stamp==55 and c.observation_stamp_ns==55 and c.time_basis=='source_capture'


def test_v1_capture_contract_still_reads_and_v2_receipt_is_explicit():
    pose=dict(frame_id='run_odom',child_frame_id='camera',position=[0.,0.,0.],
              quaternion=[0.,0.,0.,1.],covariance=None)
    old=make_sample('tracking',NOW,'run','old-clock',pose)
    assert decode_slam_odom(old).clock_id == 'old-clock'
    c=TimePolicy('local').select(source(stamp_ns=None),NOW,123)
    new=make_sample('tracking',NOW,'run','local',pose,context=c)
    q=new['vendor']['motus_slam']
    assert q['profile']=='motus_slam/2' and q['stamp_source']=='receipt_estimated'
    assert decode_slam_odom(new).observation == c
    validate_tracking_pose(new)
    assert q['stamp_source']=='receipt_estimated'


def test_observation_metadata_is_recorded_without_consistency_policy():
    c=TimePolicy('local').select(source(stamp_ns=None),NOW,1)
    d=c.to_dict();d['mapping_verified']=True
    assert FrameContext.from_dict(d).observation_stamp_ns==d['observation_stamp_ns']
    d=c.to_dict();d['observation_stamp_ns']+=1
    assert FrameContext.from_dict(d).observation_stamp_ns==d['observation_stamp_ns']


def test_quality_unknown_is_not_an_implicit_rejection():
    from plugins.spatial_quality import QualityReport, QualityDecision
    q=QualityReport()
    assert q.depth_fraction is None and q.inliers is None
    assert json.loads(json.dumps(q.to_dict()))['depth_fraction'] is None
    assert QualityDecision('accept','optional quality unknown').action == 'accept'
    with pytest.raises(ValueError):QualityDecision('guess','not a decision')
