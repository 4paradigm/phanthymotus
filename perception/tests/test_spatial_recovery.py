import pytest
from plugins.spatial_recovery import RecoveryPolicy, PauseContext


def test_pause_budget_frozen_and_capped():
    p=RecoveryPolicy();a=p.begin(900_000_000,900_000_000,10_000_000_000)
    assert a.budget_ns==3_000_000_000
    assert a.deadline_monotonic_ns==13_000_000_000
    with pytest.raises(Exception):a.budget_ns=4_000_000_000
    assert RecoveryPolicy().begin(None,None,10).budget_ns==1_000_000_000


def test_external_pose_and_slam_share_pause_deadline():
    p=RecoveryPolicy().begin(500_000_000,200_000_000,1_000_000_000)
    wire=p.to_dict(anchor_stamp_ns=20_000_000_000,ingress_clock_id='process')
    restored=PauseContext.from_dict(wire)
    assert restored==p
    assert not restored.expired(p.deadline_monotonic_ns-1)
    assert restored.expired(p.deadline_monotonic_ns+1)
    wire['budget_ns']=4_000_000_000
    with pytest.raises(ValueError):PauseContext.from_dict(wire)


def test_legacy_pause_keeps_legacy_limit():
    assert RecoveryPolicy(version=1).begin(1_000_000_000,1_000_000_000,0).budget_ns==1_000_000_000


def test_versioned_slam_pause_uses_producer_deadline_without_extending_prediction():
    from test_pointcloud_slam_stream import stream, odom
    s=stream()
    s.ingress_clock_id='process'
    s.odom(odom(1000),now=1.)
    s.cloud(1_000_000_000,'camera',[[1,2,3]],now=1.)
    pause=RecoveryPolicy().begin(500_000_000,500_000_000,1_100_000_000)
    sample=odom(1200,status='recovering');sample['pose']=None
    sample['vendor']['pause']=dict(status='recovering',mapping_allowed=False)
    sample['vendor']['pause_context']=pause.to_dict(anchor_stamp_ns=1_000_000_000,ingress_clock_id='process')
    s.odom(sample,now=1.2)
    assert s._odom_state=='recovering'
    s.odom(odom(2500),now=2.5)
    assert s._odom_state=='tracking' and not s._requires_new_segment


def test_pause_budget_covers_four_intervals_and_one_processing_tail():
    pause=RecoveryPolicy().begin(500_000_000,200_000_000,10_000_000_000)
    assert pause.budget_ns==2_200_000_000
    assert pause.deadline_monotonic_ns==12_200_000_000
