import hashlib
import numpy as np
import pytest
from test_spatial_quality import sample
from test_pointcloud_slam_stream import stream
from plugins.spatial_contract import input_declaration, frame_packet
from plugins.pointcloud_slam_input import resolve_inputs


def test_receipt_basis_reaches_slam_with_versioned_inputs_and_exact_sidecar():
    pose,ctx=sample(receipt=True)
    frame=pose['pose']['child_frame_id']
    declarations={
        '/o':input_declaration('/o','pose',frame,ctx),
        '/c':input_declaration('/c','cloud',frame,ctx,metadata_topic='/c/meta')}
    resolved=resolve_inputs(declarations,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    s=stream(**resolved['settings'],input_contract=resolved)
    s.ingress_clock_id=ctx.ingress_clock_id
    s.odom(pose,now=1.)
    packet=frame_packet(frame,ctx,b'cloud')
    s.cloud_metadata(packet,now=1.)
    s.cloud(ctx.observation_stamp_ns,frame,[[1,2,3]],now=1.,data_sha256=packet['data_sha256'])
    assert len(s.accepted)==1
    assert s.accepted[0].capture_context['observation']['time_basis']=='receipt_estimated'


def test_v2_receipt_clock_conflict_cannot_be_hidden_by_metadata_fallback():
    _,ctx=sample(receipt=True)
    declarations={'/o':input_declaration('/o','pose','camera',ctx),
                  '/c':input_declaration('/c','cloud','camera',ctx,metadata_topic='/c/meta')}
    declarations['/c']['clock_id']='foreign'
    with pytest.raises(ValueError):resolve_inputs(declarations,{'/o':'state/odom','/c':'sensor/pointcloud2'})


def test_v2_missing_or_corrupt_sidecar_never_falls_back():
    pose,ctx=sample(receipt=True);frame=pose['pose']['child_frame_id']
    declarations={'/o':input_declaration('/o','pose',frame,ctx),
                  '/c':input_declaration('/c','cloud',frame,ctx,metadata_topic='/c/meta')}
    resolved=resolve_inputs(declarations,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    s=stream(**resolved['settings'],input_contract=resolved)
    s.odom(pose,now=1.)
    packet=frame_packet(frame,ctx,b'correct')
    s.cloud(ctx.observation_stamp_ns,frame,[[1,2,3]],now=1.,data_sha256='0'*64)
    s.cloud_metadata(packet,now=1.01)
    s._release_without_metadata(1.3,None)
    assert not s.accepted
    assert 'payload' in s.last_error


def test_nanosecond_pose_alignment_preserves_v2_observation_stamp():
    from plugins.pointcloud_slam_alignment import align_pose
    from plugins.pointcloud_slam_odom import decode_slam_odom
    p,c=sample(1_000_000_123,receipt=True)
    aligned=align_pose([decode_slam_odom(p)],c.observation_stamp_ns,max_gap_ms=200)
    assert aligned.observation==c


@pytest.mark.parametrize('stamp',[True,'1000000000',float('nan'),1_001_000_000])
def test_legacy_pose_precise_stamp_cannot_conflict_with_milliseconds(stamp):
    from test_pointcloud_slam_stream import odom
    from plugins.pointcloud_slam_odom import decode_slam_odom
    value=odom(1000);value['vendor']['motus_slam']['stamp_ns']=stamp
    with pytest.raises(ValueError):decode_slam_odom(value)


def test_receipt_time_makes_imu_unavailable_without_guessing_sample_intervals():
    from plugins.rgbd_odometry_runtime import observation_calibration, estimator_parameters
    original=dict(time_basis='receipt_estimated',clock_id=None,frame_id='optical',imu_topic='/imu',short_term_compensation=True)
    adapted=observation_calibration(original,'ingress')
    assert adapted['clock_id']=='ingress'
    assert 'imu_topic' not in adapted and not adapted['short_term_compensation']
    assert 'wait_imu_to_init' not in estimator_parameters(adapted,'run')
    assert original['imu_topic']=='/imu'


def test_missing_pose_capture_uses_receipt_but_known_measurement_is_preserved():
    from plugins.spatial_contract import normalize_pose_time
    from test_pointcloud_slam_stream import odom
    original=odom(1000)
    _,ctx=sample(1_100_000_000,receipt=True)
    assert normalize_pose_time(original,ctx)==original
    original['vendor']['motus_slam']['stamp_source']='unknown'
    normalized=normalize_pose_time(original,ctx)
    assert normalized['stamp_ms']==1100
    assert normalized['vendor']['motus_slam']['stamp_source']=='receipt_estimated'
    assert original['stamp_ms']==1000


def test_context_change_requires_new_segment_and_new_segment_can_map():
    from dataclasses import replace
    from plugins.rgbd_odometry_contract import make_sample
    first,context=sample(receipt=True)
    s=stream(cloud_frame=first['pose']['child_frame_id'],child_frame=first['pose']['child_frame_id'],clock_id=context.normalized_clock_id)
    s.odom(first,now=1.)
    s.cloud(context.observation_stamp_ns,s.cloud_frame,[[1,2,3]],now=1.)
    later=replace(context,source_epoch='changed',observation_stamp_ns=1_100_000_000,ingress_realtime_ns=1_100_000_000)
    changed=make_sample('tracking',later.observation_stamp_ns,'run1',later.normalized_clock_id,first['pose'],context=later)
    s.odom(changed,now=1.1)
    assert s.status(now=1.1)['integration']=='waiting_for_new_segment'
    changed['vendor']['motus_slam']['segment_id']='run2'
    s.odom(changed,now=1.1)
    s.cloud(later.observation_stamp_ns,s.cloud_frame,[[1,2,3]],now=1.1)
    assert s.status(now=1.1)['integration']=='admitting'
    assert len(s.accepted)==2
