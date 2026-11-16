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
    from plugins.spatial_pose import make_sample
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


@pytest.mark.parametrize('sidecar_first',[True,False])
@pytest.mark.parametrize('field',['source_epoch','geometry_version','mapping_version'])
def test_cloud_identity_is_checked_at_final_association_in_either_arrival_order(sidecar_first,field):
    from dataclasses import replace
    from plugins.spatial_pose import make_sample
    value,c=sample(receipt=True)
    current=replace(c,source_epoch='current',geometry_version='current',mapping_version='current')
    p=make_sample('tracking',current.observation_stamp_ns,'run1',current.normalized_clock_id,value['pose'],context=current)
    old=replace(current,**{field:'previous'})
    frame=p['pose']['child_frame_id']
    metadata={'/o':input_declaration('/o','pose',frame,current),
              '/c':input_declaration('/c','cloud',frame,current,metadata_topic='/meta')}
    for declaration in metadata.values():
        for key in ('source_epoch','geometry_version','mapping_version'):declaration.pop(key,None)
    resolved=resolve_inputs(metadata,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    s=stream(**resolved['settings'],input_contract=resolved)
    packet=frame_packet(frame,old,b'cloud')
    if sidecar_first:s.cloud_metadata(packet,now=1.)
    s.odom(p,now=1.)
    if not sidecar_first:s.cloud_metadata(packet,now=1.)
    s.cloud(current.observation_stamp_ns,frame,[[1,2,3]],now=1.,data_sha256=packet['data_sha256'])
    assert not s.accepted


def test_mixed_capture_receipt_keeps_actual_bases_and_marks_association_approximate():
    from dataclasses import replace
    from plugins.spatial_pose import make_sample
    value,receipt=sample(receipt=True)
    capture=replace(receipt,time_basis='source_capture',mapping_source='explicit_host_mapping',mapping_version='v1')
    p=make_sample('tracking',capture.observation_stamp_ns,'run1',capture.normalized_clock_id,value['pose'],context=capture)
    frame=p['pose']['child_frame_id']
    metadata={'/o':input_declaration('/o','pose',frame,capture),
              '/c':input_declaration('/c','cloud',frame,receipt,metadata_topic='/meta')}
    resolved=resolve_inputs(metadata,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    s=stream(**resolved['settings'],input_contract=resolved)
    packet=frame_packet(frame,receipt,b'cloud')
    s.odom(p,now=1.);s.cloud_metadata(packet,now=1.)
    s.cloud(receipt.observation_stamp_ns,frame,[[1,2,3]],now=1.,data_sha256=packet['data_sha256'])
    assert len(s.accepted)==1
    assert s.accepted[0].capture_context['association_method']=='arrival_time'
    assert s.accepted[0].pose.observation.time_basis=='source_capture'
    assert s.accepted[0].capture_context['observation']['time_basis']=='receipt_estimated'


@pytest.mark.parametrize('offset',[-100_000,100_000])
def test_v2_slew_pose_keeps_identity_and_admits_bracketed_camera_cloud(offset):
    from dataclasses import replace
    from test_pointcloud_slam_input import slew_odom
    from plugins.spatial_time import TimePolicy
    def pose(ms,delta):
        value=slew_odom(ms,delta)
        m=value['vendor']['motus_slam']['measurement']
        ctx=TimePolicy('ingress').select(dict(stamp_ns=m['stamp_ns'],clock_id='clock',
            timestamp_semantics='source_capture_mapped',source_id='chassis',source_epoch='e1',
            source_stamp=ms,source_unit='ms',source_clock='hardware_clock',
            clock_mapping='chassis_clock_request_interval_slew',mapping_version='2',
            mapping_offset_ns=delta,clock_mapping_verified=True,
            clock_mapping_uncertainty_ns=50_000_000,geometry_version='body1'),m['stamp_ns'],m['stamp_ns'])
        value['vendor']['motus_slam'].update(profile='motus_slam/2',stamp_ns=m['stamp_ns'],observation=ctx.to_dict())
        return value,ctx
    first,ctx=pose(1000,0)
    camera=replace(ctx,source_id='camera',source_epoch='camera1',mapping_source='camera_global',
                   mapping_version='1',geometry_version='camera1',observation_stamp_ns=1_050_000_000)
    declarations={'/o':input_declaration('/o','pose','body',ctx),
                  '/c':input_declaration('/c','cloud','body',camera,metadata_topic='/c/meta')}
    contract=resolve_inputs(declarations,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    s=stream(**contract['settings'],input_contract=contract);s.ingress_clock_id='ingress'
    s.odom(first,now=1.)
    packet=frame_packet('body',camera,b'raw cloud')
    s.cloud_metadata(packet,now=1.05)
    s.cloud(camera.observation_stamp_ns,'body',[[1,2,3]],now=1.05,data_sha256=packet['data_sha256'])
    assert not s.accepted
    second,_=pose(1100,offset);s.odom(second,now=1.1)
    assert len(s.accepted)==1
    assert s.accepted[0].pose.segment_id=='a'
    assert s.accepted[0].cloud==[[1,2,3]]
    assert s.accepted[0].capture_context['observation']['source_epoch']=='camera1'
    assert s.status(now=1.1)['odometry']=='tracking'
