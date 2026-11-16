"""Shared spatial input admits numeric data independently of provenance policy."""
import numpy as np
import pytest

from plugins.spatial_time import FrameContext, TimePolicy, compatible_sources
from plugins.spatial_input import InputAdapterRegistry
from plugins.rgbd_body_mount import CaptureMounts
from test_spatial_input import Backend, message, declaration, wait_for
from test_rgbd_body_mount import declaration as mount_declaration, packet, sample


@pytest.mark.parametrize('stamp',[-1,0,10,1_000_000_000,100_000_000_000])
def test_every_integer_stamp_is_used_without_source_label_or_wall_age_policy(stamp):
    c=TimePolicy('consumer').select(dict(stamp_ns=stamp,clock_id='producer',
        timestamp_semantics='custom',clock_mapping_verified=False),10_000_000_000,100)
    assert c.observation_stamp_ns==stamp
    assert c.original_stamp==stamp


def test_source_declarations_and_observation_metadata_are_not_consistency_gates():
    compatible_sources({'source_epoch':'a','clock_id':'a','mapping_offset_ns':10},
                       {'source_epoch':'b','clock_id':'b','mapping_offset_ns':20})
    c=TimePolicy('consumer').select(dict(stamp_ns=10,clock_id='source'),100,100)
    wire=c.to_dict()
    wire.update(time_basis='receipt_estimated',normalized_clock_id='other',mapping_verified=True,
                uncertainty_ns=50,original_unit='custom',vendor_detail={'arbitrary':True})
    restored=FrameContext.from_dict(wire)
    assert restored.observation_stamp_ns==10 and restored.mapping_verified is True


def test_observation_numeric_time_does_not_require_diagnostic_names():
    context=FrameContext.from_dict({'schema':'motus.observation/1','observation_stamp_ns':10})
    assert context.observation_stamp_ns==10


def test_subscribers_with_different_descriptive_geometry_share_live_input():
    backend=Backend();registry=InputAdapterRegistry(subscribe=backend.subscribe)
    got=[]
    first=registry.acquire('/data',object,declaration(),lambda m,c:got.append(c))
    second_declaration=declaration('other')
    second_declaration.update(id='other',width=10,height=20,K=[2.]*9)
    second=registry.acquire('/data',object,second_declaration,lambda m,c:got.append(c))
    try:
        backend.callbacks[0](message(1))
        wait_for(lambda:len(got)==2)
        assert len(backend.callbacks)==1
    finally:first.close();second.close();registry.close()


def test_consumer_receives_old_numeric_context_without_age_drop():
    backend=Backend();registry=InputAdapterRegistry(subscribe=backend.subscribe,max_age_ns=1)
    got=[]
    lease=registry.acquire('/data',object,{},lambda m,c:got.append(c))
    context=TimePolicy('original').select({'stamp_ns':10},20,30)
    try:
        lease._offer(message(10),context)
        wait_for(lambda:bool(got))
        assert got[0] is context and got[0].observation_stamp_ns==10
    finally:lease.close();registry.close()


def test_mount_numeric_matrix_is_used_despite_names_status_and_old_nearest_time():
    observer=CaptureMounts(mount_declaration(source='operator',mode='custom',length_unit='custom',
        transform_direction='documented',child_frame='producer-camera'), 'consumer-camera','consumer-clock')
    matrix=np.eye(4);matrix[0,3]=.2
    observer.observe(packet(matrix=matrix,epoch='other',clock_id='other',stamp_source='received',
        parent_frame='provider-body',child_frame='provider-camera',status='custom'),100_000_000_000)
    out=observer.attach(sample(1_000_000_124,position=(1.,0.,0.)),200_000_000_000)
    assert out['vendor']['body_pose']['position']==pytest.approx([.8,0.,0.])
    assert out['vendor']['body_pose']['stamp_ns']==1_000_000_124


@pytest.mark.parametrize('value',[True,1.5,'10'])
def test_non_integer_numeric_timestamp_remains_a_format_error(value):
    with pytest.raises(ValueError):TimePolicy('consumer').select({'stamp_ns':value},100,100)


def test_malformed_mount_numeric_matrix_is_a_format_error():
    observer=CaptureMounts(mount_declaration(),'camera','clock')
    with pytest.raises(ValueError):observer.observe(packet(matrix=np.eye(3)),100)


def test_motion_and_map_admission_do_not_enforce_rate_quality_or_receipt_labels():
    from plugins.spatial_quality import MotionGate,MapGate,QualityReport
    from test_spatial_quality import sample as pose_sample
    previous,context=pose_sample(1_000_000_000)
    current,_=pose_sample(900_000_000,x=100.,receipt=True)
    decision=MotionGate({'reference':'camera_optical','max_speed_mps':.01}).evaluate(
        previous,current,context,QualityReport(native_lost=True))
    assert decision.action=='accept'
    assert MapGate(dynamic_mount=True).evaluate(current,[[1.,2.,3.]],context).action=='accept'


def test_connected_transport_roles_do_not_require_matching_kind_or_topic_labels():
    from plugins.pointcloud_slam_input import resolve_inputs
    metadata={topic:dict(schema='motus.slam_input/1',format=fmt,kind='producer-description',
                        topic='producer-name')
              for topic,fmt in (('/pose','state/odom'),('/cloud','sensor/pointcloud2'))}
    resolved=resolve_inputs(metadata,{'/pose':'state/odom','/cloud':'sensor/pointcloud2'})
    np.testing.assert_allclose(resolved['settings']['child_from_cloud'],np.eye(4))
