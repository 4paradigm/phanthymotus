"""New map metadata stays immutable; legacy maps remain independently viewable."""
from copy import deepcopy
import json

import numpy as np
import pytest

from plugins import pointcloud_slam_library as library
from plugins.pointcloud_slam_paged_archive import load_paged_archive
from test_pointcloud_slam_library import saved
from test_pointcloud_slam_paged_store import add, store
from vision_stubs import _FakeExecutor


def evidence():
    return dict(resolved_inputs={'/cloud': {'source_id':'camera','clock': {'verified':False}}},
                resolved_observation={'schema':'motus.slam_observation/1','kind':'pinhole'},
                resolved_scoring={'schema':'motus.slam_score/1','metric_id':'range_stratified_rmse','version':1},
                algorithm_profile={'observation':'pinhole','legacy':False},
                resource_profile={'max_frames':3,'max_points':30,'max_cloud_points':10},
                map_plane='xy')


def test_rename_changes_only_display_name_and_preserves_revision_tags(tmp_path, saved):
    map_id, _, _ = saved
    tag=library.create_tag(tmp_path,map_id,1,'地点',1,2,segment_id='a')
    before=deepcopy(library.map_entries(tmp_path)[0])
    renamed=library.rename_map(tmp_path,map_id,' 新地图 ')
    after=library.map_entries(tmp_path)[0]
    assert renamed['id']==map_id and renamed['name']=='新地图'
    assert after=={**before,'name':'新地图'}
    assert library.list_tags(tmp_path,map_id,1)==[tag]


def test_rename_conflicts_and_failed_write_leave_original_manifest(tmp_path, saved, monkeypatch):
    map_id, save, _=saved
    save('其他',request='other')
    path=tmp_path/'maps'/map_id/'manifest.json'
    original=path.read_bytes()
    for name in ('其他','../bad',''):
        with pytest.raises(ValueError): library.rename_map(tmp_path,map_id,name)
        assert path.read_bytes()==original
    def fail(*args,**kwargs): raise OSError('write failed')
    monkeypatch.setattr(library,'_write_json',fail)
    with pytest.raises(OSError): library.rename_map(tmp_path,map_id,'新名字')
    assert path.read_bytes()==original


def test_paged_v2_snapshot_keeps_resolved_facts_and_legacy_read_is_unknown(tmp_path):
    from plugins.pointcloud_slam_paged_archive import save_paged_archive
    frames=store(tmp_path/'sessions');add(frames,0)
    cfg={'cloud_frame':'camera','child_frame':'body','clock_id':'clock',
         'child_from_cloud':np.eye(4).tolist(),
         'persistent_history': {'max_disk_bytes':10_000_000,'reserve_bytes':0},**evidence()}
    path=tmp_path/'map.json'
    save_paged_archive(path,frames,{},cfg)
    data=json.loads(path.read_text())
    assert data['schema']=='motus.experimental-slam-paged/2'
    assert data['resolved']['resolved_inputs']==cfg['resolved_inputs']
    assert data['resolved']['resolved_scoring']==cfg['resolved_scoring']
    viewed,_=load_paged_archive(path,{},max_frames=3,max_points=30,read_only=True)
    assert viewed.archive_metadata['map_plane']=='xy'
    assert viewed.archive_metadata['resolved_inputs']['/cloud']['clock']['verified'] is False
    data['schema']='motus.experimental-slam-paged/1';data.pop('resolved')
    path.write_text(json.dumps(data))
    old,_=load_paged_archive(path,{},max_frames=3,max_points=30,read_only=True)
    assert old.archive_metadata['resolved_inputs'] is None
    assert old.archive_metadata['evidence']['source']=='legacy_archive'
    assert old.archive_metadata['map_plane'] is None


def test_view_saved_map_requires_no_live_input_settings_and_uses_archived_identity(tmp_path,saved,monkeypatch):
    import plugins.pointcloud_slam_plugin as plugin_module
    map_id,_,cfg=saved
    monkeypatch.setattr(plugin_module,'_ARCHIVE_DIR',str(tmp_path))
    archive=tmp_path/'maps'/map_id/'1.json'
    old=json.loads(archive.read_text());old['schema']='motus.experimental-slam-paged/1';old.pop('resolved')
    archive.write_text(json.dumps(old))
    plugin=plugin_module.PointCloudSlamPlugin({'persistent_history':cfg['persistent_history']},_FakeExecutor())
    try:
        result=plugin.dispatch('pointcloud_slam',{'action':'select_map','map_id':map_id})
        assert result['state']=='map_selected',result
        assert result['tracking'] is False
        assert result['archive_metadata']['evidence']['source']=='legacy_archive'
        output=result['topic_out'][0]
        view=output['metadata']['map_view']
        assert (view['map_id'],view['revision'],view['segment_id'],view['frame_id'])==(map_id,1,'a','odom_archive')
        assert view['plane'] is None
        closed=plugin.dispatch('pointcloud_slam',{'action':'close_map_view','map_id':map_id})
        assert closed['closed_topics'] and output['topic'] in closed['closed_topics']
    finally: plugin.shutdown()


def test_map_tool_declares_generic_resources_actions_and_owner_independent_capabilities():
    from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
    plugin=PointCloudSlamPlugin({'persistent_history':{'max_disk_bytes':10_000_000,'reserve_bytes':0}},_FakeExecutor())
    tool=plugin.get_tools()[0];schema=tool['inputSchema']
    assert 'rename_map' in schema['properties']['action']['enum']
    assert schema['properties']['map_id']['x-resource-options']==dict(action='list_maps',items='maps',value='id',label='name')
    assert schema['x-map-manager']['schema']=='motus.map-manager/1'
    assert schema['x-action-lifecycle']['actions']['select_map']['background'] is True
    assert schema['x-stop']['timeout_s']==300


def dynamic_configuration():
    return dict(cloud_frame='camera', child_frame='body', clock_id='clock',
                extrinsic_mode='dynamic', child_from_cloud=None,
                persistent_history={'max_disk_bytes':10_000_000,'reserve_bytes':0},
                **evidence())


def capture_context(ms, offset):
    matrix=np.eye(4);matrix[0,3]=offset
    return dict(child_from_cloud=matrix.tolist(),
                extrinsic=dict(source='tf',version='mount/1',parent_frame='body',
                    child_frame='camera',stamp_ns=ms*1_000_000,clock_id='clock',
                    clock_epoch='epoch/1',matrix=matrix.tolist()),
                measurement=dict(schema='motus.slam_frame/1',stamp_ns=ms*1_000_000,
                    clock_id='clock',clock_epoch='epoch/1',source_generation='cloud/1'))


@pytest.mark.parametrize('paged',[False,True])
@pytest.mark.parametrize('read_only',[False,True])
def test_dynamic_archive_roundtrip_retains_each_capture_context(tmp_path,paged,read_only):
    from plugins.pointcloud_slam_archive import save_archive,load_archive
    from plugins.pointcloud_slam_keyframes import KeyframeStore
    from plugins.pointcloud_slam_odom import decode_slam_odom
    from plugins.pointcloud_slam_paged_archive import save_paged_archive,inspect_paged_archive
    from test_pointcloud_slam_stream import odom
    frames=store(tmp_path/'sessions') if paged else KeyframeStore(max_frames=3,max_points=30)
    contexts=[capture_context(1000,0.2),capture_context(1100,0.5)]
    for i,context in enumerate(contexts):
        pose=decode_slam_odom(odom(1000+100*i,x=i))
        frames.add(pose,[[1.,2.,3.]],cloud_stamp_ns=pose.stamp_ms*1_000_000,
                   capture_context=context)
    path=tmp_path/'dynamic.json';cfg=dynamic_configuration()
    if paged: save_paged_archive(path,frames,{},cfg)
    else: save_archive(path,frames,[],{},cfg)
    data=json.loads(path.read_text())
    assert data['configuration']['extrinsic_mode']=='dynamic'
    assert data['configuration']['child_from_cloud'] is None
    assert data['frames'][0]['capture_context']==contexts[0]
    if paged:
        assert inspect_paged_archive(path)[0]['child_from_cloud'] is None
        restored,_=load_paged_archive(path,{} if read_only else cfg,max_frames=3,max_points=30,read_only=read_only)
        assert [r['capture_context'] for r in restored.records]==contexts
    else:
        restored,_,_=load_archive(path,{} if read_only else cfg,max_frames=3,max_points=30,read_only=read_only)
    restored_frames=restored.segments[0].frames
    assert [f.capture_context for f in restored_frames]==contexts
    assert [f.cloud_stamp_ns for f in restored_frames]==[1_000_000_000,1_100_000_000]
    assert restored.archive_metadata['resolved_observation']==cfg['resolved_observation']
    for frame in restored_frames:
        np.testing.assert_array_equal(np.asarray(frame.points),[[1.,2.,3.]])
    contexts[0]['child_from_cloud'][0][3]=99
    assert restored_frames[0].capture_context['child_from_cloud'][0][3]==0.2
    assert restored.requires_new_segment


@pytest.mark.parametrize('paged',[False,True])
def test_dynamic_archive_missing_capture_keeps_unknown_transform(tmp_path,paged):
    from plugins.pointcloud_slam_archive import save_archive,load_archive
    from plugins.pointcloud_slam_keyframes import KeyframeStore
    from plugins.pointcloud_slam_odom import decode_slam_odom
    from plugins.pointcloud_slam_paged_archive import save_paged_archive
    from test_pointcloud_slam_stream import odom
    frames=store(tmp_path/'sessions') if paged else KeyframeStore(max_frames=3,max_points=30)
    frames.add(decode_slam_odom(odom(1000)),[[1,2,3]],cloud_stamp_ns=1_000_000_000)
    cfg=dynamic_configuration();path=tmp_path/'unknown.json'
    if paged: save_paged_archive(path,frames,{},cfg)
    else: save_archive(path,frames,[],{},cfg)
    data=json.loads(path.read_text())
    assert data['configuration']['child_from_cloud'] is None
    assert 'capture_context' not in data['frames'][0]
    if paged: restored,_=load_paged_archive(path,{},max_frames=3,max_points=30,read_only=True)
    else: restored,_,_=load_archive(path,{},max_frames=3,max_points=30,read_only=True)
    assert restored.segments[0].frames[0].capture_context is None


def test_dynamic_archive_configuration_rejects_a_startup_matrix():
    from plugins.pointcloud_slam_archive import _configuration
    cfg=dynamic_configuration();cfg['child_from_cloud']=np.eye(4).tolist()
    with pytest.raises(ValueError,match='dynamic'):
        _configuration(cfg)


@pytest.mark.parametrize('legacy',[False,True])
def test_restored_live_session_publishes_archived_history_evidence(tmp_path,monkeypatch,legacy):
    pytest.importorskip('open3d');pytest.importorskip('gtsam')
    import plugins.pointcloud_slam_plugin as module
    from plugins.pointcloud_slam_paged_archive import save_paged_archive
    from plugins.pointcloud_slam_ros import SlamInputNode
    from test_pointcloud_slam_plugin import config
    from test_pointcloud_slam_strategy import options
    from test_pointcloud_slam_preview import unpack
    monkeypatch.setattr(module,'_ARCHIVE_DIR',str(tmp_path))
    monkeypatch.setattr(SlamInputNode,'start_preview',lambda self:None)
    source=store(tmp_path/'sessions');add(source,0)
    cfg={**config(),'preview_backend':'geometry_experimental','preview_geometry':options(),
         'persistent_history':{'max_disk_bytes':10_000_000,'reserve_bytes':0}}
    path=tmp_path/'history.json'
    save_paged_archive(path,source,{},{**cfg,'clock_relation_verified':False,'extrinsic_calibrated':False})
    if legacy:
        data=json.loads(path.read_text());data['schema']='motus.experimental-slam-paged/1';data.pop('resolved')
        path.write_text(json.dumps(data))
    plugin=module.PointCloudSlamPlugin({**cfg,'preview_enabled':True,
        'restore_archive':True,'archive_name':'history'},_FakeExecutor())
    try:
        result=plugin.dispatch('pointcloud_slam',dict(action='start',input_topics=['/o','/c'],
            input_bindings={'/o':'state/odom','/c':'sensor/pointcloud2'},
            odom_interfaces={'/o':{'schema':'motus.odom/1','slam_pose':{'available':True,'profile':'motus_slam/1'}}}))
        assert result['state']=='ready',result
        result=plugin.dispatch('pointcloud_slam',dict(action='start_mapping',map_name='恢复'))
        assert result['state']=='running',result
        node=plugin._node;node.publish_preview_once()
        history_topic=node.history_topics()[0]['topic']
        publisher=next(p for p in node.publishers if p.topic==history_topic)
        metadata,_=unpack(publisher.messages[-1])
        expected=None if legacy else False
        assert metadata['input_validation']['clock_relation_verified'] is expected
        assert metadata['input_validation']['extrinsic_calibrated'] is expected
        assert node._preview['input_validation']['clock_relation_verified'] is True
    finally: plugin.shutdown()
