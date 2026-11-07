"""Both live compute paths receive one explicit model/metric contract."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('open3d');pytest.importorskip('gtsam')
from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
from plugins.pointcloud_slam_rolling import RollingGeometryHistory
from test_pointcloud_slam_input import declarations
from test_pointcloud_slam_plugin import config
from vision_stubs import _FakeExecutor


def scoring(kind='pinhole'):
    return dict(schema='motus.slam_score/1',metric_id='range_stratified_rmse',version=1,
        support_set='target_visible_source',range_definition='sensor_optical_z' if kind=='pinhole' else 'sensor_radial',
        band_edges_m=[2.,4.],required_bands=[0],missing_match_policy='clipped_max_distance',
        visibility={'pixel_bin':4} if kind=='pinhole' else {'azimuth_bin_rad':.1,'elevation_bin_rad':.1})


def options(kind='pinhole'):
    score=scoring(kind)
    return dict(descriptor_voxel_m=.1,min_separation_s=1.,candidate_separation_s=.1,
        max_per_frame=1,max_candidates=3,voxel_m=.08,max_distance_m=.3,
        odom_sigmas=[.1]*6,loop_sigmas=[.05]*6,huber_k=1.,scoring=score,
        revisit={'enabled':False},thresholds=dict(min_fitness=.5,max_range_stratified_rmse_m=.1,
            min_correspondences=3,max_cycle_translation_m=.1,max_cycle_angle_deg=2.,score_contract=deepcopy(score)))


def args(kind='pinhole'):
    meta=declarations();meta['/c']['observation_model']=kind
    if kind=='lidar':
        meta['/c'].update(cloud_semantics='scan',timing_model='deskewed',observation=dict(
            axis_convention='x_forward_y_left_z_up',azimuth_min_rad=-np.pi,azimuth_span_rad=2*np.pi,
            elevation_min_rad=-.5,elevation_max_rad=.5,min_range_m=.1,max_range_m=20.))
    return dict(action='start',input_topics=['/o','/c'],input_bindings={'/o':'state/odom','/c':'sensor/pointcloud2'},
        input_metadata=meta,odom_interfaces={'/o':{'schema':'motus.odom/1','slam_pose':{'available':True,'profile':'motus_slam/1'}}},
        camera_info={'/c':dict(schema='motus.camera/1',id='camera',topic='/c',format='sensor/pointcloud2',
            width=640,height=480,K=[400.,0.,320.,0.,400.,240.,0.,0.,1.],D=[],distortion_model='pinhole')})


@pytest.mark.parametrize('kind',['pinhole','lidar'])
def test_new_geometry_uses_declared_model_and_archives_same_strategy(kind):
    executor=_FakeExecutor()
    plugin=PointCloudSlamPlugin({**config(),'input_policy':'experimental','preview_enabled':True,
        'preview_backend':'geometry_experimental','preview_geometry':options(kind)},executor)
    try:
        result=plugin.dispatch('pointcloud_slam',args(kind))
        assert result['state']=='running',result
        strategy=plugin._node._loop_search.strategy_snapshot()
        assert strategy['resolved_observation']['kind']==kind
        assert strategy['resolved_scoring']==scoring(kind)
        assert plugin._session_settings['resolved_observation']==strategy['resolved_observation']
        assert plugin._session_settings['resolved_scoring']==strategy['resolved_scoring']
        assert plugin._session_settings['resolved_inputs']['/c']['source_id']=='camera'
        info=plugin.dispatch('pointcloud_slam',{'action':'info'})
        assert info['resolved_observation']['kind']==kind
        assert info['compatibility']==[]
    finally: plugin.shutdown()


def test_paged_and_unpaged_share_one_resolved_strategy(tmp_path,monkeypatch):
    import plugins.pointcloud_slam_plugin as module
    monkeypatch.setattr(module,'_ARCHIVE_DIR',str(tmp_path))
    cfg={**config(),'input_policy':'experimental','preview_enabled':True,
        'preview_backend':'geometry_experimental','preview_geometry':options(),
        'persistent_history':{'max_disk_bytes':10000000,'reserve_bytes':0}}
    plugin=PointCloudSlamPlugin(cfg,_FakeExecutor())
    try:
        assert plugin.dispatch('pointcloud_slam',args())['state']=='ready'
        result=plugin.dispatch('pointcloud_slam',{'action':'start_mapping','map_name':'test'})
        assert result['state']=='running',result
        preview=plugin._node._preview
        worker=RollingGeometryHistory(preview['geometry'],window_frames=10,window_points=1000)
        assert worker.strategy_snapshot()['resolved_observation']==plugin._session_settings['resolved_observation']
        assert worker.strategy_snapshot()['resolved_scoring']==scoring()
    finally: plugin.shutdown()


def test_dynamic_transform_declared_on_pose_port_reaches_model_without_identity():
    request=args()
    extrinsic=request['input_metadata']['/c'].pop('extrinsic')
    extrinsic.update(mode='dynamic',source='tf');extrinsic.pop('matrix')
    request['input_metadata']['/o']['extrinsic']=extrinsic
    plugin=PointCloudSlamPlugin({**config(),'input_policy':'experimental','preview_enabled':True,
        'preview_backend':'geometry_experimental','preview_geometry':options()},_FakeExecutor())
    try:
        result=plugin.dispatch('pointcloud_slam',request)
        assert result['state']=='running',result
        assert plugin._node.stream.extrinsic is None
        assert plugin._strategy['resolved_observation']['sensor_from_child'] is None
        assert plugin._strategy['resolved_observation']['transform_mode']=='per_capture'
    finally: plugin.shutdown()


def test_disabled_geometry_preview_does_not_require_camera_or_scoring():
    request=args();request.pop('camera_info')
    plugin=PointCloudSlamPlugin({**config(),'input_policy':'experimental','preview_enabled':False,
        'preview_backend':'geometry_experimental'},_FakeExecutor())
    try:
        result=plugin.dispatch('pointcloud_slam',request)
        assert result['state']=='running',result
        assert plugin._node._preview is None
        assert plugin._node._loop_search is None
    finally: plugin.shutdown()


@pytest.mark.parametrize('fault',['missing_camera','missing_model_parameters','legacy_threshold','no_scoring_profile'])
def test_declared_geometry_does_not_fall_back_when_model_or_contract_is_missing(fault):
    kind='lidar' if fault=='missing_model_parameters' else 'pinhole'
    request=args(kind);opts=options(kind)
    if fault=='missing_camera': request.pop('camera_info')
    if fault=='missing_model_parameters': request['input_metadata']['/c'].pop('observation')
    if fault=='legacy_threshold': opts['thresholds']['max_forward_depth_score_m']=.1
    if fault=='no_scoring_profile': opts.pop('scoring')
    plugin=PointCloudSlamPlugin({**config(),'input_policy':'experimental','preview_enabled':True,
        'preview_backend':'geometry_experimental','preview_geometry':opts},_FakeExecutor())
    result=plugin.dispatch('pointcloud_slam',request)
    assert result['state']=='error',result
    assert plugin._node is None
