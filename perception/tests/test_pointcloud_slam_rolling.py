import numpy as np
import pytest
from plugins.pointcloud_slam_rolling import RollingGeometryHistory
from test_pointcloud_slam_paged_store import store, add
from plugins.pointcloud_slam_stream import _matrix


class FakeSearch:
    sizes = []
    def __init__(self, **kw): pass
    def update(self, segment, graph, **kw):
        self.sizes.append(len(segment.frames))
        # A changed local reference tests full SE(3) overlap alignment.
        rotate = np.array([[0,-1,0,3],[1,0,0,4],[0,0,1,0],[0,0,0,1.]])
        poses = [rotate @ p for p in graph['poses']]
        return {'poses':poses,'passed_candidates':[], 'candidate_attempts':0,
                'descriptor_skips':0, 'descriptor_unindexed':0,
                'timings_ms':{'descriptor':0.,'registration_and_graph':0.}}


def test_three_batches_keep_full_history_and_overlap_coordinates(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_rolling as module
    monkeypatch.setattr(module, 'ExperimentalLoopSearch', FakeSearch)
    FakeSearch.sizes=[]
    s=store(tmp_path)
    for i in range(605): add(s,i)
    worker=RollingGeometryHistory({}, window_frames=300, window_points=2000)
    result=worker.update(s.segments[0], max_points=100)
    assert max(FakeSearch.sizes)<=300
    assert result['submaps']==3
    assert len(result['poses'])==605 and len(result['after_map'])==100
    for frame,pose in zip(s.segments[0].frames,result['poses']):
        np.testing.assert_allclose(pose,_matrix(frame.pose),atol=1e-9)
    assert result['after_map'][0,0]==pytest.approx(1)
    assert result['after_map'][-1,0]==pytest.approx(605)
    old=result['poses'][0].copy()
    add(s,605)
    result=worker.update(s.segments[0],max_points=100)
    np.testing.assert_array_equal(result['poses'][0],old)
    np.testing.assert_allclose(result['map_from_odom'],np.eye(4),atol=1e-9)


def test_window_point_budget_and_real_segment_separation(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_rolling as module
    monkeypatch.setattr(module,'ExperimentalLoopSearch',FakeSearch)
    FakeSearch.sizes=[];s=store(tmp_path)
    for i in range(9): add(s,i)
    w=RollingGeometryHistory({},window_frames=300,window_points=12)
    w.update(s.segments[0],max_points=10)
    assert max(FakeSearch.sizes)==3
    add(s,10,segment='b')
    with pytest.raises(ValueError): w.update(s.segments[1],max_points=10)
