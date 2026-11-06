import numpy as np
import pytest

pytest.importorskip('open3d')
from tools.pointcloud_slam_visible_overlap import ReferenceWindow, VisibilityView, split_decision


def test_occluded_points_reduce_coverage_without_contributing_visible_error():
    p = np.array([[0., 0., 1.], [.4, .2, 3.], [-.6, .4, 5.]])
    camera = (160, 120, 100., 100., 80., 60.)
    target = VisibilityView(p, camera, np.eye(4), .01)
    source = VisibilityView(np.vstack((p, 2*p)), camera, np.eye(4), .01)
    s = source.score(target, np.eye(4), .3)
    assert s['coverage'] == .5 and s['counts'] == [1, 1, 1]
    assert s['error_m'] == pytest.approx(0.)
    assert split_decision(s,.6,.09,1) == dict(coverage=False, range_support=True, visible_error=True)


def test_free_space_disagreement_is_scored_and_empty_band_fails():
    cam = (160, 120, 100., 100., 80., 60.)
    query = VisibilityView(np.array([[0.,0.,1.]]),cam,np.eye(4),.01)
    target = VisibilityView(np.array([[0.,0.,2.]]),cam,np.eye(4),.01)
    s = query.score(target,np.eye(4),.3)
    assert s['coverage'] == 1 and s['fitness'] == 0
    assert split_decision(s,.3,.09,1) == dict(coverage=True, range_support=False, visible_error=False)


def test_reference_window_counts_repeated_observations_once_in_anchor_frame():
    camera = (160, 120, 100., 100., 80., 60.)
    points = np.array([[0., 0., 1.], [.4, .2, 3.], [-.6, .4, 5.]])
    poses = [np.eye(4) for _ in range(3)]
    poses[1][0, 3] = .1
    poses[2][0, 3] = -.1
    clouds = [points - pose[:3, 3] for pose in poses]
    views = [VisibilityView(cloud, camera, np.eye(4), .08) for cloud in clouds]
    window = ReferenceWindow(clouds, views, poses, .08)
    assert len(window.points) == 3
    for source, target in [(window, views[0]), (views[0], window)]:
        scores = source.score(target, np.eye(4), .30)
        assert scores['counts'] == [1, 1, 1]
        assert scores['coverage'] == 1 and scores['error_m'] == pytest.approx(0)
