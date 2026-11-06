from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('open3d')
from plugins.pointcloud_slam_revisit import RevisitSearch


@pytest.mark.parametrize('break_support', [False, True])
def test_only_two_independent_moving_holdouts_commit_the_seed_edge(break_support):
    # A physical departure and return, followed by a repeated stationary pose
    # and two translated observations of the same scene.
    xs = [0., .12, .24, 2., 0., 0., .12, .24]
    poses = [np.eye(4) for _ in xs]
    for pose in poses[4:]:
        pose[0, 3] = .02
    for x, pose in zip(xs, poses):
        pose[0, 3] += x
    rng = np.random.default_rng(272)
    surfaces = []
    for depth, u in [(1., -.45), (3., -.1), (5., .3)]:
        angles = rng.uniform([u, -.25], [u+.2, .25], (1000, 2))
        z = depth + .1*angles[:, 1]
        surfaces.append(np.column_stack((angles*z[:, None], z)))
    points = np.concatenate(surfaces)
    frames = [SimpleNamespace(points=points-[x, 0, 0], cloud_stamp_ns=i*5_000_000_000)
              for i, x in enumerate(xs)]
    if break_support:
        frames[6].points = frames[6].points + [0, 5, 0]
    opts = dict(voxel_m=.04, max_distance_m=.3, min_separation_s=10,
                thresholds=dict(max_forward_depth_score_m=.09, min_fitness=.6,
                                min_correspondences=30, max_cycle_translation_m=.03,
                                max_cycle_angle_deg=2))
    search = RevisitSearch((320,240,150,150,160,120), np.eye(4), opts)
    for i in (4, 5, 6, 7):
        factor = search.update(frames[:i+1], poses[:i+1], [0] if i==4 else [])
        if i<7 or break_support:
            assert factor is None
        else:
            assert (factor['source'], factor['target']) == (4, 0)
            assert factor['confirmation_stamps_ns'] == [frames[6].cloud_stamp_ns, frames[7].cloud_stamp_ns]
    if break_support:
        assert search.last_seed_ns is None
